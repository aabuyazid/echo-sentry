// See frontend_core.h for the design rationale (one source, two compile targets).
//
// Sequence and exact parameter wiring were derived by reading the real vendored
// sources under managed_components/espressif__esp-tflite-micro/signal/{src,micro/
// kernels}/ and tensorflow/tflite-micro's python/tflite_micro/signal/examples/
// micro_speech/audio_preprocessor.py (fetched from
// github.com/tensorflow/tflite-micro) — not reconstructed from memory. Two details
// worth flagging for whoever touches this next:
//
//  * PCAN's `snr_shift` argument is NOT derived from config at runtime — the
//    reference Python computes a value only to validate it's >= 1, then calls the
//    op with a hardcoded 6 (== kPcanOutputBits in pcan_argc_fixed.h, the fixed
//    number of fractional bits PCAN's output always has). Do not "fix" this to use
//    the computed value; that would silently change every downstream number.
//  * FilterbankConfig has `output_scale`/`input_correction_bits` fields that
//    FilterbankAccumulateChannels() never reads (confirmed against its actual body,
//    not just the header) — they're left zeroed here deliberately, not omitted by
//    oversight.

#include "frontend_core.h"

#include <cstring>
#include <new>

#include "gen/frontend_config.h"
#include "gen/frontend_tables.h"

#include "signal/src/energy.h"
#include "signal/src/fft_auto_scale.h"
#include "signal/src/filter_bank.h"
#include "signal/src/filter_bank_log.h"
#include "signal/src/filter_bank_spectral_subtraction.h"
#include "signal/src/filter_bank_square_root.h"
#include "signal/src/pcan_argc_fixed.h"
#include "signal/src/rfft.h"
#include "signal/src/window.h"

namespace echo_sentry {

namespace {

// window.h / rfft.h declare `tflm_signal` at global scope; every other signal/src
// header wraps it in `tflite::tflm_signal`. Genuinely two different namespaces
// upstream (confirmed by reading both), not a typo here.
namespace sig = ::tflm_signal;
namespace ts = ::tflite::tflm_signal;

constexpr int kWindowSampleCount =
    ECHO_SENTRY_WINDOW_SIZE_MS * ECHO_SENTRY_SAMPLE_RATE_HZ / 1000;
// Must be a power of 2 — enforced by config/frontend.yaml's comment and by
// tools/gen_config.py, since we skip the general zero-pad-to-next-pow2 case the
// TFLM kernel wrapper handles and require window_size_ms to land exactly on one.
constexpr int kFftLength = kWindowSampleCount;
constexpr int kSpectrumSize = kFftLength / 2 + 1;
constexpr int kNumChannels = ECHO_SENTRY_MEL_CHANNELS;

// PCAN's output always has this many fractional bits — hardcoded upstream
// (pcan_argc_fixed.h's kPcanOutputBits), not derived from our config. See the file
// header comment above.
constexpr int kPcanSnrShift = 6;

}  // namespace

struct FrontendState {
  // Recomputed every frame.
  int16_t windowed[kWindowSampleCount];
  int16_t windowed_scaled[kWindowSampleCount];
  int16_t fft_input[kFftLength];
  Complex<int16_t> fft_output[kSpectrumSize];
  uint32_t energy[kSpectrumSize];
  uint64_t filterbank_work[kNumChannels + 1];
  uint32_t filterbank_sqrt[kNumChannels];
  uint32_t spectral_sub_output[kNumChannels];
  uint32_t pcan_output[kNumChannels];

  // Persistent across frames within one continuous stream.
  void* rfft_state;
  ts::FilterbankConfig fb_config;
  ts::SpectralSubtractionConfig ss_config;
  uint32_t noise_estimate[kNumChannels];
};

FrontendState* FrontendCreate() {
  auto* state = new (std::nothrow) FrontendState();
  if (state == nullptr) return nullptr;

  size_t rfft_state_size = sig::RfftInt16GetNeededMemory(kFftLength);
  auto* rfft_buffer = new (std::nothrow) uint8_t[rfft_state_size];
  if (rfft_buffer == nullptr) {
    delete state;
    return nullptr;
  }
  state->rfft_state = sig::RfftInt16Init(kFftLength, rfft_buffer, rfft_state_size);

  state->fb_config.num_channels = kNumChannels;
  state->fb_config.channel_frequency_starts = kEchoSentryFilterbankChannelFrequencyStarts;
  state->fb_config.channel_weight_starts = kEchoSentryFilterbankChannelWeightStarts;
  state->fb_config.channel_widths = kEchoSentryFilterbankChannelWidths;
  state->fb_config.weights = kEchoSentryFilterbankWeights;
  state->fb_config.unweights = kEchoSentryFilterbankUnweights;
  // Unused by FilterbankAccumulateChannels — see the file header comment above.
  state->fb_config.output_scale = 0;
  state->fb_config.input_correction_bits = 0;

  const int spectral_subtraction_bits = ECHO_SENTRY_FB_SPECTRAL_SUBTRACTION_BITS;
  const uint32_t smoothing =
      static_cast<uint32_t>(kEchoSentryEvenSmoothing * (1 << spectral_subtraction_bits));
  const uint32_t alt_smoothing =
      static_cast<uint32_t>(kEchoSentryOddSmoothing * (1 << spectral_subtraction_bits));
  state->ss_config.num_channels = kNumChannels;
  state->ss_config.smoothing = smoothing;
  state->ss_config.one_minus_smoothing = (1u << spectral_subtraction_bits) - smoothing;
  state->ss_config.alternate_smoothing = alt_smoothing;
  state->ss_config.alternate_one_minus_smoothing =
      (1u << spectral_subtraction_bits) - alt_smoothing;
  state->ss_config.smoothing_bits = ECHO_SENTRY_FB_SMOOTHING_BITS;
  state->ss_config.spectral_subtraction_bits = spectral_subtraction_bits;
  state->ss_config.min_signal_remaining = static_cast<uint32_t>(
      kEchoSentryMinSignalRemaining * (1 << spectral_subtraction_bits));
  state->ss_config.clamping = kEchoSentryFilterBankClamping;

  FrontendReset(state);
  return state;
}

void FrontendDestroy(FrontendState* state) {
  if (state == nullptr) return;
  delete[] static_cast<uint8_t*>(state->rfft_state);  // matches the RfftInit buffer
  delete state;
}

void FrontendReset(FrontendState* state) {
  std::memset(state->noise_estimate, 0, sizeof(state->noise_estimate));
}

void FrontendProcessFrame(FrontendState* state, const int16_t* audio_frame,
                          int16_t* out_frame) {
  sig::ApplyWindow(audio_frame, kEchoSentryHannWindow, kWindowSampleCount,
                   ECHO_SENTRY_WINDOW_SCALING_BITS, state->windowed);

  const int scaling_shift = ts::FftAutoScale(state->windowed, kWindowSampleCount,
                                             state->windowed_scaled);

  std::memcpy(state->fft_input, state->windowed_scaled,
             sizeof(int16_t) * kWindowSampleCount);
  if (kFftLength > kWindowSampleCount) {
    std::memset(state->fft_input + kWindowSampleCount, 0,
               sizeof(int16_t) * (kFftLength - kWindowSampleCount));
  }

  sig::RfftInt16Apply(state->rfft_state, state->fft_input, state->fft_output);

  std::memset(state->energy, 0, sizeof(state->energy));
  ts::SpectrumToEnergy(state->fft_output, ECHO_SENTRY_FB_START_INDEX,
                       ECHO_SENTRY_FB_END_INDEX, state->energy);

  ts::FilterbankAccumulateChannels(&state->fb_config, state->energy,
                                   state->filterbank_work);
  // Element 0 is scratch (see filter_bank.cc's own comment) — the caller discards it.
  const uint64_t* filterbank_output = state->filterbank_work + 1;

  ts::FilterbankSqrt(filterbank_output, kNumChannels, scaling_shift,
                     state->filterbank_sqrt);

  ts::FilterbankSpectralSubtraction(&state->ss_config, state->filterbank_sqrt,
                                    state->spectral_sub_output, state->noise_estimate);

  const uint32_t* pre_log_input = state->spectral_sub_output;
  if (kEchoSentryEnablePcan) {
    std::memcpy(state->pcan_output, state->spectral_sub_output,
               sizeof(state->pcan_output));
    ts::ApplyPcanAutoGainControlFixed(kEchoSentryPcanGainLut, kPcanSnrShift,
                                      state->noise_estimate, state->pcan_output,
                                      kNumChannels);
    pre_log_input = state->pcan_output;
  }

  ts::FilterbankLog(pre_log_input, kNumChannels,
                    1 << ECHO_SENTRY_FB_POST_SCALING_BITS, ECHO_SENTRY_CORRECTION_BITS,
                    out_frame);
}

}  // namespace echo_sentry

// ---- extern "C" bridge -----------------------------------------------------------

extern "C" {

void* echo_sentry_frontend_create(void) {
  return echo_sentry::FrontendCreate();
}

void echo_sentry_frontend_destroy(void* state) {
  echo_sentry::FrontendDestroy(static_cast<echo_sentry::FrontendState*>(state));
}

void echo_sentry_frontend_reset(void* state) {
  echo_sentry::FrontendReset(static_cast<echo_sentry::FrontendState*>(state));
}

void echo_sentry_frontend_process_frame(void* state, const int16_t* audio_frame,
                                        int16_t* out_frame) {
  echo_sentry::FrontendProcessFrame(static_cast<echo_sentry::FrontendState*>(state),
                                    audio_frame, out_frame);
}

int echo_sentry_frontend_window_sample_count(void) {
  return ECHO_SENTRY_WINDOW_SIZE_MS * ECHO_SENTRY_SAMPLE_RATE_HZ / 1000;
}

int echo_sentry_frontend_mel_channels(void) { return ECHO_SENTRY_MEL_CHANNELS; }

}  // extern "C"
