// Portable audio frontend core — comprehensive-plan.md §8 ("the golden-vector test").
//
// This file is compiled TWICE from the same source, unmodified:
//   1. Into firmware, as part of the ESP32-S3 build (firmware/main/CMakeLists.txt),
//      linked against the SIGNAL kernels already built into the esp-tflite-micro
//      component (managed_components/espressif__esp-tflite-micro/signal/src).
//   2. Into a native macOS shared library (tools/build_native_frontend.sh), which
//      training/src/echo_sentry_train/features.py loads via ctypes.
//
// Being the literal same source file on both sides is the anti-skew strategy: instead
// of two independent implementations that are *supposed* to agree (the original design
// risk this project flagged early on), there is exactly one implementation, compiled
// for two targets. See tools/gen_frontend_tables.cc for where the constant tables
// (Hann window, mel filterbank weights, PCAN gain LUT) it depends on come from.
//
// Sequence (ported from tflite-micro's python/tflite_micro/signal/examples/
// micro_speech/audio_preprocessor.py's _GenerateFeature.generate_feature_for_frame,
// but calling the underlying signal/src primitives directly rather than building a
// TFLite graph — see the module comment in gen_frontend_tables.cc for why):
//   window -> fft_auto_scale -> rfft -> energy -> filterbank -> filterbank_sqrt
//   -> filterbank_spectral_subtraction -> [pcan] -> filterbank_log

#pragma once

// This header is included from plain C translation units too (firmware/main's C
// files call only the extern "C" API below) — everything C++-only is guarded so
// #include "frontend_core.h" works from either language.
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
namespace echo_sentry {

// Opaque handle; owns the RFFT scratch state and the spectral-subtraction noise
// estimate, both of which persist across frames within one continuous audio stream —
// see comprehensive-plan.md §8 on why this must not be reset per-clip on-device.
struct FrontendState;

// Allocates and initializes a FrontendState. Returns nullptr on failure.
FrontendState* FrontendCreate();

void FrontendDestroy(FrontendState* state);

// Resets the persistent noise-estimate (spectral subtraction) state without
// reallocating. Call this at boot and nowhere else in production — comprehensive-
// plan.md's continuous-monitoring design deliberately does NOT reset per clip.
void FrontendReset(FrontendState* state);

// Processes one window's worth of audio (ECHO_SENTRY_WINDOW_SIZE_MS at
// ECHO_SENTRY_SAMPLE_RATE_HZ -> ECHO_SENTRY_MEL_CHANNELS-length int16 frame).
// `audio_frame` must have exactly kWindowSampleCount int16 samples;
// `out_frame` must have space for ECHO_SENTRY_MEL_CHANNELS int16 values.
void FrontendProcessFrame(FrontendState* state, const int16_t* audio_frame,
                          int16_t* out_frame);

}  // namespace echo_sentry
#endif  // __cplusplus

// extern "C" API — used by both firmware/main's plain-C files and the ctypes bridge
// for the native training-side library.
#ifdef __cplusplus
extern "C" {
#endif

void* echo_sentry_frontend_create(void);
void echo_sentry_frontend_destroy(void* state);
void echo_sentry_frontend_reset(void* state);
void echo_sentry_frontend_process_frame(void* state, const int16_t* audio_frame,
                                        int16_t* out_frame);

// Fixed shape constants the caller needs but that live in gen/frontend_config.h,
// re-exported here so ctypes callers don't need to parse a C header.
int echo_sentry_frontend_window_sample_count(void);
int echo_sentry_frontend_mel_channels(void);

#ifdef __cplusplus
}  // extern "C"
#endif
