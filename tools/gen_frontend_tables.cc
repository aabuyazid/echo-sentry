// Generates the constant lookup tables the SIGNAL-ops audio frontend needs
// (Hann window, mel filterbank weights, PCAN gain LUT) — comprehensive-plan.md §8.
//
// Ported from the reference design-time algorithms in
// tensorflow/tflite-micro's python/tflite_micro/signal/{ops/filter_bank_ops.py,
// ops/window_op.py, utils/{freq_to_mel_wrapper,wide_dynamic_func_lut_wrapper}.cc} —
// fetched and read directly from source (not reimplemented from memory) specifically
// because those files' own comments explain that a pure-Python/NumPy port introduces
// small floating-point differences vs. the "Speech Micro" C `float` (32-bit)
// arithmetic this project must match bit-for-bit on-device. FreqToMel and
// WideDynamicFuncLut below are verbatim copies of that reference C++; only the
// call-site plumbing (reading config, emitting a C header instead of a Python list)
// is original.
//
// This is a HOST-side build tool (compiled for and run on the Mac, never on-device —
// see tools/build_native_frontend.sh) invoked by tools/gen_config.py. Its only output
// is firmware/main/gen/frontend_tables.h, which both firmware/main/frontend_core.cc
// and the native training-side frontend library (same source file, different compile
// target) #include — one generated table, two consumers, by construction no
// opportunity for the tables themselves to diverge between training and firmware.
//
// Usage: gen_frontend_tables \
//   --sample_rate=16000 --window_sample_count=1024 --window_scaling_bits=12 \
//   --num_channels=64 --lower_band_limit=50 --upper_band_limit=7500 \
//   --fft_length=1024 --filter_bank_scaling_bits=12 --filter_bank_alignment=4 \
//   --filter_bank_channel_block_size=4 --pcan_strength=0.95 --pcan_offset=80.0 \
//   --pcan_gain_bits=21 --pcan_smoothing_bits=10 --correction_bits=4 \
//   > firmware/main/gen/frontend_tables.h

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FreqToMel — verbatim from
// python/tflite_micro/signal/utils/freq_to_mel_wrapper.cc
// ---------------------------------------------------------------------------
static float FreqToMel(float freq) { return 1127.0f * log1pf(freq / 700.0f); }

// ---------------------------------------------------------------------------
// Filterbank weight design — ported from
// python/tflite_micro/signal/ops/filter_bank_ops.py's
// _calc_center_freq / _quantize_filterbank_weight / _init_filter_bank_weights,
// using `float` throughout in place of the reference's `numpy.float32` (the
// Python code explicitly chose float32 specifically to match this original C
// `float` arithmetic — see that file's "A note about precision" comment).
// ---------------------------------------------------------------------------

struct FilterbankDesign {
  int start_index;
  int end_index;
  std::vector<int16_t> weights;
  std::vector<int16_t> unweights;
  std::vector<int16_t> channel_frequency_starts;
  std::vector<int16_t> channel_weight_starts;
  std::vector<int16_t> channel_widths;
};

static std::vector<float> CalcCenterFreqs(int channel_num, float lower_freq_limit,
                                          float upper_freq_limit) {
  float mel_lower = FreqToMel(lower_freq_limit);
  float mel_upper = FreqToMel(upper_freq_limit);
  float mel_span = mel_upper - mel_lower;
  float mel_spacing = mel_span / static_cast<float>(channel_num);
  std::vector<float> out(channel_num);
  for (int i = 0; i < channel_num; ++i) {
    out[i] = mel_lower + mel_spacing * static_cast<float>(i + 1);
  }
  return out;
}

static void QuantizeFilterbankWeight(float float_weight, int scale_bits,
                                     int16_t* weight, int16_t* unweight) {
  *weight = static_cast<int16_t>(float_weight * static_cast<float>(1 << scale_bits));
  *unweight =
      static_cast<int16_t>((1.0f - float_weight) * static_cast<float>(1 << scale_bits));
}

static FilterbankDesign InitFilterbankWeights(int spectrum_size, int sample_rate,
                                              int alignment, int channel_block_size,
                                              int num_channels, float lower_band_limit,
                                              float upper_band_limit,
                                              int weight_scaling_bits) {
  const int item_size = 2;  // sizeof(int16_t)
  int index_alignment = (alignment < item_size) ? 1 : (alignment / item_size);

  const int n = num_channels + 1;
  std::vector<int16_t> channel_frequency_starts(n, 0);
  std::vector<int16_t> channel_weight_starts(n, 0);
  std::vector<int16_t> channel_widths(n, 0);
  std::vector<int> actual_channel_starts(n, 0);
  std::vector<int> actual_channel_widths(n, 0);

  std::vector<float> center_mel_freqs =
      CalcCenterFreqs(num_channels + 1, lower_band_limit, upper_band_limit);

  float hz_per_sbin = (static_cast<float>(sample_rate) / 2.0f) /
                     static_cast<float>(spectrum_size - 1);
  int start_index =
      static_cast<int>(1 + (lower_band_limit / hz_per_sbin) + 0.5f);  // round()

  int chan_freq_index_start = start_index;
  int weight_index_start = 0;
  int needs_zeros = 0;

  for (int chan = 0; chan < num_channels + 1; ++chan) {
    int freq_index = chan_freq_index_start;
    while (FreqToMel(static_cast<float>(freq_index) * hz_per_sbin) <=
           center_mel_freqs[chan]) {
      freq_index += 1;
    }

    int width = freq_index - chan_freq_index_start;
    actual_channel_starts[chan] = chan_freq_index_start;
    actual_channel_widths[chan] = width;

    if (width == 0) {
      channel_frequency_starts[chan] = 0;
      channel_weight_starts[chan] = 0;
      channel_widths[chan] = static_cast<int16_t>(channel_block_size);
      if (needs_zeros == 0) {
        needs_zeros = 1;
        for (int j = 0; j < chan; ++j) {
          channel_weight_starts[j] =
              static_cast<int16_t>(channel_weight_starts[j] + channel_block_size);
        }
        weight_index_start += channel_block_size;
      }
    } else {
      int aligned_start = (chan_freq_index_start / index_alignment) * index_alignment;
      int aligned_width = chan_freq_index_start - aligned_start + width;
      int padded_width =
          ((aligned_width - 1) / channel_block_size + 1) * channel_block_size;

      channel_frequency_starts[chan] = static_cast<int16_t>(aligned_start);
      channel_weight_starts[chan] = static_cast<int16_t>(weight_index_start);
      channel_widths[chan] = static_cast<int16_t>(padded_width);
      weight_index_start += padded_width;
    }
    chan_freq_index_start = freq_index;
  }

  int num_weights = weight_index_start;
  std::vector<int16_t> weights(num_weights, 0);
  std::vector<int16_t> unweights(num_weights, 0);

  int end_index = 0;
  float mel_low = FreqToMel(lower_band_limit);
  for (int chan = 0; chan < num_channels + 1; ++chan) {
    int frequency = actual_channel_starts[chan];
    int num_frequencies = actual_channel_widths[chan];
    int frequency_offset = frequency - channel_frequency_starts[chan];
    int weight_start = channel_weight_starts[chan];
    float denom_val = (chan == 0) ? mel_low : center_mel_freqs[chan - 1];

    for (int j = 0; j < num_frequencies; ++j) {
      float num = center_mel_freqs[chan] - FreqToMel(static_cast<float>(frequency) * hz_per_sbin);
      float den = center_mel_freqs[chan] - denom_val;
      float weight = num / den;
      int weight_index = weight_start + frequency_offset + j;
      QuantizeFilterbankWeight(weight, weight_scaling_bits, &weights[weight_index],
                               &unweights[weight_index]);
      frequency += 1;
    }
    if (frequency > end_index) end_index = frequency;
  }

  if (end_index >= spectrum_size) {
    fprintf(stderr, "error: upper_band_limit too close to Nyquist "
                    "(end_index %d >= spectrum_size %d)\n",
            end_index, spectrum_size);
    exit(1);
  }

  FilterbankDesign d;
  d.start_index = start_index;
  d.end_index = end_index;
  d.weights = weights;
  d.unweights = unweights;
  d.channel_frequency_starts = channel_frequency_starts;
  d.channel_weight_starts = channel_weight_starts;
  d.channel_widths = channel_widths;
  return d;
}

// ---------------------------------------------------------------------------
// Hann window weights — ported from
// python/tflite_micro/signal/ops/window_op.py's hann_window_weights.
// ---------------------------------------------------------------------------

static std::vector<int16_t> HannWindowWeights(int window_length, int shift) {
  std::vector<int16_t> out(window_length);
  const double arg = M_PI * 2.0 / static_cast<double>(window_length);
  for (int n = 0; n < window_length; ++n) {
    double w = 0.5 - 0.5 * std::cos(arg * (static_cast<double>(n) + 0.5));
    out[n] = static_cast<int16_t>(std::lround(w * static_cast<double>(1 << shift)));
  }
  return out;
}

// ---------------------------------------------------------------------------
// PCAN gain LUT — verbatim from
// python/tflite_micro/signal/utils/wide_dynamic_func_lut_wrapper.cc
// ---------------------------------------------------------------------------

static constexpr size_t kWideDynamicFunctionBits = 32;
static constexpr size_t kWideDynamicFunctionLUTSize = (4 * kWideDynamicFunctionBits - 3);

static int16_t PcanGainLookupFunction(float strength, float offset, int gain_bits,
                                      int32_t input_bits, uint32_t x) {
  const float x_as_float =
      static_cast<float>(x) / static_cast<float>(static_cast<uint32_t>(1) << input_bits);
  const float gain_as_float = static_cast<float>(static_cast<uint32_t>(1) << gain_bits) *
                             powf(x_as_float + offset, -strength);
  if (gain_as_float > std::numeric_limits<int16_t>::max()) {
    return std::numeric_limits<int16_t>::max();
  }
  return static_cast<int16_t>(gain_as_float + 0.5f);
}

static std::vector<int16_t> WideDynamicFuncLut(float strength, float offset,
                                                int input_bits, int gain_bits) {
  std::vector<int16_t> storage(kWideDynamicFunctionLUTSize + 1);
  int16_t* gain_lut = storage.data();

  gain_lut[0] = PcanGainLookupFunction(strength, offset, gain_bits, input_bits, 0);
  gain_lut[1] = PcanGainLookupFunction(strength, offset, gain_bits, input_bits, 1);
  gain_lut -= 6;

  for (size_t interval = 2; interval <= kWideDynamicFunctionBits; ++interval) {
    const uint32_t x0 = static_cast<uint32_t>(1) << (interval - 1);
    const uint32_t x1 = x0 + (x0 >> 1);
    const uint32_t x2 =
        (interval == kWideDynamicFunctionBits) ? x0 + (x0 - 1) : 2 * x0;

    const int16_t y0 = PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x0);
    const int16_t y1 = PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x1);
    const int16_t y2 = PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x2);

    const int32_t diff1 = static_cast<int32_t>(y1 - y0);
    const int32_t diff2 = static_cast<int32_t>(y2 - y0);
    const int32_t a1 = 4 * diff1 - diff2;
    const int32_t a2 = diff2 - a1;

    gain_lut[4 * interval] = y0;
    gain_lut[4 * interval + 1] = static_cast<int16_t>(a1);
    gain_lut[4 * interval + 2] = static_cast<int16_t>(a2);
    gain_lut[4 * interval + 3] = 0;
  }
  gain_lut += 6;

  std::vector<int16_t> lut(gain_lut, gain_lut + kWideDynamicFunctionLUTSize);
  return lut;
}

// ---------------------------------------------------------------------------
// CLI + C header emission
// ---------------------------------------------------------------------------

struct Args {
  int sample_rate = 16000;
  int window_sample_count = 1024;
  int window_scaling_bits = 12;
  int num_channels = 64;
  float lower_band_limit = 50.0f;
  float upper_band_limit = 7500.0f;
  int fft_length = 1024;
  int filter_bank_scaling_bits = 12;
  int filter_bank_alignment = 4;
  int filter_bank_channel_block_size = 4;
  float pcan_strength = 0.95f;
  float pcan_offset = 80.0f;
  int pcan_gain_bits = 21;
  int pcan_smoothing_bits = 10;
  int correction_bits = 4;
};

static bool ParseArg(const std::string& arg, const char* name, std::string* out) {
  std::string prefix = std::string("--") + name + "=";
  if (arg.rfind(prefix, 0) == 0) {
    *out = arg.substr(prefix.size());
    return true;
  }
  return false;
}

static Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    std::string v;
    if (ParseArg(arg, "sample_rate", &v)) a.sample_rate = std::atoi(v.c_str());
    else if (ParseArg(arg, "window_sample_count", &v)) a.window_sample_count = std::atoi(v.c_str());
    else if (ParseArg(arg, "window_scaling_bits", &v)) a.window_scaling_bits = std::atoi(v.c_str());
    else if (ParseArg(arg, "num_channels", &v)) a.num_channels = std::atoi(v.c_str());
    else if (ParseArg(arg, "lower_band_limit", &v)) a.lower_band_limit = std::atof(v.c_str());
    else if (ParseArg(arg, "upper_band_limit", &v)) a.upper_band_limit = std::atof(v.c_str());
    else if (ParseArg(arg, "fft_length", &v)) a.fft_length = std::atoi(v.c_str());
    else if (ParseArg(arg, "filter_bank_scaling_bits", &v)) a.filter_bank_scaling_bits = std::atoi(v.c_str());
    else if (ParseArg(arg, "filter_bank_alignment", &v)) a.filter_bank_alignment = std::atoi(v.c_str());
    else if (ParseArg(arg, "filter_bank_channel_block_size", &v)) a.filter_bank_channel_block_size = std::atoi(v.c_str());
    else if (ParseArg(arg, "pcan_strength", &v)) a.pcan_strength = std::atof(v.c_str());
    else if (ParseArg(arg, "pcan_offset", &v)) a.pcan_offset = std::atof(v.c_str());
    else if (ParseArg(arg, "pcan_gain_bits", &v)) a.pcan_gain_bits = std::atoi(v.c_str());
    else if (ParseArg(arg, "pcan_smoothing_bits", &v)) a.pcan_smoothing_bits = std::atoi(v.c_str());
    else if (ParseArg(arg, "correction_bits", &v)) a.correction_bits = std::atoi(v.c_str());
    else {
      fprintf(stderr, "unrecognized argument: %s\n", arg.c_str());
      exit(1);
    }
  }
  return a;
}

template <typename T>
static void EmitArray(const char* type, const char* name, const std::vector<T>& v) {
  printf("static const %s %s[%zu] = {\n  ", type, name, v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    printf("%d, ", static_cast<int>(v[i]));
    if ((i + 1) % 16 == 0) printf("\n  ");
  }
  printf("\n};\n\n");
}

int main(int argc, char** argv) {
  Args a = ParseArgs(argc, argv);
  const int spectrum_size = a.fft_length / 2 + 1;

  FilterbankDesign fb = InitFilterbankWeights(
      spectrum_size, a.sample_rate, a.filter_bank_alignment,
      a.filter_bank_channel_block_size, a.num_channels, a.lower_band_limit,
      a.upper_band_limit, a.filter_bank_scaling_bits);

  std::vector<int16_t> hann = HannWindowWeights(a.window_sample_count, a.window_scaling_bits);

  std::vector<int16_t> pcan_lut = WideDynamicFuncLut(
      a.pcan_strength, a.pcan_offset, /*input_bits=*/a.pcan_smoothing_bits - a.correction_bits,
      a.pcan_gain_bits);

  printf("// GENERATED FILE — DO NOT EDIT.\n");
  printf("// Produced by tools/gen_frontend_tables.cc via tools/gen_config.py.\n");
  printf("// Ported line-for-line from tflite-micro's reference design-time Python/C++\n");
  printf("// (see the comment at the top of gen_frontend_tables.cc for sources).\n");
  printf("//\n");
  printf("// #included by BOTH firmware/main/frontend_core.cc (compiled for the\n");
  printf("// ESP32-S3) and the native macOS frontend library used by training's\n");
  printf("// features.py (tools/build_native_frontend.sh) — one set of tables, two\n");
  printf("// targets, by construction no opportunity to diverge.\n\n");
  printf("#pragma once\n#include <stdint.h>\n\n");

  printf("#define ECHO_SENTRY_FB_START_INDEX %d\n", fb.start_index);
  printf("#define ECHO_SENTRY_FB_END_INDEX %d\n\n", fb.end_index);

  EmitArray("int16_t", "kEchoSentryHannWindow", hann);
  EmitArray("int16_t", "kEchoSentryFilterbankWeights", fb.weights);
  EmitArray("int16_t", "kEchoSentryFilterbankUnweights", fb.unweights);
  EmitArray("int16_t", "kEchoSentryFilterbankChannelFrequencyStarts", fb.channel_frequency_starts);
  EmitArray("int16_t", "kEchoSentryFilterbankChannelWeightStarts", fb.channel_weight_starts);
  EmitArray("int16_t", "kEchoSentryFilterbankChannelWidths", fb.channel_widths);
  EmitArray("int16_t", "kEchoSentryPcanGainLut", pcan_lut);

  return 0;
}
