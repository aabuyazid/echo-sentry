// See inference.h. Scoring matches training/src/echo_sentry_train/model.py's
// DenseAutoencoderScorer.score() and evaluate.py's run_tflite_int8() exactly:
// reconstruction MSE computed between the ORIGINAL normalized float vector and the
// model's DEQUANTIZED output (not a quantize-then-dequantize round trip of the
// input) — see evaluate.py's fp32_int8_delta for why that distinction matters for
// the fp32-vs-int8 comparison this mirrors on-device.

#include "inference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gen/frontend_config.h"

extern "C" {
// Populated by EMBED_FILES in main/CMakeLists.txt from main/models/{model.tflite,
// normalization.bin} — see main/models/README.md for how those get there.
extern const uint8_t model_tflite_start[] asm("_binary_model_tflite_start");
extern const uint8_t model_tflite_end[] asm("_binary_model_tflite_end");
extern const uint8_t normalization_bin_start[] asm("_binary_normalization_bin_start");
extern const uint8_t normalization_bin_end[] asm("_binary_normalization_bin_end");
}

namespace {

// comprehensive-plan.md §2.3's memory budget targets ~32KB; sized up for margin
// until a real trained model's actual arena_used_bytes() is measured (M2/M5).
constexpr int kArenaBytes = 48 * 1024;
alignas(16) uint8_t g_arena[kArenaBytes];

// FullyConnected + Relu only — BatchNorm folds into FullyConnected at conversion
// time (comprehensive-plan.md §4.4). If a real exported model needs more ops,
// AllocateTensors()/Invoke() below fail with a clear "op not found" style message;
// add the op here rather than reaching for AllOpsResolver (comprehensive-plan.md
// §4.4 on why: saves tens of KB).
using OpResolver = tflite::MicroMutableOpResolver<2>;

const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *g_interpreter = nullptr;
const float *g_mean = nullptr;  // ECHO_SENTRY_VECTOR_DIM floats
const float *g_std = nullptr;   // ECHO_SENTRY_VECTOR_DIM floats

float ScoreVector(const int16_t *vector_raw) {
    TfLiteTensor *input = g_interpreter->input(0);
    TfLiteTensor *output = g_interpreter->output(0);

    const float in_scale = input->params.scale;
    const int32_t in_zero = input->params.zero_point;
    int8_t *in_data = tflite::GetTensorData<int8_t>(input);

    float normalized[ECHO_SENTRY_VECTOR_DIM];
    for (int i = 0; i < ECHO_SENTRY_VECTOR_DIM; ++i) {
        normalized[i] = (static_cast<float>(vector_raw[i]) - g_mean[i]) / g_std[i];
        int32_t q = static_cast<int32_t>(std::lround(normalized[i] / in_scale)) + in_zero;
        q = std::min<int32_t>(std::max<int32_t>(q, -128), 127);
        in_data[i] = static_cast<int8_t>(q);
    }

    if (g_interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("inference: Invoke failed");
        return 0.0f;
    }

    const float out_scale = output->params.scale;
    const int32_t out_zero = output->params.zero_point;
    const int8_t *out_data = tflite::GetTensorData<int8_t>(output);

    float sum_sq_err = 0.0f;
    for (int i = 0; i < ECHO_SENTRY_VECTOR_DIM; ++i) {
        float reconstructed = (static_cast<float>(out_data[i]) - out_zero) * out_scale;
        float diff = normalized[i] - reconstructed;
        sum_sq_err += diff * diff;
    }
    return sum_sq_err / ECHO_SENTRY_VECTOR_DIM;
}

}  // namespace

extern "C" int inference_init(void) {
    g_model = tflite::GetModel(model_tflite_start);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("inference: model schema version %d != supported %d", g_model->version(),
                    TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static OpResolver resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk) return -1;
    if (resolver.AddRelu() != kTfLiteOk) return -1;

    static tflite::MicroInterpreter interpreter(g_model, resolver, g_arena, kArenaBytes);
    g_interpreter = &interpreter;
    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("inference: AllocateTensors failed (arena too small, or an "
                    "unregistered op — see the OpResolver comment above)");
        return -1;
    }

    const size_t norm_bytes = normalization_bin_end - normalization_bin_start;
    const size_t expected_bytes = sizeof(float) * ECHO_SENTRY_VECTOR_DIM * 2;
    if (norm_bytes != expected_bytes) {
        MicroPrintf("inference: normalization.bin is %u bytes, expected %u — model/"
                    "config mismatch",
                    (unsigned)norm_bytes, (unsigned)expected_bytes);
        return -1;
    }
    g_mean = reinterpret_cast<const float *>(normalization_bin_start);
    g_std = g_mean + ECHO_SENTRY_VECTOR_DIM;

    MicroPrintf("inference: ready, arena_used=%u/%u bytes",
               (unsigned)g_interpreter->arena_used_bytes(), (unsigned)kArenaBytes);
    return 0;
}

extern "C" inference_clip_result_t inference_score_clip(const int16_t *frames, int num_frames) {
    inference_clip_result_t result = {0.0f, 0.0f, 0.0f};
    if (g_interpreter == nullptr) return result;

    const int fpv = ECHO_SENTRY_FRAMES_PER_VECTOR;
    const int stride = ECHO_SENTRY_VECTOR_STRIDE;
    if (num_frames < fpv) return result;

    int16_t vector[ECHO_SENTRY_VECTOR_DIM];
    std::vector<float> scores;
    scores.reserve(ECHO_SENTRY_VECTORS_PER_CLIP);

    for (int start = 0; start + fpv <= num_frames; start += stride) {
        for (int f = 0; f < fpv; ++f) {
            std::memcpy(vector + f * ECHO_SENTRY_MEL_CHANNELS,
                       frames + (start + f) * ECHO_SENTRY_MEL_CHANNELS,
                       sizeof(int16_t) * ECHO_SENTRY_MEL_CHANNELS);
        }
        scores.push_back(ScoreVector(vector));
    }
    if (scores.empty()) return result;

    std::sort(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float s : scores) sum += s;
    result.err_mean = sum / scores.size();
    result.err_max = scores.back();
    size_t p95_idx = static_cast<size_t>(0.95f * static_cast<float>(scores.size() - 1));
    result.err_p95 = scores[p95_idx];
    return result;
}
