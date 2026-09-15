// TFLM inference over frontend_core.cc's output — comprehensive-plan.md §2.3 / §4.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float err_mean;
    float err_p95;
    float err_max;
} inference_clip_result_t;

// Sets up the TFLM interpreter against the embedded per-node model. Call once at
// boot, after provisioning_init() (needs the node's model/normalization artifacts —
// see models/README in this directory and tools/export_node_model.sh).
int inference_init(void);

// Runs inference over every log-mel frame vector built from `frames`
// (ECHO_SENTRY_CLIP_FRAMES x ECHO_SENTRY_MEL_CHANNELS int16, from frontend_core.cc),
// subsampled by ECHO_SENTRY_VECTOR_STRIDE exactly like training/src/echo_sentry_train/
// features.py's frames_to_vectors, and aggregates the per-vector reconstruction MSE
// into the clip-level statistics the score payload reports (comprehensive-plan.md
// §1.1 / §2.2(b)).
inference_clip_result_t inference_score_clip(const int16_t *frames, int num_frames);

#ifdef __cplusplus
}
#endif
