// PCM ring snapshot + HTTP multipart upload — the evidence flywheel,
// comprehensive-plan.md §3.4. The one path where PCM crosses the node -> Pi
// boundary in production; it stops at the hub (hub/src/echo_sentry_hub/evidence.py)
// — comprehensive-plan.md §2.1, "the room boundary is the privacy boundary".
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Snapshots the last CONFIG_ECHO_SENTRY_EVIDENCE_RING_SECONDS of audio from
// audio_capture.c's ring and uploads it as a WAV multipart POST. `trigger` is a
// short string (e.g. "fast_path_threshold", "manual") matching hub/src/
// echo_sentry_hub/models.py's EvidencePayload trigger values.
esp_err_t evidence_capture_and_upload(const char *trigger);

#ifdef __cplusplus
}
#endif
