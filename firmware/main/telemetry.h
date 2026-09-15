// Score payload assembly/publish and cmd/ dispatch — comprehensive-plan.md §2.2(b).
#pragma once

#include <stdint.h>

#include "esp_err.h"

#include "inference.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t telemetry_init(void);

// Builds and publishes one score payload matching hub/src/echo_sentry_hub/models.py's
// ScorePayload exactly — field-for-field drift between the two is a silent ingest
// failure on the hub side (see ingest.py's ValidationError handling).
void telemetry_publish_score(inference_clip_result_t result, float rms_dbfs, uint32_t drops);

// net.h's net_cmd_handler_t — dispatches set_threshold/capture/ota/reboot
// (hub/src/echo_sentry_hub/models.py's Cmd variants).
void telemetry_handle_cmd(const char *json_payload, int len);

#ifdef __cplusplus
}
#endif
