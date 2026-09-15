#include "telemetry.h"

#include <string.h>
#include <time.h>

#include "cJSON.h"
#include "esp_log.h"
#include "esp_system.h"

#include "gen/frontend_config.h"
#include "net.h"
#include "ota.h"
#include "provisioning.h"

static const char *TAG = "telemetry";

// TODO(M5): wire the real per-node model_id from model.json (training/src/
// echo_sentry_train/export.py) into the build — see main/models/README.md. There is
// no trained model to reference yet (needs the M4 data-collection campaign).
static const char *FIRMWARE_VERSION = "0.1.0";
static const char *MODEL_ID_PLACEHOLDER = "unassigned";

static uint32_t s_seq;

esp_err_t telemetry_init(void) {
    s_seq = 0;
    return ESP_OK;
}

void telemetry_publish_score(inference_clip_result_t result, float rms_dbfs,
                             uint32_t drops) {
    const node_config_t *cfg = provisioning_get();

    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "v", 1);
    cJSON_AddStringToObject(root, "node", cfg->node_id);
    cJSON_AddNumberToObject(root, "t", (double)time(NULL));
    cJSON_AddNumberToObject(root, "seq", s_seq++);
    cJSON_AddNumberToObject(root, "err_mean", result.err_mean);
    cJSON_AddNumberToObject(root, "err_p95", result.err_p95);
    cJSON_AddNumberToObject(root, "err_max", result.err_max);
    cJSON_AddNumberToObject(root, "rms_dbfs", rms_dbfs);

    // TODO: real octave-band spectral summary (diagnostic-only field on the hub
    // dashboard; not read by the detector).
    cJSON *bands = cJSON_CreateArray();
    for (int i = 0; i < 16; ++i) cJSON_AddItemToArray(bands, cJSON_CreateNumber(0));
    cJSON_AddItemToObject(root, "bands", bands);

    cJSON_AddNumberToObject(root, "drops", drops);
    cJSON_AddStringToObject(root, "model", MODEL_ID_PLACEHOLDER);
    cJSON_AddStringToObject(root, "cfg", ECHO_SENTRY_CONFIG_HASH);
    cJSON_AddStringToObject(root, "fw", FIRMWARE_VERSION);

    char *payload = cJSON_PrintUnformatted(root);
    if (payload != NULL) {
        net_publish("score", payload, 1, false);
        cJSON_free(payload);
    }
    cJSON_Delete(root);
}

void telemetry_handle_cmd(const char *json_payload, int len) {
    cJSON *root = cJSON_ParseWithLength(json_payload, len);
    if (root == NULL) {
        ESP_LOGW(TAG, "cmd: invalid JSON");
        return;
    }

    const cJSON *cmd = cJSON_GetObjectItemCaseSensitive(root, "cmd");
    if (!cJSON_IsString(cmd)) {
        ESP_LOGW(TAG, "cmd: missing 'cmd' field");
        cJSON_Delete(root);
        return;
    }

    if (strcmp(cmd->valuestring, "set_threshold") == 0) {
        const cJSON *threshold = cJSON_GetObjectItemCaseSensitive(root, "threshold");
        if (cJSON_IsNumber(threshold)) {
            ESP_LOGI(TAG, "cmd: set_threshold -> %f", threshold->valuedouble);
            provisioning_set_threshold((float)threshold->valuedouble);
        }
    } else if (strcmp(cmd->valuestring, "reboot") == 0) {
        ESP_LOGW(TAG, "cmd: reboot requested");
        esp_restart();
    } else if (strcmp(cmd->valuestring, "ota") == 0) {
        const cJSON *url = cJSON_GetObjectItemCaseSensitive(root, "url");
        if (cJSON_IsString(url)) {
            ESP_LOGI(TAG, "cmd: ota -> %s", url->valuestring);
            ota_start(url->valuestring);
        }
    } else if (strcmp(cmd->valuestring, "capture") == 0) {
        ESP_LOGI(TAG, "cmd: capture (manual evidence trigger) — see evidence.c");
        // evidence_trigger() is wired in app_main.c's detector callback; a manual
        // trigger here would need the same PCM-ring-snapshot path. Left as a
        // follow-up once evidence.c's trigger plumbing is exercised on real hardware.
    } else {
        ESP_LOGW(TAG, "cmd: unrecognized '%s'", cmd->valuestring);
    }

    cJSON_Delete(root);
}
