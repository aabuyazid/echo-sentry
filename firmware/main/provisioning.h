// Per-node configuration in NVS — comprehensive-plan.md §2.3. Lets a node's
// identity, Wi-Fi creds, and threshold be retuned without reflashing.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PROVISIONING_NODE_ID_MAX_LEN 16
#define PROVISIONING_SSID_MAX_LEN 32
#define PROVISIONING_PASSWORD_MAX_LEN 64
#define PROVISIONING_HOST_MAX_LEN 40

typedef struct {
    char node_id[PROVISIONING_NODE_ID_MAX_LEN];
    char wifi_ssid[PROVISIONING_SSID_MAX_LEN];
    char wifi_password[PROVISIONING_PASSWORD_MAX_LEN];
    char hub_host[PROVISIONING_HOST_MAX_LEN];  // Pi's AP-side IP, e.g. "10.42.0.1"
    uint16_t mqtt_port;
    uint16_t evidence_port;
    float anomaly_threshold;  // local fast-path trigger, comprehensive-plan.md §3.4
} node_config_t;

esp_err_t provisioning_init(void);

// Always succeeds once provisioning_init() has run; falls back to
// ECHO_SENTRY_ANOMALY_THRESHOLD_DEFAULT (gen/frontend_config.h) and a blank node_id
// if NVS was never provisioned (firmware/flash.sh's job — see that script).
const node_config_t *provisioning_get(void);

// Applied by the "set_threshold" MQTT command (telemetry.c) and persisted so it
// survives a reboot without a reflash.
esp_err_t provisioning_set_threshold(float threshold);

#ifdef __cplusplus
}
#endif
