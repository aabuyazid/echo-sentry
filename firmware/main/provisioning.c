#include "provisioning.h"

#include <string.h>

#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "gen/frontend_config.h"

static const char *TAG = "provisioning";
static const char *NVS_NAMESPACE = "echo_sentry";

static node_config_t s_config;

static void load_string(nvs_handle_t h, const char *key, char *out, size_t out_len,
                        const char *fallback) {
    size_t len = out_len;
    if (nvs_get_str(h, key, out, &len) != ESP_OK) {
        strncpy(out, fallback, out_len - 1);
        out[out_len - 1] = '\0';
    }
}

esp_err_t provisioning_init(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    memset(&s_config, 0, sizeof(s_config));
    s_config.mqtt_port = 1883;
    s_config.evidence_port = 8765;
    s_config.anomaly_threshold = ECHO_SENTRY_ANOMALY_THRESHOLD_DEFAULT;
    strncpy(s_config.hub_host, "10.42.0.1", PROVISIONING_HOST_MAX_LEN - 1);

    nvs_handle_t h;
    err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "nvs_open failed (%s) — running with defaults; node is "
                     "unprovisioned. See firmware/flash.sh.",
                esp_err_to_name(err));
        return ESP_OK;
    }

    load_string(h, "node_id", s_config.node_id, sizeof(s_config.node_id), "");
    load_string(h, "wifi_ssid", s_config.wifi_ssid, sizeof(s_config.wifi_ssid), "");
    load_string(h, "wifi_pass", s_config.wifi_password, sizeof(s_config.wifi_password), "");
    load_string(h, "hub_host", s_config.hub_host, sizeof(s_config.hub_host),
               s_config.hub_host);

    uint16_t port;
    if (nvs_get_u16(h, "mqtt_port", &port) == ESP_OK) s_config.mqtt_port = port;
    if (nvs_get_u16(h, "evid_port", &port) == ESP_OK) s_config.evidence_port = port;

    // NVS has no float API — stored as the IEEE-754 bit pattern in a u32.
    uint32_t threshold_bits;
    if (nvs_get_u32(h, "threshold", &threshold_bits) == ESP_OK) {
        memcpy(&s_config.anomaly_threshold, &threshold_bits, sizeof(float));
    }

    nvs_close(h);

    if (s_config.node_id[0] == '\0') {
        ESP_LOGW(TAG, "node_id is unset — this node has not been provisioned "
                     "(firmware/flash.sh)");
    }
    return ESP_OK;
}

const node_config_t *provisioning_get(void) { return &s_config; }

esp_err_t provisioning_set_threshold(float threshold) {
    s_config.anomaly_threshold = threshold;

    nvs_handle_t h;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) return err;

    uint32_t threshold_bits;
    memcpy(&threshold_bits, &threshold, sizeof(float));
    err = nvs_set_u32(h, "threshold", threshold_bits);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}
