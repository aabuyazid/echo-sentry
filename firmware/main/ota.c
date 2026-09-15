#include "ota.h"

#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_log.h"

static const char *TAG = "ota";

esp_err_t ota_start(const char *url) {
    esp_http_client_config_t http_cfg = {
        .url = url,
        .timeout_ms = 30000,
        .keep_alive_enable = true,
    };
    esp_https_ota_config_t ota_cfg = {
        .http_config = &http_cfg,
    };

    ESP_LOGI(TAG, "starting OTA from %s", url);
    esp_err_t err = esp_https_ota(&ota_cfg);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "OTA succeeded, rebooting");
        esp_restart();
        return ESP_OK;  // unreachable
    }
    ESP_LOGE(TAG, "OTA failed: %s", esp_err_to_name(err));
    return err;
}
