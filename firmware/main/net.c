#include "net.h"

#include <stdio.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_netif_sntp.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "mqtt_client.h"

#include "provisioning.h"

static const char *TAG = "net";
#define TOPIC_PREFIX "echo-sentry/v1"

static EventGroupHandle_t s_wifi_events;
#define WIFI_CONNECTED_BIT BIT0

static esp_mqtt_client_handle_t s_mqtt;
static volatile bool s_mqtt_connected;
static net_cmd_handler_t s_cmd_handler;
static char s_status_topic[64];
static char s_cmd_topic[64];

// ---- Wi-Fi ------------------------------------------------------------------------

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data) {
    (void)arg;
    (void)data;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi disconnected, retrying");
        xEventGroupClearBits(s_wifi_events, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(TAG, "Wi-Fi connected");
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t wifi_start(void) {
    const node_config_t *cfg = provisioning_get();
    if (cfg->wifi_ssid[0] == '\0') {
        ESP_LOGE(TAG, "no wifi_ssid provisioned — see firmware/flash.sh");
        return ESP_ERR_INVALID_STATE;
    }

    s_wifi_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               &wifi_event_handler, NULL));

    wifi_config_t wifi_cfg = {0};
    strncpy((char *)wifi_cfg.sta.ssid, cfg->wifi_ssid, sizeof(wifi_cfg.sta.ssid) - 1);
    strncpy((char *)wifi_cfg.sta.password, cfg->wifi_password,
           sizeof(wifi_cfg.sta.password) - 1);
    wifi_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "connecting to '%s'...", cfg->wifi_ssid);
    xEventGroupWaitBits(s_wifi_events, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    return ESP_OK;
}

// ---- SNTP ---------------------------------------------------------------------------
// The node network has no internet and the ESP32 has no battery-backed RTC — the Pi
// runs chrony as the node network's NTP server (comprehensive-plan.md Part 4).

static void sntp_start(void) {
    const node_config_t *cfg = provisioning_get();
    esp_sntp_config_t sntp_cfg = ESP_NETIF_SNTP_DEFAULT_CONFIG(cfg->hub_host);
    esp_netif_sntp_init(&sntp_cfg);
    if (esp_netif_sntp_sync_wait(pdMS_TO_TICKS(10000)) != ESP_OK) {
        ESP_LOGW(TAG, "SNTP sync against %s timed out — timestamps may be wrong until "
                     "the next sync attempt",
                cfg->hub_host);
    } else {
        ESP_LOGI(TAG, "SNTP synced against %s", cfg->hub_host);
    }
}

// ---- MQTT -----------------------------------------------------------------------

static void mqtt_event_handler(void *arg, esp_event_base_t base, int32_t event_id,
                               void *event_data) {
    (void)arg;
    (void)base;
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected");
            s_mqtt_connected = true;
            esp_mqtt_client_subscribe(s_mqtt, s_cmd_topic, 1);
            esp_mqtt_client_publish(s_mqtt, s_status_topic, "{\"status\":\"online\"}", 0,
                                    1, true);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "MQTT disconnected");
            s_mqtt_connected = false;
            break;
        case MQTT_EVENT_DATA:
            if (s_cmd_handler != NULL) {
                s_cmd_handler(event->data, event->data_len);
            }
            break;
        default:
            break;
    }
}

static esp_err_t mqtt_start(void) {
    const node_config_t *cfg = provisioning_get();
    snprintf(s_status_topic, sizeof(s_status_topic), TOPIC_PREFIX "/%s/status",
            cfg->node_id);
    snprintf(s_cmd_topic, sizeof(s_cmd_topic), TOPIC_PREFIX "/cmd/%s", cfg->node_id);

    char uri[80];
    snprintf(uri, sizeof(uri), "mqtt://%s:%u", cfg->hub_host, cfg->mqtt_port);

    // Last-will: comprehensive-plan.md's dead-man detection corroborates a missing
    // "seq" gap with this — the hub sees "offline" the moment the TCP connection
    // dies, not just after several missed score intervals.
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = uri,
        .session.last_will.topic = s_status_topic,
        .session.last_will.msg = "{\"status\":\"offline\"}",
        .session.last_will.qos = 1,
        .session.last_will.retain = true,
        .session.keepalive = 30,
    };

    s_mqtt = esp_mqtt_client_init(&mqtt_cfg);
    if (s_mqtt == NULL) return ESP_FAIL;
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(s_mqtt, ESP_EVENT_ANY_ID,
                                                   mqtt_event_handler, NULL));
    return esp_mqtt_client_start(s_mqtt);
}

// ---- public API ---------------------------------------------------------------------

esp_err_t net_start(net_cmd_handler_t cmd_handler) {
    s_cmd_handler = cmd_handler;
    ESP_ERROR_CHECK(wifi_start());
    sntp_start();
    return mqtt_start();
}

bool net_is_mqtt_connected(void) { return s_mqtt_connected; }

int net_publish(const char *topic_suffix, const char *payload, int qos, bool retain) {
    if (!s_mqtt_connected) return -1;
    const node_config_t *cfg = provisioning_get();
    char topic[80];
    snprintf(topic, sizeof(topic), TOPIC_PREFIX "/%s/%s", cfg->node_id, topic_suffix);
    return esp_mqtt_client_publish(s_mqtt, topic, payload, 0, qos, retain);
}
