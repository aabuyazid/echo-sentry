// Wi-Fi station (joins the Pi's AP), SNTP (against the Pi's chrony server), and MQTT
// — comprehensive-plan.md §2.1 / §2.3.
#pragma once

#include <stdbool.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Called on the MQTT client's own task for every message on
// echo-sentry/v1/cmd/<node_id> — see telemetry.c for the actual command dispatch.
typedef void (*net_cmd_handler_t)(const char *json_payload, int len);

esp_err_t net_start(net_cmd_handler_t cmd_handler);

bool net_is_mqtt_connected(void);

// Publishes to echo-sentry/v1/<node_id>/<topic_suffix>. Returns the message id on
// success or -1 if not currently connected. Nodes have no local disk to spool
// to — a publish dropped here is simply a missed sample, same as a message lost in
// transit at QoS 0/1; comprehensive-plan.md's store-and-forward guarantee lives on
// the hub side of the link, not here.
int net_publish(const char *topic_suffix, const char *payload, int qos, bool retain);

#ifdef __cplusplus
}
#endif
