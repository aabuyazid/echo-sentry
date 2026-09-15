// esp_https_ota wrapper — firmware updates served from the Pi hub, comprehensive-
// plan.md Part 4 ("flashing 5 nodes by USB in an equipment room is miserable — set
// this up early").
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Downloads and applies a firmware image from `url`, then reboots on success.
// Despite the esp_https_ota name, plain http:// is expected here — the hub is only
// reachable over the isolated node AP (comprehensive-plan.md §2.1), so TLS buys
// little for this link; see comprehensive-plan.md's "Open items" on MQTT/HTTP TLS
// for the same tradeoff made project-wide. Revisit before any production framing.
esp_err_t ota_start(const char *url);

#ifdef __cplusplus
}
#endif
