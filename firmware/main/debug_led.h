// Bench/bring-up indicator: drives the board's single addressable RGB LED from the live
// audio level and the capture path's health. See debug_led.c for the state table.
// Available in both the production and the CONFIG_ECHO_SENTRY_CAPTURE_MODE build — the
// capture build brings up no network, so this is its only live diagnostic channel.
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Starts the indicator task. Safe to call before audio_capture_start(): the task waits
// for the ring to exist before reading it. Returns ESP_ERR_NO_MEM if the task could not
// be created; callers should log rather than abort — a debug indicator must not be able
// to stop a node from booting.
esp_err_t debug_led_start(void);

#ifdef __cplusplus
}
#endif
