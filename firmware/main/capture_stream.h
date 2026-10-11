// Device side of CONFIG_ECHO_SENTRY_CAPTURE_MODE — comprehensive-plan.md §3.1. Streams
// audio_capture.c's PCM ring off the board for the data-collection campaign, over the
// transport selected by ECHO_SENTRY_CAPTURE_TRANSPORT (Kconfig.projbuild). Only the
// SERIAL transport is implemented; selecting TCP fails the build (see
// capture_stream.c) — it is a seam for the Wi-Fi field-campaign path once a hub
// exists, not a working alternative today.
//
// Host-side counterpart: tools/capture_serial.py, which unwraps this file's framing
// and writes WAV + sidecar JSON via tools/capture_sink.py — the same sink
// tools/capture_server.py (the TCP receiver) uses, so output is byte-identical
// regardless of transport.
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Installs the transport and the log-interleaving hook, then creates the streaming
// task. Returns once the task is created, not once a host has attached — the task
// itself waits (with a bounded timeout) for the handshake to go through.
esp_err_t capture_stream_start(void);

#ifdef __cplusplus
}
#endif
