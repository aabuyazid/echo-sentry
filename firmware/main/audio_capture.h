// I2S capture from the INMP441 into a PSRAM circular buffer — comprehensive-plan.md
// §2.3. Shared verbatim between production inference and CONFIG_ECHO_SENTRY_CAPTURE_MODE
// (see Kconfig.projbuild) so the data collection campaign's audio comes from the
// identical sensor path production inference runs on — comprehensive-plan.md §3.1.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t audio_capture_start(void);

// Total samples written so far (monotonic, wraps at UINT32_MAX — ~76 hours at 16kHz,
// acceptable for a device that reboots at least that often via watchdog/OTA).
uint32_t audio_capture_write_count(void);

// Blocks (polling) until sample index [start, start + num_samples) has been written,
// then copies it out of the ring. `start` must not be more than the ring's capacity
// behind the current write count, or already-overwritten data is returned — callers
// that fall behind should track drops via audio_capture_drop_count(), not rely on
// this to detect it.
void audio_capture_read_at(uint32_t start, int16_t *out, size_t num_samples);

// Cumulative I2S DMA read failures / buffer-full drops — comprehensive-plan.md §3.1's
// data-integrity signal, reported in every score payload's "drops" field.
uint32_t audio_capture_drop_count(void);

// Ring capacity in samples. audio_capture_read_at() does not detect a torn read on its
// own (see that function's note) — a caller that lags production's real-time drain
// rate (e.g. a slow transport) must check, after every audio_capture_read_at() call,
// that (audio_capture_write_count() - start) <= audio_capture_ring_capacity() still
// holds; otherwise the read may have straddled a write-head wraparound and returned a
// mix of old and overwritten samples.
uint32_t audio_capture_ring_capacity(void);

#ifdef __cplusplus
}
#endif
