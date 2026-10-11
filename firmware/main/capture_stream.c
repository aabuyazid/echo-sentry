#include "capture_stream.h"

#if CONFIG_ECHO_SENTRY_CAPTURE_MODE

#if CONFIG_ECHO_SENTRY_CAPTURE_TRANSPORT_TCP
#error "ECHO_SENTRY_CAPTURE_TRANSPORT_TCP is a Kconfig seam for the future Wi-Fi/field-" \
       "campaign transport (comprehensive-plan.md §3.1) and has no implementation yet." \
       " Select the SERIAL transport for bring-up, or implement capture_stream.c's TCP" \
       " path before building with this option."
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "driver/usb_serial_jtag.h"
#include "esp_app_desc.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "audio_capture.h"
#include "gen/frontend_config.h"
#include "provisioning.h"

static const char *TAG = "capture_stream";

// ---- wire framing ------------------------------------------------------------------
//
// magic("ESCP") | type(u8) | seq(u16 BE) | sample_index(u32 BE) | len(u32 BE) |
// payload(len) | crc32(u32 BE, over type..payload)
//
// Framing exists only for the SERIAL transport, because that pipe is shared with the
// log console: esp_log_set_vprintf() only intercepts ESP_LOG (not esp_rom_printf,
// ESP_EARLY_LOGx, or the panic handler — all of which write straight to the same TX
// FIFO), so unframed bytes can appear on the wire at any time, not just before this
// module starts. seq is a best-effort diagnostic counter, incremented non-atomically
// from whichever task sends a frame — a torn increment under concurrent logging from
// two tasks costs a duplicate/skipped sequence number, never a corrupt frame. CRC32 is
// the actual integrity check: tools/capture_serial.py scans for `magic`, and on a CRC
// mismatch or a short read, discards that one frame and resumes scanning rather than
// losing the whole stream.
#define FRAME_MAGIC "ESCP"
#define FRAME_MAGIC_LEN 4
#define FRAME_META_LEN 11  // type(1) + seq(2) + sample_index(4) + len(4)
#define FRAME_CRC_LEN 4
#define FRAME_OVERHEAD (FRAME_MAGIC_LEN + FRAME_META_LEN + FRAME_CRC_LEN)  // 19

typedef enum {
    FRAME_TYPE_HANDSHAKE = 1,
    FRAME_TYPE_SEGMENT_HEADER = 2,
    FRAME_TYPE_PCM_CHUNK = 3,
    FRAME_TYPE_LOG = 4,
} frame_type_t;

// PCM payload per frame (2048 samples * 2 bytes = 4096 B), matching the "~4KB chunks"
// budget so a segment (up to 60 s = 1.92 MB) is never buffered whole.
#define CHUNK_SAMPLES 2048

static volatile bool s_driver_ready;
static volatile bool s_in_log_hook;  // reentrancy guard, see capture_log_vprintf()
static uint16_t s_frame_seq;

// Standard reflected CRC-32 (poly 0xEDB88320, init/xorout 0xFFFFFFFF) — the same
// algorithm Python's zlib.crc32() implements, so tools/capture_serial.py verifies with
// a plain `zlib.crc32(data)` rather than needing a matching hand-rolled table. Written
// as a portable bit-loop (not the ROM CRC32 routine) so the on-wire algorithm is
// exactly what's in this file, not an assumption about ROM semantics.
static uint32_t crc32_compute(const uint8_t *buf, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= buf[i];
        for (int b = 0; b < 8; ++b) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

static uint8_t *put_u16_be(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)v;
    return p + 2;
}

static uint8_t *put_u32_be(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
    return p + 4;
}

// Builds one complete frame into `scratch` and writes it in a single
// usb_serial_jtag_write_bytes() call. usb_serial_jtag_write_bytes() is all-or-nothing
// per call (either the whole buffer is accepted into the TX ring or nothing is, on
// timeout) — building the full frame first means a failed write never leaves a
// half-sent frame on the wire; the caller just counts it as a transport drop and the
// next frame starts cleanly.
static bool send_frame_into(uint8_t *scratch, size_t scratch_cap, frame_type_t type,
                            uint32_t sample_index, const void *payload, size_t len,
                            TickType_t ticks_to_wait) {
    size_t total = FRAME_OVERHEAD + len;
    if (total > scratch_cap) {
        ESP_LOGE(TAG, "frame too large for scratch buffer (%u > %u)", (unsigned)total,
                 (unsigned)scratch_cap);
        return false;
    }

    uint8_t *p = scratch;
    memcpy(p, FRAME_MAGIC, FRAME_MAGIC_LEN);
    p += FRAME_MAGIC_LEN;
    uint8_t *meta_start = p;
    *p++ = (uint8_t)type;
    p = put_u16_be(p, s_frame_seq++);
    p = put_u32_be(p, sample_index);
    p = put_u32_be(p, (uint32_t)len);
    if (len > 0) {
        memcpy(p, payload, len);
        p += len;
    }
    uint32_t crc = crc32_compute(meta_start, FRAME_META_LEN + len);
    put_u32_be(p, crc);

    int written = usb_serial_jtag_write_bytes(scratch, total, ticks_to_wait);
    return written == (int)total;
}

// ---- log interleaving ---------------------------------------------------------------

static int capture_log_vprintf(const char *fmt, va_list args) {
    // Fall back to the ROM's own vprintf (same physical port, unframed) rather than
    // risk re-entering this hook: usb_serial_jtag_write_bytes() calls ESP_LOGE on a
    // bad argument or an uninstalled driver, which would otherwise call straight back
    // into this function. Neither condition should occur once s_driver_ready is true
    // and args are well-formed, but the guard is cheap insurance against a stack
    // overflow if it ever does.
    if (!s_driver_ready || s_in_log_hook) {
        return esp_rom_vprintf(fmt, args);
    }
    s_in_log_hook = true;
    char line[200];
    int n = vsnprintf(line, sizeof(line), fmt, args);
    if (n > 0) {
        size_t payload_len = (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1;
        uint8_t frame_buf[FRAME_OVERHEAD + sizeof(line)];
        // ticks_to_wait = 0: a log frame must never block. This hook can run on any
        // task's stack (including tasks unrelated to capture streaming), so a stalled
        // USB pipe must drop the line, not stall every logger in the system.
        send_frame_into(frame_buf, sizeof(frame_buf), FRAME_TYPE_LOG, 0, line,
                        payload_len, 0);
    }
    s_in_log_hook = false;
    return n;
}

// ---- handshake / segment header -----------------------------------------------------

static void send_handshake(uint8_t *scratch, size_t scratch_cap) {
    const node_config_t *cfg = provisioning_get();
    const esp_app_desc_t *desc = esp_app_get_description();
    const char *node_id =
        (cfg->node_id[0] != '\0') ? cfg->node_id : CONFIG_ECHO_SENTRY_CAPTURE_NODE_ID;

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "node_id", node_id);
    cJSON_AddStringToObject(root, "fw", desc->version);
    cJSON_AddStringToObject(root, "cfg", ECHO_SENTRY_CONFIG_HASH);
    char *payload = cJSON_PrintUnformatted(root);
    if (payload != NULL) {
        // Generous but finite: this is the one point where blocking until a host
        // attaches is the right behavior (there's nothing useful to do without a
        // listener), but it must not be able to wedge forever.
        bool ok = send_frame_into(scratch, scratch_cap, FRAME_TYPE_HANDSHAKE, 0, payload,
                                  strlen(payload), pdMS_TO_TICKS(10000));
        if (!ok) {
            ESP_LOGW(TAG, "handshake send failed/timed out — start "
                         "tools/capture_serial.py before resetting the board");
        }
        cJSON_free(payload);
    }
    cJSON_Delete(root);
}

static void send_segment_header(uint8_t *scratch, size_t scratch_cap,
                                uint32_t i2s_drops_cumulative,
                                uint32_t transport_drops_cumulative) {
    cJSON *root = cJSON_CreateObject();
    // No SNTP on this transport (net_start() is never called in capture-mode
    // app_main() — see app_main.c), so start_utc is meaningless on-device;
    // tools/capture_serial.py stamps receive time instead and records
    // "clock": "host" in the sidecar so it's never mistaken for SNTP-accurate.
    cJSON_AddNumberToObject(root, "start_utc", 0);
    cJSON_AddNumberToObject(root, "i2s_drops_cumulative", i2s_drops_cumulative);
    cJSON_AddNumberToObject(root, "transport_drops_cumulative", transport_drops_cumulative);
    cJSON_AddStringToObject(root, "session_note", "");
    char *payload = cJSON_PrintUnformatted(root);
    if (payload != NULL) {
        send_frame_into(scratch, scratch_cap, FRAME_TYPE_SEGMENT_HEADER, 0, payload,
                        strlen(payload), pdMS_TO_TICKS(2000));
        cJSON_free(payload);
    }
    cJSON_Delete(root);
}

// ---- streaming task -------------------------------------------------------------

static void capture_stream_task(void *arg) {
    (void)arg;

    const uint32_t segment_samples =
        (uint32_t)CONFIG_ECHO_SENTRY_CAPTURE_SEGMENT_SECONDS * ECHO_SENTRY_SAMPLE_RATE_HZ;
    const uint32_t ring_capacity = audio_capture_ring_capacity();

    const size_t scratch_cap = FRAME_OVERHEAD + (CHUNK_SAMPLES * sizeof(int16_t));
    uint8_t *scratch = malloc(scratch_cap);
    int16_t *pcm_buf = malloc(CHUNK_SAMPLES * sizeof(int16_t));
    if (scratch == NULL || pcm_buf == NULL) {
        ESP_LOGE(TAG, "capture_stream_task: allocation failed");
        free(scratch);
        free(pcm_buf);
        vTaskDelete(NULL);
        return;
    }

    send_handshake(scratch, scratch_cap);

    // Wait for the ring to hold one full segment before the first read — same pattern
    // as app_main.c's production score_loop_task waiting for CLIP_SPAN_SAMPLES.
    while (audio_capture_write_count() < segment_samples) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    uint32_t transport_drop_count = 0;
    uint32_t cursor = audio_capture_write_count() - segment_samples;

    while (1) {
        send_segment_header(scratch, scratch_cap, audio_capture_drop_count(),
                            transport_drop_count);

        uint32_t remaining = segment_samples;
        while (remaining > 0) {
            uint32_t n = remaining < CHUNK_SAMPLES ? remaining : CHUNK_SAMPLES;
            audio_capture_read_at(cursor, pcm_buf, n);

            // audio_capture_read_at() does not itself detect a torn read (see
            // audio_capture.h) — if production outran this task while it was
            // reading or blocked on the previous send, the copy just taken may mix
            // old and overwritten samples. Detect it here and resync into the
            // middle of the ring rather than silently serve corrupt audio.
            uint32_t write_count = audio_capture_write_count();
            if (write_count - cursor > ring_capacity) {
                transport_drop_count++;
                cursor = write_count - ring_capacity / 2;
                continue;  // retry this slice at the resynced cursor
            }

            bool ok = send_frame_into(scratch, scratch_cap, FRAME_TYPE_PCM_CHUNK, cursor,
                                      pcm_buf, n * sizeof(int16_t), pdMS_TO_TICKS(200));
            if (!ok) {
                transport_drop_count++;
            }
            cursor += n;
            remaining -= n;
        }
    }
}

esp_err_t capture_stream_start(void) {
    usb_serial_jtag_driver_config_t usj_cfg = {
        .tx_buffer_size = 8192,
        .rx_buffer_size = 256,
    };
    esp_err_t err = usb_serial_jtag_driver_install(&usj_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "usb_serial_jtag_driver_install failed: %s", esp_err_to_name(err));
        return err;
    }
    s_driver_ready = true;

    // Route ESP_LOG through the framing so the host can tell log lines from PCM/
    // handshake/segment frames on the shared pipe. Deliberately NOT
    // usb_serial_jtag_vfs_use_driver(): that would redirect stdio/printf through this
    // same ring buffer too, and nothing in this build's capture path uses stdio — only
    // ESP_LOG, which this hook already intercepts directly.
    esp_log_set_vprintf(capture_log_vprintf);

    BaseType_t ok = xTaskCreatePinnedToCore(capture_stream_task, "capture_stream", 4096,
                                            NULL, 4, NULL, 0);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "failed to create capture_stream task");
        return ESP_FAIL;
    }
    return ESP_OK;
}

#endif  // CONFIG_ECHO_SENTRY_CAPTURE_MODE
