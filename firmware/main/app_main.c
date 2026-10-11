// Init order + task wiring — comprehensive-plan.md §2.3.
//
// CONFIG_ECHO_SENTRY_CAPTURE_MODE (see Kconfig.projbuild) is a separate build, not a
// runtime toggle — app_main() branches at compile time below. The capture branch
// brings up no network at all (no telemetry_init(), no inference_init(), no
// net_start()), which is what makes Kconfig.projbuild's promise — "a capture build
// can never accidentally ship raw audio to the hub" — structurally true rather than a
// convention. See capture_stream.c for the device side and
// tools/capture_serial.py for the host side (comprehensive-plan.md §3.1).

#include <math.h>
#include <string.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "audio_capture.h"
#include "capture_stream.h"
#include "detector.h"
#include "evidence.h"
#include "frontend_core.h"
#include "gen/frontend_config.h"
#include "inference.h"
#include "net.h"
#include "provisioning.h"
#include "telemetry.h"

static const char *TAG = "app_main";

// comprehensive-plan.md's MQTT topic table (§2.2(b)): a score every 10s. Not
// currently sourced from a shared config value — see hub/src/echo_sentry_hub/
// config.py's DetectorConfig.score_interval_s, which this should eventually match
// by construction rather than by two people remembering to keep "10" in sync.
#define SCORE_INTERVAL_MS 10000

#define WINDOW_SAMPLE_COUNT (ECHO_SENTRY_WINDOW_SIZE_MS * ECHO_SENTRY_SAMPLE_RATE_HZ / 1000)
#define WINDOW_STRIDE_SAMPLES (ECHO_SENTRY_WINDOW_STRIDE_MS * ECHO_SENTRY_SAMPLE_RATE_HZ / 1000)
#define CLIP_SPAN_SAMPLES \
    (((ECHO_SENTRY_CLIP_FRAMES - 1) * WINDOW_STRIDE_SAMPLES) + WINDOW_SAMPLE_COUNT)

static void score_loop_task(void *arg) {
    (void)arg;

    void *frontend = echo_sentry_frontend_create();
    if (frontend == NULL) {
        ESP_LOGE(TAG, "echo_sentry_frontend_create failed");
        vTaskDelete(NULL);
        return;
    }

    int16_t *window_buf = malloc(WINDOW_SAMPLE_COUNT * sizeof(int16_t));
    int16_t *frames = malloc(ECHO_SENTRY_CLIP_FRAMES * ECHO_SENTRY_MEL_CHANNELS * sizeof(int16_t));
    if (window_buf == NULL || frames == NULL) {
        ESP_LOGE(TAG, "failed to allocate score-loop scratch buffers");
        vTaskDelete(NULL);
        return;
    }

    // Wait for the ring to fill enough for one full clip span before the first score.
    while (audio_capture_write_count() < (uint32_t)CLIP_SPAN_SAMPLES) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    TickType_t last_wake = xTaskGetTickCount();
    while (1) {
        uint32_t write_count = audio_capture_write_count();
        uint32_t clip_start = write_count - (uint32_t)CLIP_SPAN_SAMPLES;

        double sum_sq = 0.0;
        for (int f = 0; f < ECHO_SENTRY_CLIP_FRAMES; ++f) {
            uint32_t frame_start = clip_start + (uint32_t)(f * WINDOW_STRIDE_SAMPLES);
            audio_capture_read_at(frame_start, window_buf, WINDOW_SAMPLE_COUNT);
            for (int i = 0; i < WINDOW_SAMPLE_COUNT; ++i) {
                sum_sq += (double)window_buf[i] * (double)window_buf[i];
            }
            echo_sentry_frontend_process_frame(
                frontend, window_buf, frames + f * ECHO_SENTRY_MEL_CHANNELS);
        }

        double rms = sqrt(sum_sq / (ECHO_SENTRY_CLIP_FRAMES * WINDOW_SAMPLE_COUNT));
        float rms_dbfs = (rms > 1.0) ? 20.0f * log10f((float)rms / 32768.0f) : -120.0f;

        inference_clip_result_t result =
            inference_score_clip(frames, ECHO_SENTRY_CLIP_FRAMES);

        telemetry_publish_score(result, rms_dbfs, audio_capture_drop_count());

        if (detector_should_capture_evidence(result.err_mean)) {
            evidence_capture_and_upload("fast_path_threshold");
        }

        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SCORE_INTERVAL_MS));
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "Echo Sentry node starting");

    ESP_ERROR_CHECK(provisioning_init());

#if CONFIG_ECHO_SENTRY_CAPTURE_MODE
    ESP_ERROR_CHECK(audio_capture_start());
    ESP_ERROR_CHECK(capture_stream_start());
    return;
#endif

    ESP_ERROR_CHECK(telemetry_init());
    ESP_ERROR_CHECK(audio_capture_start());

    if (inference_init() != 0) {
        ESP_LOGE(TAG, "inference_init failed — no valid embedded model "
                     "(main/models/README.md); halting score loop");
    } else {
        // net_start() blocks until Wi-Fi + SNTP + MQTT are up, matching
        // comprehensive-plan.md's join-the-Pi's-AP topology (§2.1).
        ESP_ERROR_CHECK(net_start(telemetry_handle_cmd));

        BaseType_t ok = xTaskCreatePinnedToCore(score_loop_task, "score_loop", 8192,
                                                NULL, 5, NULL, 0);
        if (ok != pdPASS) {
            ESP_LOGE(TAG, "failed to create score_loop task");
        }
    }
}
