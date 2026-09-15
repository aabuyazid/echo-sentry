#include "audio_capture.h"

#include <string.h>

#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "gen/frontend_config.h"

static const char *TAG = "audio_capture";

// ~2x the evidence ring so the frontend task (which reads slightly behind the write
// head) always has margin, without needing exact synchronization with evidence
// snapshots — see the module comment in audio_capture.h.
#define RING_SECONDS (CONFIG_ECHO_SENTRY_EVIDENCE_RING_SECONDS * 2)
#define RING_SAMPLES (ECHO_SENTRY_SAMPLE_RATE_HZ * RING_SECONDS)

#define I2S_READ_CHUNK_FRAMES 256  // 32-bit frames per i2s_channel_read() call

static i2s_chan_handle_t s_rx_chan;
static int16_t *s_ring;  // PSRAM, RING_SAMPLES elements
static volatile uint32_t s_write_count;
static volatile uint32_t s_drop_count;

static void i2s_capture_task(void *arg) {
    (void)arg;
    int32_t raw[I2S_READ_CHUNK_FRAMES];

    while (1) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(s_rx_chan, raw, sizeof(raw), &bytes_read,
                                         portMAX_DELAY);
        if (err != ESP_OK) {
            s_drop_count++;
            ESP_LOGW(TAG, "i2s_channel_read failed: %s", esp_err_to_name(err));
            continue;
        }

        size_t frames = bytes_read / sizeof(int32_t);
        for (size_t i = 0; i < frames; ++i) {
            // INMP441: 24-bit sample, left-justified in the 32-bit I2S slot. Shifting
            // right by 16 keeps the sign and the most significant 16 bits of the
            // 24-bit sample — the standard truncation for this part; needs
            // hardware-level level/clipping verification (comprehensive-plan.md's
            // firmware verification table, "known 1 kHz tone through the full chain").
            int16_t sample = (int16_t)(raw[i] >> 16);
            s_ring[s_write_count % RING_SAMPLES] = sample;
            s_write_count++;
        }
    }
}

esp_err_t audio_capture_start(void) {
    s_ring = heap_caps_malloc(RING_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (s_ring == NULL) {
        ESP_LOGE(TAG, "failed to allocate %d-sample ring in PSRAM", RING_SAMPLES);
        return ESP_ERR_NO_MEM;
    }
    memset(s_ring, 0, RING_SAMPLES * sizeof(int16_t));

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(ECHO_SENTRY_SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_MONO),
        .gpio_cfg =
            {
                .mclk = I2S_GPIO_UNUSED,
                .bclk = CONFIG_ECHO_SENTRY_I2S_BCK_GPIO,
                .ws = CONFIG_ECHO_SENTRY_I2S_WS_GPIO,
                .dout = I2S_GPIO_UNUSED,
                .din = CONFIG_ECHO_SENTRY_I2S_DIN_GPIO,
                .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
            },
    };
    // L/R tied to GND (comprehensive-plan.md's INMP441 gotchas) -> left slot.
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_chan));

    BaseType_t ok = xTaskCreatePinnedToCore(i2s_capture_task, "i2s_capture", 4096, NULL,
                                            10, NULL, 1);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "failed to create i2s_capture task");
        return ESP_FAIL;
    }
    return ESP_OK;
}

uint32_t audio_capture_write_count(void) { return s_write_count; }

void audio_capture_read_at(uint32_t start, int16_t *out, size_t num_samples) {
    while ((uint32_t)(s_write_count - start) < num_samples) {
        vTaskDelay(pdMS_TO_TICKS(2));
    }
    for (size_t i = 0; i < num_samples; ++i) {
        out[i] = s_ring[(start + i) % RING_SAMPLES];
    }
}

uint32_t audio_capture_drop_count(void) { return s_drop_count; }
