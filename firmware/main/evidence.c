#include "evidence.h"

#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_random.h"

#include "audio_capture.h"
#include "gen/frontend_config.h"
#include "provisioning.h"

static const char *TAG = "evidence";

#define NUM_SAMPLES (ECHO_SENTRY_SAMPLE_RATE_HZ * CONFIG_ECHO_SENTRY_EVIDENCE_RING_SECONDS)
#define WAV_HEADER_SIZE 44
#define BOUNDARY "----echoSentryEvidence"

static void write_wav_header(uint8_t *buf, uint32_t data_bytes) {
    const uint32_t sample_rate = ECHO_SENTRY_SAMPLE_RATE_HZ;
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t riff_size = 36 + data_bytes;
    const uint32_t fmt_size = 16;
    const uint16_t audio_format = 1;  // PCM

    memcpy(buf + 0, "RIFF", 4);
    memcpy(buf + 4, &riff_size, 4);
    memcpy(buf + 8, "WAVE", 4);
    memcpy(buf + 12, "fmt ", 4);
    memcpy(buf + 16, &fmt_size, 4);
    memcpy(buf + 20, &audio_format, 2);
    memcpy(buf + 22, &channels, 2);
    memcpy(buf + 24, &sample_rate, 4);
    memcpy(buf + 28, &byte_rate, 4);
    memcpy(buf + 32, &block_align, 2);
    memcpy(buf + 34, &bits_per_sample, 2);
    memcpy(buf + 36, "data", 4);
    memcpy(buf + 40, &data_bytes, 4);
}

esp_err_t evidence_capture_and_upload(const char *trigger) {
    const node_config_t *cfg = provisioning_get();
    const uint32_t data_bytes = NUM_SAMPLES * sizeof(int16_t);

    uint8_t *wav = heap_caps_malloc(WAV_HEADER_SIZE + data_bytes, MALLOC_CAP_SPIRAM);
    if (wav == NULL) {
        ESP_LOGE(TAG, "failed to allocate %u bytes for evidence WAV",
                (unsigned)(WAV_HEADER_SIZE + data_bytes));
        return ESP_ERR_NO_MEM;
    }
    write_wav_header(wav, data_bytes);

    uint32_t write_count = audio_capture_write_count();
    uint32_t start = (write_count > NUM_SAMPLES) ? write_count - NUM_SAMPLES : 0;
    audio_capture_read_at(start, (int16_t *)(wav + WAV_HEADER_SIZE), NUM_SAMPLES);

    char clip_id[16];
    snprintf(clip_id, sizeof(clip_id), "%08x", (unsigned)esp_random());

    char part_header[512];
    int part_header_len = snprintf(
        part_header, sizeof(part_header),
        "--" BOUNDARY "\r\nContent-Disposition: form-data; name=\"trigger\"\r\n\r\n%s\r\n"
        "--" BOUNDARY "\r\nContent-Disposition: form-data; name=\"clip_id\"\r\n\r\n%s\r\n"
        "--" BOUNDARY
        "\r\nContent-Disposition: form-data; name=\"clip\"; filename=\"clip.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n",
        trigger, clip_id);
    const char part_footer[] = "\r\n--" BOUNDARY "--\r\n";
    int part_footer_len = sizeof(part_footer) - 1;

    size_t total_len = part_header_len + WAV_HEADER_SIZE + data_bytes + part_footer_len;
    uint8_t *body = heap_caps_malloc(total_len, MALLOC_CAP_SPIRAM);
    if (body == NULL) {
        heap_caps_free(wav);
        ESP_LOGE(TAG, "failed to allocate %u bytes for the multipart body",
                (unsigned)total_len);
        return ESP_ERR_NO_MEM;
    }
    memcpy(body, part_header, part_header_len);
    memcpy(body + part_header_len, wav, WAV_HEADER_SIZE + data_bytes);
    memcpy(body + part_header_len + WAV_HEADER_SIZE + data_bytes, part_footer,
          part_footer_len);
    heap_caps_free(wav);

    char url[96];
    snprintf(url, sizeof(url), "http://%s:%u/evidence/%s", cfg->hub_host,
            cfg->evidence_port, cfg->node_id);

    esp_http_client_config_t http_cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 15000,
    };
    esp_http_client_handle_t client = esp_http_client_init(&http_cfg);
    esp_http_client_set_header(client, "Content-Type",
                               "multipart/form-data; boundary=" BOUNDARY);
    esp_http_client_set_post_field(client, (const char *)body, (int)total_len);

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "evidence uploaded (trigger=%s), status=%d", trigger,
                esp_http_client_get_status_code(client));
    } else {
        ESP_LOGW(TAG, "evidence upload failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
    heap_caps_free(body);
    return err;
}
