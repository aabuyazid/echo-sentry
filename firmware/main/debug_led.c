// Live audio-level and capture-health indicator on the board's one addressable RGB LED
// — the diagnostic channel available during hardware bring-up and in a field equipment
// room where nothing is attached to the node. A dead or mis-wired INMP441 still clocks
// data and is indistinguishable from a silent room over the serial console, so the
// sample *values* are the only discriminator; that is what this module renders.
//
// Hue carries the coarse zone, intensity the fine detail (a single pixel cannot express
// position, so the one-LED-per-3-dB bar from the audio metering literature is out):
//
//   | condition                                  | appearance                     |
//   |--------------------------------------------|--------------------------------|
//   | no samples captured yet (boot)             | off, stall timer not yet armed |
//   | write count frozen > STALL_MS              | magenta, slow blink (~1 Hz)    |
//   | drop count rose since the last block       | magenta, fast blink (~5 Hz)    |
//   | any abs(sample) >= CLIP_LIMIT              | white, solid, held CLIP_HOLD_MS|
//   | every sample identically zero              | off — dead or mis-wired mic    |
//   | below -80 dBFS                             | blue  — mic alive, noise floor |
//   | -80..-55 dBFS                              | green — normal room            |
//   | -55..-30 dBFS                              | yellow — loud                  |
//   | -30 dBFS and up                            | red   — approaching full scale |
//
// Faults outrank level because a level rendered from a stalled or lossy capture path is
// misleading. White (saturation 0) for clipping because the top level zone is already
// red; magenta for both capture faults because they are the same class of problem,
// separated by blink rate.
//
// Why the availability check precedes every read: audio_capture_read_at() *blocks* until
// its samples exist (see audio_capture.h), so a hung I2S channel would park this task
// inside that call and freeze the indicator on its last color — the stall would be the
// one fault that cannot be displayed. Checking audio_capture_write_count() first keeps
// the task live and lets the starvation timer drive the stall state. This is a
// correctness requirement, not an optimization.
//
// Why MIN_ON exists: gamma correction crushes exactly the region where "the microphone
// is alive" lives. A live INMP441 at its own noise floor in a quiet room gives
// dbfs = -87 -> x = 0.094 -> x^2.2 = 0.0056 -> x 64 = 0.36, which rounds to 0. Without
// the clamp, "LED dark" would mean both "no signal" and "signal too quiet to render",
// and a dead microphone would be undiagnosable.
//
// Calibration (defaults are datasheet-derived and have never seen a real room — expect
// one bench pass; this is setup, not a defect):
//   1. Confirm the board revision on the silkscreen and set ECHO_SENTRY_DEBUG_LED_GPIO
//      (38 for DevKitC-1 v1.1, 48 for v1.0). A wrong pin presents exactly like a broken
//      driver, so get the LED lighting before debugging anything else.
//   2. Read the rms_dbfs the score loop logs for the target room's normal ambient and
//      set FLOOR/CEIL so that ambient lands mid-window, keeping FLOOR below the
//      microphone's own noise floor so "alive but quiet" stays distinct from "dead".
//   3. Compare rms_dbfs in a quiet room with the pixel forced dark against forced to
//      MAX_INTENSITY. The LED shares the 3V3 rail with the microphone; if the noise
//      floor moves, lower ECHO_SENTRY_DEBUG_LED_MAX_INTENSITY and re-measure.
//   4. If the ramp still reads as crushed on hardware, lower GAMMA toward 1.8.

#include "debug_led.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "led_strip.h"

#include "audio_capture.h"
#include "gen/frontend_config.h"

static const char *TAG = "debug_led";

#define BLOCK_SAMPLES 512  // 32 ms at 16 kHz; matches the frontend's window stride
#define BLOCK_MS (BLOCK_SAMPLES * 1000 / ECHO_SENTRY_SAMPLE_RATE_HZ)
#define POLL_MS 10  // re-check interval while starved of samples

#define FLOOR_DBFS ((float)CONFIG_ECHO_SENTRY_DEBUG_LED_FLOOR_DBFS)
#define CEIL_DBFS ((float)CONFIG_ECHO_SENTRY_DEBUG_LED_CEIL_DBFS)
#define MAX_V CONFIG_ECHO_SENTRY_DEBUG_LED_MAX_INTENSITY

#define MIN_ON 8          // lowest reliably visible WS2812 value
#define GAMMA 2.2f        // standard perceptual exponent; useful range is ~1.5-3
#define CLIP_LIMIT 31130  // 0.95 of int16 full scale
#define CLIP_HOLD_MS 150  // long enough to register on a glance
#define STALL_MS 200      // ~6 missed blocks; past any plausible scheduling jitter

// One-pole ballistics. The 130 ms falling constant reaches 90% in ~300 ms, the
// IEC 60268-17 VU ballistic; the fast attack is the deliberate departure from a true VU
// meter, so onsets stay visible without giving up a decay that suits stationary
// machinery.
#define ALPHA_RISE 0.66f  // tau 30 ms at a 32 ms block rate
#define ALPHA_FALL 0.22f  // tau 130 ms

// Hue in degrees, as led_strip_set_pixel_hsv() takes it (it computes hue / 60).
#define HUE_RED 0
#define HUE_YELLOW 60
#define HUE_GREEN 120
#define HUE_BLUE 240
#define HUE_MAGENTA 300

#define STALL_BLINK_HALF_MS 500  // ~1 Hz
#define DROP_BLINK_HALF_MS 100   // ~5 Hz

typedef struct {
    uint16_t hue;
    uint8_t saturation;
    uint8_t value;
} led_hsv_t;

static led_strip_handle_t s_strip;

// Pure: no globals, no ESP-IDF calls. `dbfs` picks the zone (and the all-zero sentinel),
// `env` is the smoothed, floor/ceiling-normalized level that drives intensity.
static led_hsv_t level_to_hsv(float dbfs, float env) {
    led_hsv_t out = {.hue = HUE_RED, .saturation = 255, .value = 0};

    if (!isfinite(dbfs)) {
        return out;  // every sample identically zero: dead or mis-wired microphone
    }

    out.hue = (dbfs < -80.0f)   ? HUE_BLUE
              : (dbfs < -55.0f) ? HUE_GREEN
              : (dbfs < -30.0f) ? HUE_YELLOW
                                : HUE_RED;

    float v = (float)MAX_V * powf(env, GAMMA);
    out.value = (v <= (float)MIN_ON)  ? MIN_ON
                : (v >= (float)MAX_V) ? MAX_V
                                      : (uint8_t)(v + 0.5f);
    return out;
}

// Pure. Asymmetric so onsets appear promptly but decay averages.
static float env_step(float env, float x) {
    float alpha = (x > env) ? ALPHA_RISE : ALPHA_FALL;
    return env + alpha * (x - env);
}

static float dbfs_to_x(float dbfs) {
    float x = (dbfs - FLOOR_DBFS) / (CEIL_DBFS - FLOOR_DBFS);
    if (!(x > 0.0f)) {
        return 0.0f;  // also catches -INFINITY and NaN
    }
    return (x > 1.0f) ? 1.0f : x;
}

// Catches the mistakes actually likely here: an inverted scale, and a floor crushed to
// fully dark by the gamma curve. Aborts the boot on failure, loudly, rather than
// shipping an indicator that lies about a dead microphone.
static void debug_led_selftest(void) {
    assert(CEIL_DBFS > FLOOR_DBFS);

    // No signal at all is fully dark; this is the state a dead microphone must produce,
    // and the only one that may be.
    assert(level_to_hsv(-INFINITY, 0.0f).value == 0);

    // Both endpoints clamp.
    assert(level_to_hsv(FLOOR_DBFS, dbfs_to_x(FLOOR_DBFS)).value == MIN_ON);
    assert(level_to_hsv(CEIL_DBFS, dbfs_to_x(CEIL_DBFS)).value == MAX_V);
    assert(level_to_hsv(FLOOR_DBFS - 20.0f, dbfs_to_x(FLOOR_DBFS - 20.0f)).value == MIN_ON);
    assert(level_to_hsv(CEIL_DBFS + 20.0f, dbfs_to_x(CEIL_DBFS + 20.0f)).value == MAX_V);

    // A live microphone at its own self-noise level stays visibly lit. This is the
    // assertion that would have caught the gamma-crush bug before hardware.
    assert(level_to_hsv(-87.0f, dbfs_to_x(-87.0f)).value > 0);

    // Intensity never decreases as level rises across the window.
    uint8_t prev = 0;
    for (int i = 0; i <= 100; ++i) {
        float dbfs = FLOOR_DBFS + ((float)i / 100.0f) * (CEIL_DBFS - FLOOR_DBFS);
        uint8_t v = level_to_hsv(dbfs, dbfs_to_x(dbfs)).value;
        assert(v >= prev);
        prev = v;
    }
    assert(prev == MAX_V);

    // Zone boundaries resolve to the documented hues.
    assert(level_to_hsv(-90.0f, 0.1f).hue == HUE_BLUE);
    assert(level_to_hsv(-80.0f, 0.2f).hue == HUE_GREEN);
    assert(level_to_hsv(-55.0f, 0.5f).hue == HUE_YELLOW);
    assert(level_to_hsv(-30.0f, 0.8f).hue == HUE_RED);

    // Falling ballistic: a full-scale-to-silence step decays 90% in ~300 ms.
    float env = 1.0f;
    int blocks = 0;
    while (env > 0.1f && blocks < 100) {
        env = env_step(env, 0.0f);
        blocks++;
    }
    assert(blocks * BLOCK_MS >= 250 && blocks * BLOCK_MS <= 350);

    // Rising ballistic: most of a step lands within the first block.
    assert(env_step(0.0f, 1.0f) > 0.6f);
}

static void show(uint16_t hue, uint8_t saturation, uint8_t value) {
    led_strip_set_pixel_hsv(s_strip, 0, hue, saturation, value);
    led_strip_refresh(s_strip);
}

// Phase from a free-running millisecond accumulator, so the blink rate does not depend
// on how many blocks the fault happens to last.
static void show_blink(uint16_t hue, uint32_t phase_ms, uint32_t half_period_ms) {
    bool on = ((phase_ms / half_period_ms) % 2) == 0;
    show(hue, 255, on ? MAX_V : 0);
}

static esp_err_t strip_init(void) {
    led_strip_config_t strip_config = {
        .strip_gpio_num = CONFIG_ECHO_SENTRY_DEBUG_LED_GPIO,
        .max_leds = 1,
        .led_model = LED_MODEL_WS2812,
        // Required: zeroed, every channel position falls to 0 and the pixel renders
        // garbage. `timings` may stay uninitialized only because the model is WS2812 —
        // it is required for LED_MODEL_CUSTOM.
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
    };
    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .mem_block_symbols = 0,   // driver picks
        .flags.with_dma = false,  // 24 bits to one pixel; DMA would be overhead
    };
    return led_strip_new_rmt_device(&strip_config, &rmt_config, &s_strip);
}

static void debug_led_task(void *arg) {
    (void)arg;
    static int16_t block[BLOCK_SAMPLES];  // static: 1 KB does not belong on this stack

    debug_led_selftest();

    esp_err_t err = strip_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "led_strip_new_rmt_device on GPIO%d failed: %s",
                 CONFIG_ECHO_SENTRY_DEBUG_LED_GPIO, esp_err_to_name(err));
        vTaskDelete(NULL);
        return;
    }
    led_strip_clear(s_strip);
    ESP_LOGI(TAG, "indicator up on GPIO%d, window %d..%d dBFS, max intensity %d",
             CONFIG_ECHO_SENTRY_DEBUG_LED_GPIO,
             CONFIG_ECHO_SENTRY_DEBUG_LED_FLOOR_DBFS,
             CONFIG_ECHO_SENTRY_DEBUG_LED_CEIL_DBFS, MAX_V);

    // Dark until the first sample exists, with the starvation timer unarmed, so the boot
    // window before audio_capture_start() fills the ring does not render as a stall.
    // score_loop_task uses this same wait-for-the-ring pattern.
    while (audio_capture_write_count() == 0) {
        vTaskDelay(pdMS_TO_TICKS(POLL_MS));
    }
    ESP_LOGI(TAG, "first samples captured; tracking level");

    uint32_t cursor = 0;
    uint32_t last_drops = audio_capture_drop_count();
    uint32_t starved_ms = 0;
    uint32_t clip_hold_ms = 0;
    uint32_t phase_ms = 0;
    float env = 0.0f;

    while (1) {
        uint32_t write_count = audio_capture_write_count();
        uint32_t avail = write_count - cursor;

        if (avail < BLOCK_SAMPLES) {
            vTaskDelay(pdMS_TO_TICKS(POLL_MS));
            starved_ms += POLL_MS;
            phase_ms += POLL_MS;
            if (starved_ms >= STALL_MS) {
                show_blink(HUE_MAGENTA, phase_ms, STALL_BLINK_HALF_MS);
            }
            continue;
        }
        starved_ms = 0;
        phase_ms += BLOCK_MS;

        // audio_capture.h's documented contract: a reader lagging further than the ring
        // can hold receives a torn mix of old and overwritten samples, and detecting it
        // is the caller's job. Half capacity is a deliberately early trigger — this task
        // is 32 ms of work per 32 ms, so reaching even half means something is badly
        // wrong and a resync beats rendering a torn read.
        if (avail > audio_capture_ring_capacity() / 2) {
            ESP_LOGW(TAG, "lagged %" PRIu32 " samples; resyncing to the write head",
                     avail);
            cursor = write_count - BLOCK_SAMPLES;
        }

        audio_capture_read_at(cursor, block, BLOCK_SAMPLES);
        cursor += BLOCK_SAMPLES;

        uint64_t sum_sq = 0;
        int32_t peak = 0;
        for (int i = 0; i < BLOCK_SAMPLES; ++i) {
            int32_t s = block[i];
            sum_sq += (uint64_t)(s * s);
            int32_t mag = (s < 0) ? -s : s;
            if (mag > peak) {
                peak = mag;
            }
        }
        // Same definition as the rms_dbfs in every score payload, so the indicator is a
        // live view of a number the hub also receives. -INFINITY is the all-zero
        // sentinel level_to_hsv() renders as fully dark.
        float rms = sqrtf((float)sum_sq / (float)BLOCK_SAMPLES);
        float dbfs = (rms > 0.0f) ? 20.0f * log10f(rms / 32768.0f) : -INFINITY;

        env = env_step(env, dbfs_to_x(dbfs));

        uint32_t drops = audio_capture_drop_count();
        bool dropping = (drops != last_drops);
        last_drops = drops;

        if (peak >= CLIP_LIMIT) {
            clip_hold_ms = CLIP_HOLD_MS;
        }

        // Precedence, highest first. Stall is handled in the starved branch above.
        if (dropping) {
            show_blink(HUE_MAGENTA, phase_ms, DROP_BLINK_HALF_MS);
        } else if (clip_hold_ms > 0) {
            show(HUE_RED, 0, MAX_V);  // saturation 0 is white
            clip_hold_ms = (clip_hold_ms > BLOCK_MS) ? clip_hold_ms - BLOCK_MS : 0;
        } else {
            led_hsv_t hsv = level_to_hsv(dbfs, env);
            show(hsv.hue, hsv.saturation, hsv.value);
        }
    }
}

esp_err_t debug_led_start(void) {
    // Priority 1 and core 0: the indicator exists to diagnose the priority-10 I2S
    // capture task on core 1 and must not be able to perturb it, since that task's drop
    // count is milestone M1's acceptance criterion.
    BaseType_t ok =
        xTaskCreatePinnedToCore(debug_led_task, "debug_led", 3072, NULL, 1, NULL, 0);
    return (ok == pdPASS) ? ESP_OK : ESP_ERR_NO_MEM;
}
