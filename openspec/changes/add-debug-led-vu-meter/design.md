# Design

## Context

See proposal.md — Why. The relevant existing structure:

- `audio_capture.c` owns an I2S capture task (priority 10, pinned to core 1) that fills
  a PSRAM ring of `ECHO_SENTRY_EVIDENCE_RING_SECONDS * 2` seconds. Its public API is
  `audio_capture_write_count()`, `audio_capture_read_at()`, `audio_capture_drop_count()`
  and `audio_capture_ring_capacity()`.
- `app_main.c`'s `score_loop_task` (priority 5, core 0) already computes exactly the
  quantity this indicator needs — an RMS in dBFS over the ring — but only once per
  10-second scoring interval.
- `app_main()` calls `audio_capture_start()` in both the capture-mode branch and the
  production branch, then the capture branch returns before any network is brought up.
- `audio_capture_read_at()` blocks (polling at 2 ms) until the requested samples exist.
  `audio_capture.h` documents that a reader lagging the write head by more than the
  ring capacity may receive a torn mix of old and overwritten samples, and that
  detecting this is the caller's responsibility.
- Nothing in the firmware currently drives a GPIO other than the three I2S pins, and
  nothing uses the RMT peripheral.

Hardware constraint: the ESP32-S3-DevKitC-1 has exactly one controllable LED, an
addressable WS2812 on GPIO38 (board v1.1) or GPIO48 (v1.0). The only other LED is the
3.3 V power LED, hardwired to the rail. There is no plain GPIO-driven user LED.

## Goals / Non-Goals

**Goals:**
- Add the indicator without touching `audio_capture.c`, the score loop, or any module
  on the inference or telemetry path.
- Make every failure mode in the spec distinguishable on a single-pixel indicator.
- Keep the level mapping consistent with the `rms_dbfs` field already published in each
  score payload, so the indicator is a live view of a number the hub also receives
  rather than a second, differently-defined measurement.

**Non-Goals:**
- Sub-block timing accuracy. The indicator is read by a human eye; a block-quantized
  update at roughly 31 Hz is sufficient and no effort is spent on finer granularity.
- Reusing the score loop's existing `rms_dbfs`. One sample every 10 seconds cannot
  animate an indicator, and coupling the two would mean the indicator's update rate was
  set by the detector's scoring interval.
- A test harness for firmware. None exists in this project; see the Verification
  decision below for what replaces it.

## Decisions

### Read the ring from a separate low-priority task

A new task (priority 1, core 0, 3072-byte stack) consumes the ring through the existing
public API. Core 1 is left to the I2S capture task.

*Alternative considered:* accumulate sum-of-squares and a running peak inside
`i2s_capture_task` itself, which would avoid re-reading samples. Rejected because it
adds work to the priority-10 task whose drop count is milestone M1's acceptance
criterion ("zero I2S drops over 1 h"). The indicator exists to diagnose that task; it
must not be able to perturb it. Re-reading 32 ms of int16 from PSRAM every 32 ms is
negligible by comparison.

*Alternative considered:* a hook or callback registered into `audio_capture.c`. Rejected
as more coupling for no benefit — the existing API is already sufficient.

### Check availability before reading, never block

The loop computes `avail = audio_capture_write_count() - cursor` and only calls
`audio_capture_read_at()` when `avail >= BLOCK_SAMPLES`. Otherwise it sleeps 10 ms and
re-checks, tracking how long it has been starved.

This is not an optimization — it is what makes the stall indication possible at all.
`audio_capture_read_at()` blocks until its samples exist, so a hung I2S channel would
park the indicator task inside that call forever and freeze the indicator on its last
color. The stall would be the one fault that cannot be displayed. Checking first keeps
the task live and lets the starvation timer drive the stall state.

### Consume contiguous blocks, with an explicit torn-read guard

The task keeps its own `cursor` and advances it by exactly `BLOCK_SAMPLES` per iteration
rather than always reading the newest block. Contiguous consumption means no sample
escapes the clipping detector through a sampling gap, which matters because clipping is
often a brief event.

Falling behind is handled per `audio_capture.h`'s documented contract: when
`write_count() - cursor > audio_capture_ring_capacity() / 2`, the cursor resynchronizes
to `write_count() - BLOCK_SAMPLES`. Half capacity is a deliberately early trigger — the
ring is tens of seconds and the task is 32 ms of work per 32 ms, so reaching even half
capacity means something is badly wrong and a resync is strictly better than rendering
a torn read.

### Drive the onboard WS2812 via `espressif/led_strip`

*Alternative considered:* an external LED and resistor on a free GPIO driven by LEDC.
This is the smaller software change — LEDC is in the `driver` component already listed
in `REQUIRES`, needs no new dependency, and gives 13-bit duty resolution instead of 8
bits per channel. Rejected because it requires soldering before the feature works at
all, and the hue axis of the RGB LED turns out to carry real weight (see below).

Consequence accepted: a new managed dependency, one RMT TX channel, and 8-bit intensity
resolution per channel.

API verified against the component source at version `3.1.0~1`, which declares
`idf: ">=5.2"` — satisfied by this project's 5.5.5 and by its own stricter `>=5.5` pin.
Initialization is `led_strip_new_rmt_device(&strip_config, &rmt_config, &handle)`. Two
fields of `led_strip_config_t` are easy to get wrong and are pinned here so they are not
rediscovered at the bench:

- `color_component_format` MUST be set to `LED_STRIP_COLOR_COMPONENT_FMT_GRB`. It has no
  usable zero value: `num_components` and `bytes_per_color` do fall back to 3 and 1, but
  `r_pos`/`g_pos`/`b_pos` all stay 0, so all three channels read the same byte and the
  pixel renders a wrong color rather than failing visibly.
- `timings` may be left uninitialized, but only because `led_model` is
  `LED_MODEL_WS2812`; model defaults apply. It is required only for `LED_MODEL_CUSTOM`.

`led_strip_rmt_config_t` takes defaults throughout for a single pixel:
`clk_src = RMT_CLK_SRC_DEFAULT`, `resolution_hz = 10 * 1000 * 1000`,
`mem_block_symbols = 0` (driver chooses), `flags.with_dma = false`.

The component also offers `led_strip_refresh_async()`. Not used: transmitting 24 bits to
one pixel is microseconds of RMT time, so the synchronous `led_strip_refresh()` costs
nothing worth managing a completion callback for.

### Map dBFS to intensity, then gamma-correct, then clamp to a minimum-on floor

The mapping is three composed stages, each correcting a different nonlinearity:

```
  rms over 512 samples
    -> dbfs  = 20*log10(rms / 32768)
    -> x     = clamp((dbfs - FLOOR) / (CEIL - FLOOR), 0, 1)
    -> env   = one-pole smoothed x
    -> value = max(MIN_ON, MAX_V * env^GAMMA)
```

`value` is the V of an HSV triple handed to `led_strip_set_pixel_hsv()`, so this stage
produces the final drive value directly with no RGB arithmetic of its own.

The dB stage fixes the *audio* dynamic range: the microphone's ambient range occupies
well under 1% of linear full scale, so a linear map is dark everywhere useful. The gamma
stage fixes the *eye*: perceived brightness is not linear in drive level. Both are
needed; they are independent physical stages, not alternatives.

`MIN_ON` is not cosmetic. With the defaults below, a live INMP441 at its own noise floor
in a quiet room produces:

```
  dbfs = -87          (datasheet: -26 dBFS @ 94 dB SPL, 61 dB SNR)
  x    = (-87 + 95) / (-10 + 95)  = 0.094
  x^2.2                            = 0.0056
  * MAX_V (64)                     = 0.36   -> rounds to 0
```

Gamma correction crushes exactly the region where "the microphone is alive" lives, which
would make the spec's absence-of-signal requirement unsatisfiable. Clamping to
`MIN_ON = 8/255` costs one line and restores it.

### Hue carries the coarse zone, intensity carries the fine detail

A single pixel cannot express position, so the bar-of-LEDs approach from the audio
metering literature (one LED per 3 dB) is unavailable. Instead a five-entry zone table
selects hue by dBFS while intensity ramps continuously across the whole window:

| dBFS band | Hue | Meaning |
|---|---|---|
| no signal at all | off | dead or mis-wired microphone |
| below −80 | blue | microphone alive, at its noise floor |
| −80 to −55 | green | normal room |
| −55 to −30 | yellow | loud |
| −30 to CEIL | red | approaching full scale |

Discrete zones with no interpolation, because a readable five-way distinction is the
point and a continuous hue sweep would be harder to judge, not easier. This also means
no level reading ever depends on the crushed bottom of the 8-bit intensity ramp.

The component's `led_strip_set_pixel_hsv(handle, 0, hue, saturation, value)` takes this
decomposition directly — hue in degrees (blue 240, green 120, yellow 60, red 0, magenta
300), saturation 255, and the computed `value`. Chosen over `led_strip_set_pixel()`
because the zone-to-hue-to-RGB conversion and the scaling of each component by the level
is precisely what the HSV setter already does; writing an RGB triple table plus a
per-component multiply would duplicate tested component code and add a class of bug for
nothing. Clipping's white becomes `saturation = 0` rather than a special-cased triple.

### Fault states take precedence, as distinct patterns

Precedence order, highest first — faults override level because a level rendered from a
stalled or lossy capture path is misleading:

| Condition | Appearance |
|---|---|
| `write_count()` still 0 (pre-first-sample) | off, and the starvation timer is not armed |
| `write_count()` frozen > `STALL_MS` | magenta, slow blink (~1 Hz) |
| `drop_count()` increased since last block | magenta, fast blink (~5 Hz) |
| any `abs(sample) >= CLIP_LIMIT` | white, solid, held `CLIP_HOLD_MS` |
| otherwise | hue and intensity from the level mapping |

White for clipping rather than red, because the top level zone is already red and the
spec requires clipping to be distinct from loud. Magenta for both capture faults because
they are the same class of problem, separated by blink rate.

Not arming the starvation timer until the first sample arrives is required by the spec's
startup scenario — otherwise every boot begins with a spurious stall indication during
the window between task start and `audio_capture_start()` filling the ring.

### One call site, above the capture-mode branch

`debug_led_start()` is called once at the top of `app_main()`, before the
`#if CONFIG_ECHO_SENTRY_CAPTURE_MODE` branch, so both builds reach it with no
preprocessor duplication. This is safe even though it precedes `audio_capture_start()`:
the task's first action is to wait for `audio_capture_write_count()` to become non-zero,
which cannot happen until the ring is allocated. `score_loop_task` already uses this
same wait-for-the-ring pattern.

### Constants and ballistics

| Constant | Value | Rationale |
|---|---|---|
| `BLOCK_SAMPLES` | 512 | 32 ms at 16 kHz; matches the frontend's window stride |
| update rate | ~31 Hz | one block per iteration; well above flicker perception for a smoothly varying value |
| `FLOOR` | −95 dBFS | below the microphone's −87 dBFS noise floor, so "alive but quiet" stays distinguishable from "dead" |
| `CEIL` | −10 dBFS | close enough to full scale that red genuinely means near-clipping |
| `GAMMA` | 2.2 | standard perceptual exponent; the useful range is roughly 1.5–3 |
| `MAX_V` | 64/255 | caps WS2812 draw to single-digit mA and keeps the pixel readable in a dark room |
| `MIN_ON` | 8/255 | lowest reliably visible WS2812 value |
| `CLIP_LIMIT` | 31130 | 0.95 of int16 full scale |
| `CLIP_HOLD_MS` | 150 | long enough to register on a glance |
| `STALL_MS` | 200 | ~6 missed blocks; past any plausible scheduling jitter |

One-pole smoothing, `env += alpha * (x - env)`, with alpha chosen per direction:

| Direction | Time constant | alpha at 32 ms |
|---|---|---|
| rising | 30 ms | 0.66 |
| falling | 130 ms | 0.22 |

A 130 ms time constant reaches 90% in about 300 ms, which is the IEC 60268-17 VU
ballistic. The asymmetric fast attack is the practical departure from a true VU meter:
it makes onsets visible without giving up the averaged decay that suits stationary
machinery.

`FLOOR`, `CEIL`, `MAX_V` and the output GPIO become Kconfig values because they are the
four that need bench calibration or differ by board revision. The rest are compile-time
constants in `debug_led.c` — adding knobs nobody will turn is a cost, not a feature.

### Verification: a pure mapping function plus boot-time assertions

The project has no firmware test harness (only `hub/tests` and `training/tests`), and
`tools/build_native_frontend.sh` shows that host-compiling firmware sources is reserved
for the frontend, where skew against the training pipeline would be a correctness bug.
Standing up equivalent infrastructure for a debug indicator is not proportionate.

Instead, the mapping is written as a pure function with no hardware or state
dependencies:

```c
// -> {hue_degrees, saturation, value}, fed straight to led_strip_set_pixel_hsv()
static led_hsv_t level_to_hsv(float dbfs, float env);
```

and a `debug_led_selftest()` of roughly eight assertions runs once at task start:
monotonic intensity across the window, both endpoints clamped, no-signal maps to off,
and — the case this design talked itself into needing — a level just above `FLOOR` maps
to a non-zero intensity. That last assertion is the one that would have caught the
gamma-crush bug before hardware. Ten lines, no new infrastructure, and it fails loudly
on the mistakes that are actually likely here: an inverted scale or a crushed floor.

## Risks / Trade-offs

**WS2812 switching current shares the 3V3 rail with the microphone** → The INMP441 is a
digital-output part, so the exposure is limited to supply ripple reaching its internal
preamp rather than direct injection into an analog signal path. `MAX_V` caps the draw at
single-digit mA. Verified by measurement, not assumption: compare the published
`rms_dbfs` in a quiet room with the pixel forced dark against forced to maximum. If the
noise floor moves, lower `MAX_V` or move to the external-LED alternative.

**8-bit intensity resolution with gamma 2.2 leaves few distinguishable steps** →
Mitigated structurally by the hue-carries-the-zone decision, so no reading depends on
the bottom of the ramp, plus the `MIN_ON` clamp. Quantified by running the mapping on
the host at the default window: everything below about −58 dBFS pins at `MIN_ON`, so the
whole blue and green range (−95 to −58, roughly the bottom 43% of the window) is
separated by hue alone and not by intensity. Perceptually the rest is correct — mid
window renders drive 14/64, which is half maximum brightness to the eye — so this is the
8-bit floor, not a mapping error. The lever is calibration, not `GAMMA`: narrowing
`FLOOR` toward the room's measured ambient moves that ambient up the ramp, which is what
task 6.2 is for. Lowering `GAMMA` toward 1.8 barely moves the pin point (−58 to −56);
only a near-linear curve would, at the cost of crushing the top instead.

**Datasheet-derived defaults, never measured in a real room** → All four
calibration-sensitive values are Kconfig. Expect one bench pass before the indicator
reads usefully; this is setup, not a defect.

**Wrong board revision presents exactly like a software bug** → A wrong pin shows up as
"the LED never lights", indistinguishable from a broken driver or a dead task. Mitigated
by making the pin a Kconfig value with the v1.0 default documented, and by confirming
the revision against the board silkscreen before debugging anything else.

**A production node carries a permanently lit indicator, the new dependency, and an RMT
channel** → Accepted deliberately. The counter-argument is that this project's
established pattern makes capture-versus-production differences structural rather than
conventional. Reversing it later is a `default y if ECHO_SENTRY_CAPTURE_MODE` in Kconfig
and one `#if` — two lines, no redesign.

**In a capture build the indicator freezes for ~2 s at the first segment boundary** →
Measured on hardware 2026-10-16: the priority-4 `capture_stream` task holds core 0 for
1.84 s while it pushes its first PCM segment over USB, and the priority-1 indicator
task, pinned to the same core, is preempted for that whole window and holds its last
color. Observed exactly once per boot — it did not recur over the following 900 blocks.
Accepted, because the alternative is raising the indicator above the transport and the
whole point of priority 1 on core 0 is that a diagnostic LED must not be able to perturb
capture. It does not produce a false stall: preemption makes the available-sample count
larger, not smaller, so the starvation timer stays at zero. The consequence is narrow
but worth knowing before debugging it twice — the "tracks level smoothly with no visible
flicker" expectation in task 5.2 holds for a production build and not for the first two
seconds of a capture build's first segment.

**First build after this change requires network** → `espressif/led_strip` is pulled
from the Espressif component registry and recorded in `dependencies.lock`. Offline
builds work normally once `managed_components/` is populated.

## Migration Plan

No migration. The change is additive, affects no persisted state, no NVS layout, no
partition table, and no MQTT payload. Rollback is reverting the commit and rebuilding;
nothing on a node or the hub needs to be reset.

Deploy order is unconstrained — a node running this firmware is wire-compatible with a
hub that has never heard of it, because nothing about the indicator crosses the network.
