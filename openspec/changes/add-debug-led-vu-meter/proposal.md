# Proposal

## Why

During hardware bring-up the node gives no local indication of whether the microphone
is working. The capture build deliberately brings up no network at all, so the only
diagnostic channels are a host-attached serial console and post-hoc WAV inspection —
and several distinct failure modes are indistinguishable through them: a dead or
mis-wired INMP441 still clocks data and looks exactly like a silent room, while a wrong
bit shift or slot configuration rails the samples in a way that only shows up when
someone opens the capture in an audio editor.

The board has an addressable RGB LED that nothing currently drives. Using it as a live
level-and-health indicator turns all of those failures into something visible at the
bench in under a second, which is what milestone M1 ("zero I2S drops over 1 h;
spectrogram of a known tone is correct") actually needs to be debuggable.

## What Changes

- New `debug_led` firmware module: a low-priority task that reads the existing PSRAM
  audio ring and drives the onboard addressable RGB LED (WS2812) at roughly 31 Hz.
- The LED indicates six visually distinct states: dead/no-signal, mic-alive-at-noise-
  floor, normal level, loud, approaching full scale, and clipping — plus a separate
  blink pattern for I2S stall and for rising DMA drop counts.
- Level is mapped on a **dB** scale (not linear amplitude), smoothed with VU-style
  ballistics (fast attack, ~300 ms release), and gamma-corrected for perceived
  brightness, with a minimum-on floor so "LED dark" reliably means "no signal" rather
  than "signal too quiet to render".
- New managed dependency: `espressif/led_strip` (RMT backend, one channel).
- Four new Kconfig values for bench calibration: LED GPIO, level floor and ceiling in
  dBFS, and maximum brightness.
- Runs in both the production and capture builds, unconditionally.
- No changes to `audio_capture.c`, the score loop, telemetry, or the model path. The
  module is a pure consumer of the existing public `audio_capture` API.

## Capabilities

### New Capabilities
- `node-status-indicator`: What the node communicates locally, on-device, about its own
  audio input and health — independent of network, hub, or host tooling. Owns the
  mapping from measured audio level and capture-path health to a visible indication,
  and the calibration surface that mapping needs.

### Modified Capabilities

(None. No existing spec's requirements change; the audio capture path, telemetry
payload, and detector behavior are untouched.)

## Impact

**Code** (all under `firmware/`):
- `main/debug_led.c`, `main/debug_led.h` — new, the entire feature
- `main/app_main.c` — one added call, above the capture-mode branch so both builds
  reach it
- `main/CMakeLists.txt` — new source entry and `espressif__led_strip` requirement
- `main/idf_component.yml`, `dependencies.lock` — new dependency
- `main/Kconfig.projbuild` — four calibration values

**Dependencies**: `espressif/led_strip` ^3.1 is pulled from the Espressif component
registry, so the first build after this change requires network access.

**Hardware**: drives GPIO38 (DevKitC-1 v1.1) or GPIO48 (v1.0) — selectable via Kconfig
because the pin differs by board revision and a wrong pin is indistinguishable from a
software bug. Consumes one RMT TX channel, which nothing else in the build uses.
GPIO35–37 are unavailable on this octal-PSRAM board and are not candidates.

**Accepted risks**:
- A production node carries a permanently lit LED, the new dependency, and an RMT
  channel in its image. This is deliberate; gating it to capture builds later costs
  two lines of Kconfig.
- WS2812 switching current shares the 3V3 rail with the microphone. The INMP441 is a
  digital-output part so the exposure is limited to supply ripple reaching its internal
  preamp; maximum brightness is capped to keep draw in single-digit milliamps, and the
  effect is verified by measurement rather than assumed.

**Out of scope**: reporting the LED state over MQTT, using the LED for detector or
alert state, external LED hardware, and any multi-LED bar display.
