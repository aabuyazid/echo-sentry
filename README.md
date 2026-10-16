# Echo Sentry

Five ESP32-S3 + INMP441 acoustic edge nodes run on-device anomaly detection (log-mel features →
quantized TFLite Micro autoencoder) and publish only anomaly scores + metadata over MQTT to a
Raspberry Pi hub. The hub deduplicates, persists to SQLite, and forwards alerts to Microsoft Teams
and Power BI. No raw audio ever leaves the equipment room.

Full environment setup, architecture rationale, and verification steps: see
[`plan.md`](../plan.md) (also at `~/.claude/plans/i-need-to-set-temporal-bee.md`).

## Layout

| Directory | Toolchain | Produces |
|---|---|---|
| `firmware/` | ESP-IDF v5.5.5 (Xtensa/CMake) | node firmware image |
| `training/` | Python 3.12 + TensorFlow | quantized `.tflite` model |
| `hub/` | Python 3.13 | Raspberry Pi systemd service |
| `config/frontend.yaml` | — | single source of truth for audio feature params (see `tools/gen_config.py`) |

## Wiring

INMP441 breakout to an ESP32-S3-DevKitC-1. Pins are the `firmware/main/Kconfig.projbuild`
defaults; change them there, not in `audio_capture.c`.

| Mic pin | Board pin | Kconfig |
|---|---|---|
| VDD | **3V3** | — (1.8–3.3 V part; 5 V destroys it) |
| GND | GND | — |
| SCK | GPIO5 | `ECHO_SENTRY_I2S_BCK_GPIO` |
| WS | GPIO6 | `ECHO_SENTRY_I2S_WS_GPIO` |
| SD | GPIO7 | `ECHO_SENTRY_I2S_DIN_GPIO` |
| L/R | **GND** | — (required: `audio_capture.c` reads the left slot only) |

GPIO 5/6/7 avoid the S3's strapping pins (0, 3, 45, 46) and its octal-PSRAM pins (33–37).
The debug indicator needs no wiring — it is the onboard WS2812, GPIO48 on a DevKitC-1 v1.0
(the default) and GPIO38 on a v1.1 (`ECHO_SENTRY_DEBUG_LED_GPIO`); check the silkscreen,
because a wrong pin there is indistinguishable from a dead task. The separate red LED is a
hardwired power indicator with no GPIO behind it.

Bring-up order that isolates faults fastest: power and L/R first, confirm the LED lights,
then add SCK/WS/SD. A mic with data connected but L/R left floating reads as a dead-quiet
room — the one failure the indicator cannot tell apart from a genuinely quiet one.

## Quickstart

```bash
uv python install 3.12 3.13   # once
get_idf                        # alias for `. ~/esp/esp-idf/export.sh`

cd training && uv sync && uv run pytest
cd ../hub      && uv sync && uv run pytest

cd ../firmware
idf.py set-target esp32s3
idf.py build
```

Status: all three toolchains build, and the node firmware flashes and boots on an
ESP32-S3 with the TFLM interpreter live. Not yet exercised on hardware: provisioning
(`firmware/flash.sh`), the Wi-Fi/MQTT path to the hub, and the 14-day data-collection
campaign. The embedded `model.tflite` is a placeholder trained on synthetic audio, to be
replaced per node once that campaign runs — see `firmware/main/models/README.md`.
