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

Status: environment scaffolding only — no application code yet.
