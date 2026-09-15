#!/usr/bin/env bash
# Builds firmware/main/frontend_core.cc natively for the development machine, linked
# against the same signal/src/*.cc sources firmware compiles for the ESP32-S3 — see
# frontend_core.cc's header comment for why this is the anti-skew strategy rather than
# a from-scratch Python reimplementation. Output is loaded via ctypes by
# training/src/echo_sentry_train/features.py.
#
# Requires firmware/managed_components/espressif__esp-tflite-micro to already be
# fetched (run `idf.py build` once in firmware/, or `idf.py reconfigure`, to pull it
# via the ESP-IDF component manager) and firmware/main/gen/frontend_{config,tables}.h
# to already be generated (tools/gen_config.py).
#
# Usage: tools/build_native_frontend.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRMWARE_MAIN="$REPO_ROOT/firmware/main"
TFLM="$REPO_ROOT/firmware/managed_components/espressif__esp-tflite-micro"
SIGNAL_SRC="$TFLM/signal/src"
OUT="$REPO_ROOT/training/src/echo_sentry_train/libechosentry_frontend.dylib"

if [[ ! -d "$TFLM" ]]; then
  echo "error: $TFLM not found — run 'idf.py reconfigure' in firmware/ first" >&2
  exit 1
fi
if [[ ! -f "$FIRMWARE_MAIN/gen/frontend_config.h" ]]; then
  echo "error: firmware/main/gen/frontend_config.h missing — run tools/gen_config.py first" >&2
  exit 1
fi

SRCS=(
  "$FIRMWARE_MAIN/frontend_core.cc"
  "$SIGNAL_SRC"/*.cc
  "$SIGNAL_SRC"/kiss_fft_wrappers/*.cc
)

clang++ -std=c++17 -O2 -shared -fPIC \
  -I "$FIRMWARE_MAIN" \
  -I "$TFLM" \
  -I "$TFLM/third_party/kissfft" \
  -o "$OUT" \
  "${SRCS[@]}"

echo "wrote $OUT"
