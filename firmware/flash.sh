#!/usr/bin/env bash
# Provisions a node's identity into NVS, then builds and flashes it — comprehensive-
# plan.md §2.3. Run from firmware/, with the ESP32-S3 connected over its native USB
# port (comprehensive-plan.md Part 2c) and `get_idf` sourced in this shell.
#
# Usage:
#   ./flash.sh es-01 MyWifiSSID MyWifiPassword [hub_host] [port]
#
# Provisioning writes directly to an NVS partition image and flashes it to the `nvs`
# partition (see partitions.csv) via idf.py's NVS partition-generation tooling —
# nothing here talks to the device except esptool. Re-run with a different node_id to
# repurpose a board; re-run with the same node_id to update Wi-Fi creds without
# touching the accumulated threshold value if you skip that argument (not currently
# supported by this script — it always writes the compiled-in default; use the
# "set_threshold" MQTT command from the hub for a live retune instead).

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <node_id> <wifi_ssid> <wifi_password> [hub_host] [mqtt_port]" >&2
    exit 1
fi

NODE_ID="$1"
WIFI_SSID="$2"
WIFI_PASSWORD="$3"
HUB_HOST="${4:-10.42.0.1}"
MQTT_PORT="${5:-1883}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v idf.py >/dev/null 2>&1; then
    echo "error: idf.py not found — run 'get_idf' first" >&2
    exit 1
fi

CSV="build/nvs_provision.csv"
mkdir -p build
cat > "$CSV" <<EOF
key,type,encoding,value
echo_sentry,namespace,,
node_id,data,string,${NODE_ID}
wifi_ssid,data,string,${WIFI_SSID}
wifi_pass,data,string,${WIFI_PASSWORD}
hub_host,data,string,${HUB_HOST}
mqtt_port,data,u16,${MQTT_PORT}
EOF

# NVS partition size must match partitions.csv's "nvs" row (0x6000).
python "$IDF_PATH/components/nvs_flash/nvs_partition_generator/nvs_partition_gen.py" \
    generate "$CSV" build/nvs_provision.bin 0x6000

echo "Building..."
idf.py build

echo "Flashing app + bootloader + partition table..."
idf.py flash

echo "Flashing NVS provisioning data for node '${NODE_ID}'..."
python -m esptool --chip esp32s3 write_flash 0x9000 build/nvs_provision.bin

echo "Done. Reset the board or run: idf.py monitor"
