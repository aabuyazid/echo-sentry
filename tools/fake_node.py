#!/usr/bin/env python3
"""Simulates N ESP32 nodes publishing scores over MQTT — comprehensive-plan.md §2.4 / §7 (M6).

Lets the whole hub (ingest -> detector -> alerting -> forwarder -> sinks) be developed
and tested end-to-end without any hardware. Publishes a stationary baseline with
configurable per-node drift, spikes, and dropout.

Usage (run mosquitto locally first: `brew services start mosquitto`):
    uv run --project hub python tools/fake_node.py
    uv run --project hub python tools/fake_node.py --spike-node es-01 --spike-after-s 60
    uv run --project hub python tools/fake_node.py --all-spike-after-s 60   # common-mode test
    uv run --project hub python tools/fake_node.py --dropout-node es-03 --dropout-after-s 90
"""

from __future__ import annotations

import argparse
import json
import random
import time

import paho.mqtt.client as mqtt

TOPIC_PREFIX = "echo-sentry/v1"


def build_score(node: str, seq: int, err_mean: float) -> dict:
    now = int(time.time())
    return {
        "v": 1, "node": node, "t": now, "seq": seq,
        "err_mean": round(err_mean, 4), "err_p95": round(err_mean * 1.4, 4),
        "err_max": round(err_mean * 2.0, 4), "rms_dbfs": round(-40.0 + random.uniform(-2, 2), 2),
        "bands": [0] * 16, "drops": 0,
        "model": "fake-node", "cfg": "fake", "fw": "0.0.0-sim",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--nodes", type=int, default=5)
    p.add_argument("--interval", type=float, default=10.0, help="seconds between scores")
    p.add_argument("--baseline", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--spike-node", action="append", default=[],
                    help="node id to spike (repeatable)")
    p.add_argument("--spike-magnitude", type=float, default=5.0)
    p.add_argument("--spike-after-s", type=float, default=60.0,
                    help="start spiking this many seconds after launch")
    p.add_argument("--spike-duration-s", type=float, default=0.0,
                    help="0 = spike continues indefinitely once triggered")
    p.add_argument("--dropout-node", action="append", default=[],
                    help="node id to stop publishing after --dropout-after-s (repeatable)")
    p.add_argument("--dropout-after-s", type=float, default=90.0)
    p.add_argument("--all-spike-after-s", type=float, default=None,
                    help="spike every node at once — for exercising common-mode rejection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nodes = [f"es-0{i}" for i in range(1, args.nodes + 1)]

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    seq = dict.fromkeys(nodes, 0)
    start = time.monotonic()
    print(f"fake_node: publishing {len(nodes)} nodes to {args.host}:{args.port} "
          f"every {args.interval}s (ctrl-C to stop)")
    try:
        while True:
            elapsed = time.monotonic() - start
            for node in nodes:
                if node in args.dropout_node and elapsed >= args.dropout_after_s:
                    continue

                spiking = (
                    (node in args.spike_node and elapsed >= args.spike_after_s
                     and (args.spike_duration_s == 0.0
                          or elapsed < args.spike_after_s + args.spike_duration_s))
                    or (args.all_spike_after_s is not None and elapsed >= args.all_spike_after_s)
                )
                err_mean = args.baseline + (args.spike_magnitude if spiking else 0.0)
                err_mean += random.uniform(-args.noise, args.noise)

                payload = build_score(node, seq[node], max(err_mean, 0.0))
                client.publish(f"{TOPIC_PREFIX}/{node}/score", json.dumps(payload), qos=1)
                seq[node] += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nfake_node: stopping")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
