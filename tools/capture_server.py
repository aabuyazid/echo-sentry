#!/usr/bin/env python3
"""TCP receiver for node PCM during the data-collection campaign — comprehensive-plan.md
§3.1. Stdlib-only so it runs identically under either project's venv (or plain python3).

Wire protocol (implemented device-side by firmware/main/capture_stream.c when
CONFIG_ECHO_SENTRY_CAPTURE_MODE is enabled with the TCP transport —
comprehensive-plan.md §3.1; the TCP transport itself is not yet implemented, see that
file — this is the target protocol once it is). tools/capture_serial.py speaks the same
handshake/header/PCM fields over the SERIAL transport, wrapped in an additional framing
envelope because that pipe is shared with the log console.

  1. Node opens one persistent TCP connection for the whole capture session.
  2. Handshake, once per connection:
       uint32 BE length N, then N bytes of UTF-8 JSON: {"node_id", "fw", "cfg"}
  3. Then a sequence of ~60s segments, each:
       uint32 BE length M, then M bytes of UTF-8 JSON segment header:
         {"start_utc": <float, SNTP-synced>, "i2s_drops_cumulative": <int>,
          "session_note": <str>}
       uint32 BE length P (PCM byte length)
       P bytes of raw little-endian int16 mono PCM at SAMPLE_RATE_HZ
  4. Connection closes when the capture session ends.

Each segment is written (via capture_sink.write_segment — shared with
tools/capture_serial.py so output is byte-identical regardless of transport) as one WAV
file plus a sidecar JSON carrying everything training/src/echo_sentry_train/clean.py's
manifest step needs — critically the per-segment I2S drop *delta*, which is the
integrity signal a segment gets quarantined on (comprehensive-plan.md §3.1, §3.3).

Usage:
    python3 tools/capture_server.py --out-dir ./captures --port 9000
"""

from __future__ import annotations

import argparse
import json
import socketserver
import struct
from pathlib import Path

import capture_sink

_LEN_STRUCT = struct.Struct(">I")


def _recv_exact(conn, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"connection closed with {n - len(buf)} bytes still expected")
        buf += chunk
    return bytes(buf)


def _recv_framed_json(conn) -> dict:
    (length,) = _LEN_STRUCT.unpack(_recv_exact(conn, 4))
    return json.loads(_recv_exact(conn, length).decode("utf-8"))


def _recv_framed_bytes(conn) -> bytes:
    (length,) = _LEN_STRUCT.unpack(_recv_exact(conn, 4))
    return _recv_exact(conn, length)


class CaptureHandler(socketserver.BaseRequestHandler):
    out_dir: Path  # set by main() before serve_forever()

    def handle(self) -> None:
        conn = self.request
        try:
            handshake = _recv_framed_json(conn)
        except (ConnectionError, json.JSONDecodeError, struct.error) as exc:
            print(f"[{self.client_address}] handshake failed: {exc}")
            return

        node_id = handshake.get("node_id", "unknown")
        fw = handshake.get("fw", "")
        cfg = handshake.get("cfg", "")
        print(f"[{self.client_address}] capture session started: node={node_id} fw={fw} cfg={cfg}")

        node_dir = self.out_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)

        segment_count = 0
        try:
            while True:
                header = _recv_framed_json(conn)
                pcm = _recv_framed_bytes(conn)
                self._write_segment(node_id, fw, cfg, header, pcm, node_dir, segment_count)
                segment_count += 1
        except ConnectionError:
            pass  # normal end-of-session
        except (json.JSONDecodeError, struct.error) as exc:
            print(f"[{self.client_address}] protocol error mid-session: {exc}")
        finally:
            print(f"[{self.client_address}] session ended: node={node_id}, "
                  f"{segment_count} segment(s) captured")

    def _write_segment(self, node_id: str, fw: str, cfg: str, header: dict, pcm: bytes,
                        node_dir: Path, segment_index: int) -> None:
        # "sntp": by the time a node is on this transport it has joined the Pi's AP
        # and synced against chrony (net.c) — start_utc in `header` is trustworthy,
        # unlike tools/capture_serial.py's bench path.
        capture_sink.write_segment(node_id, fw, cfg, header, pcm, node_dir.parent,
                                    segment_index, clock="sntp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - node-network-only by design
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--out-dir", default="./captures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    CaptureHandler.out_dir = out_dir

    print(f"capture_server: listening on {args.host}:{args.port}, writing to {out_dir}")
    with socketserver.ThreadingTCPServer((args.host, args.port), CaptureHandler) as server:
        server.daemon_threads = True
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\ncapture_server: stopping")


if __name__ == "__main__":
    main()
