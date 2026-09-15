#!/usr/bin/env python3
"""TCP receiver for node PCM during the data-collection campaign — comprehensive-plan.md
§3.1. Stdlib-only so it runs identically under either project's venv (or plain python3).

Wire protocol (implemented by firmware/main/evidence.c's capture-mode sibling, when
CONFIG_ECHO_SENTRY_CAPTURE_MODE is enabled — comprehensive-plan.md §3.1):

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

Each segment is written as one WAV file plus a sidecar JSON carrying everything
training/src/echo_sentry_train/clean.py's manifest step needs — critically the
per-segment I2S drop *delta*, which is the integrity signal a segment gets quarantined
on (comprehensive-plan.md §3.1, §3.3).

Usage:
    python3 tools/capture_server.py --out-dir ./captures --port 9000
"""

from __future__ import annotations

import argparse
import json
import socketserver
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_RATE_HZ = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
_LEN_STRUCT = struct.Struct(">I")

# Tracks each node's last-seen cumulative I2S drop count across segments/connections,
# so the sidecar JSON can record a per-segment *delta* rather than a running total.
_last_cumulative_drops: dict[str, int] = {}


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
        start_utc = float(header.get("start_utc", 0.0))
        cumulative_drops = int(header.get("i2s_drops_cumulative", 0))
        session_note = header.get("session_note", "")

        previous = _last_cumulative_drops.get(node_id, cumulative_drops)
        drop_delta = max(0, cumulative_drops - previous)
        _last_cumulative_drops[node_id] = cumulative_drops

        # segment_index disambiguates same-second segments (1s timestamp resolution);
        # real ~60s segment cadence makes this rare, but zero-length or back-to-back
        # test/replay segments hit it easily — a silent overwrite is worse than a
        # slightly uglier filename.
        # timezone.utc (not the newer datetime.UTC alias) deliberately — this file is
        # stdlib-only so it runs under whatever `python3` a field laptop resolves to,
        # which has meant Apple's system 3.9 in practice (no `datetime.UTC` before 3.11).
        stamp = datetime.fromtimestamp(start_utc, tz=timezone.utc).strftime(  # noqa: UP017
            "%Y%m%dT%H%M%SZ"
        )
        base = node_dir / f"{stamp}_{segment_index:05d}"
        wav_path = base.with_suffix(".wav")
        json_path = base.with_suffix(".json")

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(SAMPLE_RATE_HZ)
            wf.writeframes(pcm)

        num_samples = len(pcm) // SAMPLE_WIDTH_BYTES
        duration_s = num_samples / SAMPLE_RATE_HZ
        sidecar = {
            "node_id": node_id, "fw": fw, "cfg": cfg,
            "start_utc": start_utc, "end_utc": start_utc + duration_s,
            "duration_s": duration_s, "i2s_drops_delta": drop_delta,
            "i2s_drops_cumulative": cumulative_drops, "session_note": session_note,
        }
        json_path.write_text(json.dumps(sidecar, indent=2))

        flag = " ⚠️ DROPS" if drop_delta else ""
        print(f"  segment: {wav_path.name} ({duration_s:.1f}s, drops={drop_delta}){flag}")


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
