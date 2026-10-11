#!/usr/bin/env python3
"""Serial receiver for node PCM over the native USB port — bring-up/bench companion to
tools/capture_server.py (comprehensive-plan.md §3.1's Wi-Fi field-campaign transport).
See ~/.claude/plans/create-a-plan-to-serialized-prism.md for why this transport exists.

Host side of firmware/main/capture_stream.c's framing:

    magic("ESCP", 4B) | type(u8) | seq(u16 BE) | sample_index(u32 BE) | len(u32 BE)
    | payload(len) | crc32(u32 BE, standard CRC-32 over type..payload)

Framing exists because the port also carries the device's log console — ESP_LOG lines
arrive as type=4 frames, but esp_rom_printf/ESP_EARLY_LOGx/the panic handler bypass the
firmware's log hook and write straight to the same wire, so unframed bytes can appear
between frames at any time. This receiver resyncs on a bad/incomplete frame rather than
losing the whole stream: on a CRC mismatch, a length that would read past a timeout, or
any read timeout, it discards what it has and rescans for the next magic.

Within a segment, PCM chunks carry a ring-relative `sample_index`; a gap between the
expected next sample and what actually arrives (a lost chunk, or the firmware resyncing
past a ring overrun — see capture_stream.c) is zero-filled so the WAV stays
sample-accurate, and counted into the sidecar's transport_drops_delta via the segment
header's transport_drops_cumulative field.

No SNTP on this transport (capture-mode app_main() never calls net_start()), so segment
start times are this script's *receive* time, not the node's — sidecars are stamped
"clock": "host". Compare against tools/capture_server.py's "clock": "sntp".

Usage:
    python3 tools/capture_serial.py --port /dev/cu.usbmodemXXXX --out-dir ./captures \\
        [--note "1kHz tone at 30cm"]

Start this BEFORE resetting/flashing the board. The handshake is sent once, early in
boot, with a bounded (10s) wait for a host to be attached; this is a live USB CDC
stream, not a replay log, so nothing is buffered for a late-attaching listener — miss
the handshake and every segment in that boot is silently dropped (this script warns
once if that happens).
"""

from __future__ import annotations

import argparse
import json
import os
import select
import struct
import sys
import termios
import time
import zlib
from pathlib import Path

import capture_sink

MAGIC = b"ESCP"
FRAME_META_LEN = 11  # type(1) + seq(2) + sample_index(4) + len(4)
FRAME_TYPE_HANDSHAKE = 1
FRAME_TYPE_SEGMENT_HEADER = 2
FRAME_TYPE_PCM_CHUNK = 3
FRAME_TYPE_LOG = 4

BODY_READ_TIMEOUT_S = 1.0  # for the rest of a frame once its header has been seen
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024  # sanity bound; real frames are a few KB at most
MAX_GAP_SAMPLES = 10 * capture_sink.SAMPLE_RATE_HZ * 60  # 10 minutes — beyond this,
# treat sample_index as a fresh start rather than zero-filling a huge, likely-bogus gap
SAMPLE_WIDTH_BYTES = capture_sink.SAMPLE_WIDTH_BYTES


def _open_raw(port: str) -> int:
    """Opens `port` in raw mode without pyserial — a USB CDC endpoint ignores the baud
    rate, so termios' B115200 here is a conventional placeholder, not a real line
    speed."""
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0  # iflag: no input processing
    attrs[1] = 0  # oflag: no output processing
    attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS8 | termios.CLOCAL | termios.CREAD
    attrs[3] = 0  # lflag: no echo, no canonical mode, no signal generation
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    cc = attrs[6]
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0  # non-blocking reads; timeouts are handled via select() below
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


class ByteReader:
    """Buffered, timeout-aware byte reader over a raw fd — the primitive the frame
    scanner and body reader both build on."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buf = bytearray()

    def _fill(self, deadline: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        r, _, _ = select.select([self._fd], [], [], remaining)
        if not r:
            return False
        chunk = os.read(self._fd, 4096)
        if chunk:
            self._buf += chunk
        return bool(chunk)

    def read_exact(self, n: int, timeout_s: float) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        while len(self._buf) < n:
            if not self._fill(deadline):
                return None
        data = bytes(self._buf[:n])
        del self._buf[:n]
        return data

    def find_magic(self, timeout_s: float) -> bool:
        """Scans forward, consuming everything up to and including one match of
        MAGIC. Returns False on timeout with no magic found."""
        deadline = time.monotonic() + timeout_s
        while True:
            idx = self._buf.find(MAGIC)
            if idx >= 0:
                del self._buf[: idx + len(MAGIC)]
                return True
            # Keep only enough trailing bytes for a magic to straddle the next read.
            if len(self._buf) > len(MAGIC) - 1:
                del self._buf[: len(self._buf) - (len(MAGIC) - 1)]
            if not self._fill(deadline):
                return False


def _read_frame(reader: ByteReader) -> tuple[int, int, bytes] | None:
    """Blocks waiting for the next valid frame (effectively indefinitely — Ctrl-C to
    stop). A malformed candidate is discarded and scanning resumes automatically; this
    only returns None if the port itself goes away with no magic ever found again
    within an hour."""
    while True:
        if not reader.find_magic(timeout_s=3600):
            return None
        meta = reader.read_exact(FRAME_META_LEN, timeout_s=BODY_READ_TIMEOUT_S)
        if meta is None:
            continue
        frame_type = meta[0]
        (sample_index,) = struct.unpack(">I", meta[3:7])
        (length,) = struct.unpack(">I", meta[7:11])
        if length > MAX_PAYLOAD_BYTES:
            continue
        rest = reader.read_exact(length + 4, timeout_s=BODY_READ_TIMEOUT_S)
        if rest is None:
            continue
        payload, crc_bytes = rest[:length], rest[length:]
        (expected_crc,) = struct.unpack(">I", crc_bytes)
        if zlib.crc32(meta + payload) != expected_crc:
            continue
        return frame_type, sample_index, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--out-dir", default="./captures")
    parser.add_argument("--note", default="", help="session_note recorded in every sidecar")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fd = _open_raw(args.port)
    reader = ByteReader(fd)
    print(f"capture_serial: listening on {args.port}, writing to {out_dir}")

    node_id: str | None = None
    fw = cfg = ""
    warned_no_handshake = False
    segment_index = 0
    seg_pcm: bytearray | None = None
    seg_header: dict | None = None
    expected_next_sample: int | None = None

    def finalize_segment() -> None:
        nonlocal segment_index, seg_pcm, seg_header
        if seg_pcm is not None and seg_header is not None and node_id is not None:
            capture_sink.write_segment(node_id, fw, cfg, seg_header, bytes(seg_pcm),
                                        out_dir, segment_index, clock="host")
            segment_index += 1
        seg_pcm = None
        seg_header = None

    try:
        while True:
            frame = _read_frame(reader)
            if frame is None:
                print("capture_serial: stream ended")
                break
            frame_type, sample_index, payload = frame

            if frame_type == FRAME_TYPE_LOG:
                sys.stderr.write(payload.decode(errors="replace") + "\n")
                continue

            if frame_type == FRAME_TYPE_HANDSHAKE:
                try:
                    hs = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                node_id = hs.get("node_id", "unknown")
                fw = hs.get("fw", "")
                cfg = hs.get("cfg", "")
                print(f"handshake: node={node_id} fw={fw} cfg={cfg}")
                continue

            if frame_type == FRAME_TYPE_SEGMENT_HEADER:
                finalize_segment()
                if node_id is None and not warned_no_handshake:
                    print("capture_serial: segment data arriving with no handshake "
                          "seen yet — this session's data cannot be attributed to a "
                          "node and will be dropped. Reset the board while this "
                          "script is running to pick up the handshake.", file=sys.stderr)
                    warned_no_handshake = True
                try:
                    header = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    header = {}
                header["start_utc"] = time.time()
                header["session_note"] = args.note
                seg_header = header
                seg_pcm = bytearray()
                expected_next_sample = None  # set by this segment's first PCM chunk
                continue

            if frame_type == FRAME_TYPE_PCM_CHUNK:
                if seg_pcm is None:
                    continue  # PCM before any segment_header — nothing to attach it to
                if expected_next_sample is not None and sample_index != expected_next_sample:
                    gap = sample_index - expected_next_sample
                    if 0 < gap <= MAX_GAP_SAMPLES:
                        seg_pcm += b"\x00\x00" * gap
                    # gap <= 0 (a re-delivery) or an implausibly large gap: don't
                    # zero-fill, just resume from where this chunk actually says it is.
                seg_pcm += payload
                expected_next_sample = sample_index + len(payload) // SAMPLE_WIDTH_BYTES
                continue
    except KeyboardInterrupt:
        print("\ncapture_serial: stopping")
    finally:
        finalize_segment()
        os.close(fd)


if __name__ == "__main__":
    main()
