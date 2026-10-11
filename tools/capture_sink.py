"""Shared WAV + sidecar JSON writer for capture transports — comprehensive-plan.md §3.1.

Both tools/capture_server.py (TCP, field campaign) and tools/capture_serial.py (USB
serial, bench bring-up — ~/.claude/plans/create-a-plan-to-serialized-prism.md) call
write_segment() so a captured segment's on-disk format is byte-identical regardless of
transport; training/src/echo_sentry_train/clean.py's manifest step sees one format.

Stdlib-only, same rationale as capture_server.py: this must run under whatever
`python3` a field laptop resolves to.
"""

from __future__ import annotations

import json
import wave
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_RATE_HZ = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1

# Tracks each node's last-seen *cumulative* drop counters across segments/connections,
# so the sidecar JSON can record per-segment *deltas* rather than running totals. Two
# counters: i2s (audio_capture.c's DMA/ring drops — present on every transport) and
# transport (this transport's own losses: chunk send timeouts, ring-overrun resyncs;
# only the SERIAL transport reports this today — see firmware/main/capture_stream.c).
_last_i2s_cumulative: dict[str, int] = {}
_last_transport_cumulative: dict[str, int] = {}


def write_segment(node_id: str, fw: str, cfg: str, header: dict, pcm: bytes,
                   out_dir: Path, segment_index: int, *, clock: str) -> Path:
    """Writes one segment's WAV + sidecar JSON under out_dir/node_id/. Returns the WAV
    path.

    `clock` is "sntp" for a transport whose start_utc is trustworthy (TCP, once the
    node has synced against the Pi's chrony — comprehensive-plan.md §2.1) or "host"
    when the receiver stamped receive time itself because the node has no SNTP on this
    path (SERIAL — capture-mode app_main() never calls net_start()).
    """
    node_dir = out_dir / node_id
    node_dir.mkdir(parents=True, exist_ok=True)

    start_utc = float(header.get("start_utc", 0.0))
    i2s_cumulative = int(header.get("i2s_drops_cumulative", 0))
    transport_cumulative = int(header.get("transport_drops_cumulative", 0))
    session_note = header.get("session_note", "")

    i2s_delta = max(0, i2s_cumulative - _last_i2s_cumulative.get(node_id, i2s_cumulative))
    _last_i2s_cumulative[node_id] = i2s_cumulative
    transport_delta = max(
        0, transport_cumulative - _last_transport_cumulative.get(node_id, transport_cumulative)
    )
    _last_transport_cumulative[node_id] = transport_cumulative

    # segment_index disambiguates same-second segments (1s timestamp resolution); real
    # ~60s segment cadence makes this rare, but zero-length or back-to-back test/replay
    # segments hit it easily — a silent overwrite is worse than a slightly uglier name.
    # timezone.utc (not the newer datetime.UTC alias) deliberately — see the note above
    # about running under whatever python3 a field laptop resolves to.
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
        "duration_s": duration_s,
        "i2s_drops_delta": i2s_delta, "i2s_drops_cumulative": i2s_cumulative,
        "transport_drops_delta": transport_delta,
        "transport_drops_cumulative": transport_cumulative,
        "session_note": session_note, "clock": clock,
    }
    json_path.write_text(json.dumps(sidecar, indent=2))

    flags = []
    if i2s_delta:
        flags.append("I2S DROPS")
    if transport_delta:
        flags.append("TRANSPORT DROPS")
    flag = f" ⚠️ {', '.join(flags)}" if flags else ""
    print(f"  segment: {wav_path.name} ({duration_s:.1f}s){flag}")

    return wav_path
