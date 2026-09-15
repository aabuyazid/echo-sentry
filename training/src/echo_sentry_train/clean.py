"""Cleaning pipeline — comprehensive-plan.md §3.3.

Turns raw captured WAV + sidecar JSON (as written by tools/capture_server.py) into a
manifest CSV. The manifest is the dataset; the WAVs are just storage — dataset.py reads
only the manifest plus the WAV paths it names.

Usage:
    uv run --project training python -m echo_sentry_train.clean \\
        --captures-dir ./captures --logbook ./logbook.csv --out ./manifest.csv \\
        --domain-shift-date 2026-09-27 --domain-shift-date 2026-09-28
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import stats as scipy_stats

CLIP_FRACTION_THRESHOLD = 0.0001  # >0.01% of samples at/above CLIP_LEVEL -> reject
CLIP_LEVEL = 0.99  # fraction of full scale
MACHINE_OFF_RMS_DBFS_DEFAULT = -60.0
IMPULSE_CREST_FACTOR_THRESHOLD = 8.0  # peak/RMS, linear — well above steady machine noise
IMPULSE_KURTOSIS_THRESHOLD = 8.0  # excess kurtosis; steady noise sits near 0

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2
# remaining ~0.2 -> test


@dataclass
class ManifestRow:
    path: str
    node: str
    start_utc: float
    duration_s: float
    machine_state: str  # "normal" | "off" | "unknown"
    reject_reason: str | None  # None | "integrity" | "clipping" | "logbook"
    split: str | None  # None | "train" | "val" | "test" | "domain_shift" | "nuisance"


def load_sidecar(json_path: Path) -> dict:
    return json.loads(json_path.read_text())


def read_pcm_float(wav_path: Path) -> np.ndarray:
    data, _sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    return data


def check_integrity(sidecar: dict) -> str | None:
    """Step 1 — non-zero I2S drop delta means the segment contains at least one buffer
    overrun; a genuine acoustic discontinuity can't be distinguished from a glitch."""
    if sidecar.get("i2s_drops_delta", 0) > 0:
        return "integrity"
    return None


def check_clipping(pcm: np.ndarray) -> str | None:
    """Step 2 — clipping is unrecoverable and silently destroys high-frequency content."""
    if pcm.size == 0:
        return "integrity"
    fraction = float(np.mean(np.abs(pcm) >= CLIP_LEVEL))
    if fraction > CLIP_FRACTION_THRESHOLD:
        return "clipping"
    return None


def classify_machine_state(pcm: np.ndarray, *, off_rms_dbfs: float) -> str:
    """Step 4 — labeled, not dropped: 'off' is a legitimate state the detector must
    learn to suppress rather than flag as anomalous silence."""
    if pcm.size == 0:
        return "unknown"
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    rms_dbfs = 20 * np.log10(max(rms, 1e-9))
    return "off" if rms_dbfs < off_rms_dbfs else "normal"


def check_impulse(pcm: np.ndarray) -> bool:
    """Step 5 — True if this segment looks impulsive (a knock, dropped tool, forklift)
    rather than steady machine noise. Impulsive segments are excluded from AE training
    but retained in the `nuisance` split for tuning the hub's rejection logic — never
    simply discarded (comprehensive-plan.md §3.3)."""
    if pcm.size == 0:
        return False
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(pcm)))
    crest_factor = peak / max(rms, 1e-9)
    kurtosis = float(scipy_stats.kurtosis(pcm, fisher=True))
    return crest_factor > IMPULSE_CREST_FACTOR_THRESHOLD or kurtosis > IMPULSE_KURTOSIS_THRESHOLD


def load_logbook(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def logbook_overlaps(entry_ts: float, logbook: list[dict], *, window_s: float = 60.0) -> bool:
    """Step 6 — a logbook entry is assumed to cover +/- window_s around its own
    timestamp; widen window_s at the call site for a known longer event (e.g. a whole
    maintenance shift)."""
    for row in logbook:
        try:
            ts = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError):
            continue
        if abs(entry_ts - ts) <= window_s:
            return True
    return False


def build_manifest(captures_dir: Path, logbook_path: Path | None, *,
                    off_rms_dbfs: float = MACHINE_OFF_RMS_DBFS_DEFAULT,
                    domain_shift_dates: set[str] | None = None) -> list[ManifestRow]:
    logbook = load_logbook(logbook_path)
    domain_shift_dates = domain_shift_dates or set()

    rows: list[ManifestRow] = []
    for json_path in sorted(captures_dir.rglob("*.json")):
        wav_path = json_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        sidecar = load_sidecar(json_path)
        node = sidecar.get("node_id", json_path.parent.name)
        start_utc = float(sidecar.get("start_utc", 0.0))
        duration_s = float(sidecar.get("duration_s", 0.0))

        reject = check_integrity(sidecar)
        machine_state = "unknown"
        is_impulse = False

        if reject is None:
            pcm = read_pcm_float(wav_path)
            reject = check_clipping(pcm)
            machine_state = classify_machine_state(pcm, off_rms_dbfs=off_rms_dbfs)
            is_impulse = check_impulse(pcm)

        if reject is None and logbook_overlaps(start_utc, logbook):
            reject = "logbook"

        split: str | None = None
        if reject is None:
            if is_impulse:
                split = "nuisance"
            else:
                date_str = datetime.fromtimestamp(start_utc, tz=UTC).strftime("%Y-%m-%d")
                if date_str in domain_shift_dates:
                    split = "domain_shift"
                # else: assigned chronologically below, per node

        rows.append(ManifestRow(
            path=str(wav_path), node=node, start_utc=start_utc, duration_s=duration_s,
            machine_state=machine_state, reject_reason=reject, split=split,
        ))

    _assign_time_splits(rows)
    return rows


def _assign_time_splits(rows: list[ManifestRow]) -> None:
    """Step 7 — chronological 60/20/20 train/val/test split PER NODE, never random.
    Only touches rows not already rejected, marked nuisance, or a domain-shift
    holdout (step 8), and only "normal" machine-state segments — "off" segments
    aren't part of the autoencoder's training distribution.

    Random splitting is the most common serious error in this problem domain:
    adjacent frames in a continuous recording are near-identical, so a random split
    leaks the test set into training and produces a model that looks excellent and
    generalizes to nothing.
    """
    by_node: dict[str, list[ManifestRow]] = {}
    for row in rows:
        if row.reject_reason is not None or row.split is not None:
            continue
        if row.machine_state != "normal":
            continue
        by_node.setdefault(row.node, []).append(row)

    for node_rows in by_node.values():
        node_rows.sort(key=lambda r: r.start_utc)
        n = len(node_rows)
        train_end = int(n * TRAIN_FRACTION)
        val_end = train_end + int(n * VAL_FRACTION)
        for i, row in enumerate(node_rows):
            row.split = "train" if i < train_end else "val" if i < val_end else "test"


def write_manifest(rows: list[ManifestRow], out_path: Path) -> None:
    fieldnames = [f.name for f in fields(ManifestRow)]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            rows.append(ManifestRow(
                path=raw["path"], node=raw["node"], start_utc=float(raw["start_utc"]),
                duration_s=float(raw["duration_s"]), machine_state=raw["machine_state"],
                reject_reason=raw["reject_reason"] or None, split=raw["split"] or None,
            ))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures-dir", required=True, type=Path)
    parser.add_argument("--logbook", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--off-rms-dbfs", type=float, default=MACHINE_OFF_RMS_DBFS_DEFAULT)
    parser.add_argument("--domain-shift-date", action="append", default=[],
                         help="YYYY-MM-DD to reserve as a domain-shift holdout (repeatable)")
    args = parser.parse_args()

    rows = build_manifest(args.captures_dir, args.logbook, off_rms_dbfs=args.off_rms_dbfs,
                           domain_shift_dates=set(args.domain_shift_date))
    write_manifest(rows, args.out)

    kept = sum(1 for r in rows if r.reject_reason is None)
    print(f"{len(rows)} segments scanned, {kept} kept, {len(rows) - kept} rejected")
    for reason in ("integrity", "clipping", "logbook"):
        n = sum(1 for r in rows if r.reject_reason == reason)
        if n:
            print(f"  rejected ({reason}): {n}")
    for split in ("train", "val", "test", "domain_shift", "nuisance"):
        n = sum(1 for r in rows if r.split == split)
        print(f"  split={split}: {n}")


if __name__ == "__main__":
    main()
