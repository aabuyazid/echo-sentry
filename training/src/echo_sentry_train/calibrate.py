"""Derives per-node baseline calibration (median/MAD of the trained model's score
distribution on normal data) — comprehensive-plan.md §2.5 and §5 Stage 1.

This computes the SAME statistic hub/detector.py eventually computes live from real
traffic. Running it here lets you pre-seed the hub's `detector_state` table so the
first ~100 live scores (MIN_BASELINE_SAMPLES) aren't stuck bootstrapping against an
empty baseline — see hub/src/echo_sentry_hub/store.py's DetectorState.

Usage:
    uv run --project training python -m echo_sentry_train.calibrate \\
        --manifest manifest.csv --node es-01 --model-dir models/es-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from echo_sentry_train.dataset import Normalization, load_split
from echo_sentry_train.model import load_trained_scorer


def compute_baseline(scores: np.ndarray) -> dict:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    return {"baseline_median": median, "baseline_mad": mad, "n_samples": int(len(scores))}


def calibrate_node(manifest_path: Path, node: str, model_dir: Path) -> dict:
    norm = Normalization.from_dict(json.loads((model_dir / "normalization.json").read_text()))
    scorer = load_trained_scorer(model_dir)

    x_cal = load_split(manifest_path, node, "val")
    if len(x_cal) == 0:
        x_cal = load_split(manifest_path, node, "train")
    if len(x_cal) == 0:
        raise ValueError(f"no data available to calibrate node {node!r}")

    scores = scorer.score(norm.apply(x_cal))
    result = compute_baseline(scores)
    result["node"] = node
    (model_dir / "baseline.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()

    result = calibrate_node(args.manifest, args.node, args.model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
