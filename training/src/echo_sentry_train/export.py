"""Final export: model.tflite + model.json sidecar — comprehensive-plan.md §2.2(c).

Firmware refuses to run a model whose config hash doesn't match its own; model.json is
what carries that hash (and everything else worth knowing about a model at a glance)
alongside the .tflite firmware embeds via EMBED_FILES in main/CMakeLists.txt.

Usage:
    uv run --project training python -m echo_sentry_train.export \\
        --model-dir models/es-01 --tflite models/es-01/model.tflite \\
        [--fp32-int8-delta 0.004]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import UTC, datetime
from pathlib import Path

import tensorflow as tf

from echo_sentry_train import frontend_config as fc
from echo_sentry_train.dataset import Normalization


def _quantization_params(tflite_path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = in_detail["quantization"]
    out_scale, out_zero = out_detail["quantization"]
    return {
        "input_dtype": str(in_detail["dtype"].__name__), "input_scale": float(in_scale),
        "input_zero_point": int(in_zero), "output_dtype": str(out_detail["dtype"].__name__),
        "output_scale": float(out_scale), "output_zero_point": int(out_zero),
    }


def export_normalization_bin(model_dir: Path) -> Path:
    """Raw float32[VECTOR_DIM] mean followed by float32[VECTOR_DIM] std —
    firmware/main/inference.cc reads this directly (no JSON parsing on-device) to
    reproduce dataset.py's Normalization.apply() before quantizing to int8.
    """
    norm = Normalization.from_dict(json.loads((model_dir / "normalization.json").read_text()))
    if len(norm.mean) != fc.VECTOR_DIM or len(norm.std) != fc.VECTOR_DIM:
        raise ValueError(
            f"normalization.json has {len(norm.mean)} dims, expected VECTOR_DIM="
            f"{fc.VECTOR_DIM} — regenerate it (train.py) against the current config"
        )
    out_path = model_dir / "normalization.bin"
    with out_path.open("wb") as f:
        f.write(struct.pack(f"<{fc.VECTOR_DIM}f", *norm.mean.tolist()))
        f.write(struct.pack(f"<{fc.VECTOR_DIM}f", *norm.std.tolist()))
    return out_path


def export_model_json(model_dir: Path, tflite_path: Path, *,
                       fp32_int8_auc_delta: float | None = None) -> dict:
    train_metadata = json.loads((model_dir / "train_metadata.json").read_text())
    baseline = None
    baseline_path = model_dir / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())

    tflite_bytes = tflite_path.read_bytes()
    sha256_12 = hashlib.sha256(tflite_bytes).hexdigest()[:12]
    iso = datetime.now(tz=UTC).isocalendar()
    model_id = f"{train_metadata['arch']}-{iso.year}w{iso.week:02d}-{sha256_12[:6]}"

    manifest = {
        "model_id": model_id,
        "node": train_metadata["node"],
        "arch": train_metadata["arch"],
        "config_hash": fc.CONFIG_HASH,
        "vector_dim": fc.VECTOR_DIM,
        "size_bytes": len(tflite_bytes),
        "sha256_12": sha256_12,
        "quantization": _quantization_params(tflite_path),
        "train_vectors": train_metadata.get("train_vectors"),
        "val_vectors": train_metadata.get("val_vectors"),
        "seed": train_metadata.get("seed"),
        "baseline": baseline,
        "fp32_int8_auc_delta": fp32_int8_auc_delta,
        "exported_at_utc": datetime.now(tz=UTC).isoformat(),
    }

    (model_dir / "model.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tflite", required=True, type=Path)
    parser.add_argument("--fp32-int8-delta", type=float, default=None)
    args = parser.parse_args()

    manifest = export_model_json(args.model_dir, args.tflite,
                                  fp32_int8_auc_delta=args.fp32_int8_delta)
    norm_bin_path = export_normalization_bin(args.model_dir)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {norm_bin_path}")


if __name__ == "__main__":
    main()
