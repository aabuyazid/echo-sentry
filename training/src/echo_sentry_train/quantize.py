"""Full-integer post-training quantization — comprehensive-plan.md §4.4.

`inference_input_type`/`inference_output_type` are both int8 and ops are constrained
to TFLITE_BUILTINS_INT8 — anything that would fall back to float doesn't run under
TFLM. The representative dataset is drawn from the training split, sampled across the
full range of normal operating states rather than one contiguous hour.

Usage:
    uv run --project training python -m echo_sentry_train.quantize \\
        --manifest manifest.csv --model-dir models/es-01 --out models/es-01/model.tflite
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tensorflow as tf

from echo_sentry_train import frontend_config as fc
from echo_sentry_train.dataset import Normalization, load_split
from echo_sentry_train.model import split_context_target

REPRESENTATIVE_SAMPLE_SIZE = 500


def _representative_input(manifest_path: Path, node: str, norm: Normalization,
                           arch: str) -> np.ndarray:
    x = load_split(manifest_path, node, "train")
    if len(x) == 0:
        raise ValueError(f"no training vectors for node {node!r}")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=min(REPRESENTATIVE_SAMPLE_SIZE, len(x)), replace=False)
    x_n = norm.apply(x[idx])

    if arch == "idnn":
        context, _target = split_context_target(x_n)
        return context
    if arch == "conv_ae":
        return x_n.reshape(-1, fc.FRAMES_PER_VECTOR, fc.MEL_CHANNELS, 1)
    return x_n  # dense_ae


def _representative_dataset_fn(rep_input: np.ndarray):
    def gen() -> Iterator[list[np.ndarray]]:
        for row in rep_input:
            yield [row[np.newaxis, ...].astype(np.float32)]

    return gen


def quantize(model_dir: Path, manifest_path: Path, out_path: Path) -> dict:
    metadata = json.loads((model_dir / "train_metadata.json").read_text())
    node, arch = metadata["node"], metadata["arch"]
    norm = Normalization.from_dict(json.loads((model_dir / "normalization.json").read_text()))
    keras_model = tf.keras.models.load_model(model_dir / "model.keras")

    rep_input = _representative_input(manifest_path, node, norm, arch)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_fn(rep_input)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    return {
        "node": node, "arch": arch, "tflite_path": str(out_path),
        "size_bytes": len(tflite_model), "config_hash": fc.CONFIG_HASH,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = quantize(args.model_dir, args.manifest, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
