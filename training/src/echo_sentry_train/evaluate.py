"""Evaluation — comprehensive-plan.md §4.5.

Tier A (public benchmark correctness against MIMII/DCASE) and the fp32->int8
quantization delta gate are both driven from here, by pointing --normal-dir /
--anomaly-dir at a downloaded corpus. Tier B (synthetic SNR injection) reuses the same
`evaluate_clips` against synthetically mixed room audio. Tier C (in-situ detection
rate) isn't computable at all until real operational history exists — see
comprehensive-plan.md §4.5 for why that's stated up front rather than hidden.

Usage:
    uv run --project training python -m echo_sentry_train.evaluate \\
        --model-dir models/es-01 \\
        --normal-dir data/mimii/fan/normal --anomaly-dir data/mimii/fan/abnormal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from echo_sentry_train.dataset import Normalization
from echo_sentry_train.features import pcm_to_vectors
from echo_sentry_train.model import load_trained_scorer, split_context_target

PAUC_FPR = 0.1  # p for partial AUC — matches the DCASE Task 2 convention


def _load_pcm_int16(path: str) -> np.ndarray:
    data, _sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    return data


def clip_score(scorer, norm: Normalization, wav_path: str) -> float:
    """One score per clip = mean of its per-vector scores — comprehensive-plan.md §1.1,
    matching the MIMII/DCASE baseline's own scoring convention."""
    pcm = _load_pcm_int16(wav_path)
    vectors = pcm_to_vectors(pcm)
    if len(vectors) == 0:
        return float("nan")
    scores = scorer.score(norm.apply(vectors))
    return float(np.mean(scores))


def evaluate_clips(scorer, norm: Normalization, normal_paths: list[str],
                    anomalous_paths: list[str]) -> dict:
    normal_scores = np.array([clip_score(scorer, norm, p) for p in normal_paths])
    anomalous_scores = np.array([clip_score(scorer, norm, p) for p in anomalous_paths])
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
    y_score = np.concatenate([normal_scores, anomalous_scores])

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "pauc": float(roc_auc_score(y_true, y_score, max_fpr=PAUC_FPR)),
        "n_normal": len(normal_paths), "n_anomalous": len(anomalous_paths),
    }


def run_tflite_int8(tflite_path: Path, x_float: np.ndarray) -> np.ndarray:
    """Runs a fully-int8 .tflite model over already-normalized float vectors, handling
    the input/output quantization affine transform, and returns dequantized outputs
    comparable to the fp32 model's — needed for the fp32-vs-int8 AUC delta gate.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_scale, in_zero = in_detail["quantization"]
    out_scale, out_zero = out_detail["quantization"]

    outputs = np.empty((len(x_float), *out_detail["shape"][1:]), dtype=np.float32)
    for i, row in enumerate(x_float):
        quantized = np.round(row / in_scale + in_zero).astype(in_detail["dtype"])
        interpreter.set_tensor(in_detail["index"], quantized[np.newaxis, ...])
        interpreter.invoke()
        raw_out = interpreter.get_tensor(out_detail["index"])[0]
        outputs[i] = (raw_out.astype(np.float32) - out_zero) * out_scale
    return outputs


class _Int8Scorer:
    """Scores int8 .tflite output the same way the matching fp32 scorer would —
    reconstruction MSE for dense_ae/conv_ae, context/target MSE for idnn.
    """

    def __init__(self, tflite_path: Path, arch: str) -> None:
        self.tflite_path = tflite_path
        self.arch = arch

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.arch == "idnn":
            context, target = split_context_target(x)
            predicted = run_tflite_int8(self.tflite_path, context).reshape(len(x), -1)
            return np.mean((target - predicted) ** 2, axis=1)
        outputs = run_tflite_int8(self.tflite_path, x).reshape(len(x), -1)
        return np.mean((x - outputs) ** 2, axis=1)


def fp32_int8_delta(model_dir: Path, tflite_path: Path, normal_paths: list[str],
                     anomalous_paths: list[str]) -> dict:
    metadata = json.loads((model_dir / "train_metadata.json").read_text())
    norm = Normalization.from_dict(json.loads((model_dir / "normalization.json").read_text()))
    fp32_scorer = load_trained_scorer(model_dir)
    int8_scorer = _Int8Scorer(tflite_path, metadata["arch"])

    fp32_result = evaluate_clips(fp32_scorer, norm, normal_paths, anomalous_paths)
    int8_result = evaluate_clips(int8_scorer, norm, normal_paths, anomalous_paths)

    return {
        "auc_fp32": fp32_result["auc"], "auc_int8": int8_result["auc"],
        "delta": abs(fp32_result["auc"] - int8_result["auc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--normal-dir", required=True, type=Path)
    parser.add_argument("--anomaly-dir", required=True, type=Path)
    parser.add_argument("--tflite", type=Path, default=None,
                         help="also report the fp32->int8 AUC delta against this .tflite")
    args = parser.parse_args()

    norm = Normalization.from_dict(
        json.loads((args.model_dir / "normalization.json").read_text())
    )
    scorer = load_trained_scorer(args.model_dir)
    normal_paths = sorted(str(p) for p in args.normal_dir.glob("*.wav"))
    anomalous_paths = sorted(str(p) for p in args.anomaly_dir.glob("*.wav"))

    result = evaluate_clips(scorer, norm, normal_paths, anomalous_paths)
    print(f"AUC:  {result['auc']:.4f}")
    print(f"pAUC (p={PAUC_FPR}): {result['pauc']:.4f}")
    print(f"n_normal={result['n_normal']} n_anomalous={result['n_anomalous']}")
    print("Compare against comprehensive-plan.md §4.5's Tier A gate table.")

    if args.tflite:
        delta = fp32_int8_delta(args.model_dir, args.tflite, normal_paths, anomalous_paths)
        print()
        print(f"AUC fp32: {delta['auc_fp32']:.4f}  AUC int8: {delta['auc_int8']:.4f}  "
              f"delta: {delta['delta']:.4f}  (gate: <= 0.01)")


if __name__ == "__main__":
    main()
