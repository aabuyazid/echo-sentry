"""End-to-end integration test for the whole training pipeline: clean -> train ->
quantize -> calibrate -> export -> evaluate, using synthetic captures.

No real MIMII/DCASE corpus is downloaded in this environment (comprehensive-plan.md
§4.5 Tier A needs one, and that's a large external download deliberately left as a
follow-up rather than fetched automatically) — this instead exercises every stage of
the pipeline with a synthetic "steady machine hum" signal, which is enough to catch
wiring bugs (shape mismatches, split logic, quantization, sidecar generation) even
though it says nothing about real-world AUC.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest
from echo_sentry_train import frontend_config as fc
from echo_sentry_train.calibrate import calibrate_node
from echo_sentry_train.clean import build_manifest, write_manifest
from echo_sentry_train.dataset import Normalization
from echo_sentry_train.evaluate import evaluate_clips, fp32_int8_delta
from echo_sentry_train.export import export_model_json, export_normalization_bin
from echo_sentry_train.model import load_trained_scorer
from echo_sentry_train.quantize import quantize
from echo_sentry_train.train import train_node

SR = fc.SAMPLE_RATE_HZ
NODE = "es-01"


def _write_wav(path: Path, pcm_int16: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm_int16.tobytes())


def _hum(duration_s: float, *, rng: np.random.Generator, amplitude: float = 0.15,
         freq_hz: float = 220.0) -> np.ndarray:
    """Synthetic "steady machine noise": a tone plus broadband noise, standing in for
    real bearing/rotation acoustics — enough spectral structure for the AE to learn
    something non-trivial, without needing a real recording."""
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t)
    signal += rng.normal(0, amplitude * 0.2, size=n)
    return np.clip(signal * 32767, -32767, 32767).astype(np.int16)


def _write_capture_segment(captures_dir: Path, node: str, *, start_utc: float,
                            duration_s: float, pcm: np.ndarray, drops_delta: int = 0,
                            cumulative_drops: int = 0) -> None:
    node_dir = captures_dir / node
    node_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(start_utc)}"
    wav_path = node_dir / f"{stamp}.wav"
    json_path = node_dir / f"{stamp}.json"
    _write_wav(wav_path, pcm)
    json_path.write_text(json.dumps({
        "node_id": node, "fw": "test", "cfg": fc.CONFIG_HASH,
        "start_utc": start_utc, "end_utc": start_utc + duration_s,
        "duration_s": duration_s, "i2s_drops_delta": drops_delta,
        "i2s_drops_cumulative": cumulative_drops, "session_note": "synthetic test",
    }))


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("training_pipeline")
    captures_dir = root / "captures"
    rng = np.random.default_rng(7)

    day_s = 86400
    base_ts = 1_800_000_000.0
    # 40 clean 6s "normal" segments spread across 10 simulated days -> plenty for the
    # chronological 60/20/20 split to have something in every bucket.
    for day in range(10):
        for seg in range(4):
            start = base_ts + day * day_s + seg * 3600
            pcm = _hum(6.0, rng=rng)
            _write_capture_segment(captures_dir, NODE, start_utc=start, duration_s=6.0, pcm=pcm)

    # One segment with a non-zero drop delta -> must be rejected ("integrity").
    _write_capture_segment(
        captures_dir, NODE, start_utc=base_ts + 50 * 3600, duration_s=6.0,
        pcm=_hum(6.0, rng=rng), drops_delta=2, cumulative_drops=2,
    )

    # One clipped segment -> must be rejected ("clipping").
    clipped = np.full(6 * SR, 32767, dtype=np.int16)
    _write_capture_segment(captures_dir, NODE, start_utc=base_ts + 51 * 3600, duration_s=6.0,
                            pcm=clipped)

    # One impulsive segment -> "nuisance" split, excluded from train/val/test.
    impulsive = _hum(6.0, rng=rng, amplitude=0.05)
    impulsive[SR * 3 : SR * 3 + 50] = 32000  # a sharp bang partway through
    _write_capture_segment(captures_dir, NODE, start_utc=base_ts + 52 * 3600, duration_s=6.0,
                            pcm=impulsive)

    # One near-silent segment -> "off" machine_state, excluded from AE training.
    silence = (rng.normal(0, 5, size=6 * SR)).astype(np.int16)
    _write_capture_segment(captures_dir, NODE, start_utc=base_ts + 53 * 3600, duration_s=6.0,
                            pcm=silence)

    rows = build_manifest(captures_dir, logbook_path=None)
    manifest_csv = root / "manifest.csv"
    write_manifest(rows, manifest_csv)
    return manifest_csv


def test_manifest_rejection_and_splits(manifest_path: Path):
    from echo_sentry_train.clean import read_manifest

    rows = read_manifest(manifest_path)
    by_reason = {}
    for r in rows:
        by_reason.setdefault(r.reject_reason, []).append(r)

    assert len(by_reason.get("integrity", [])) == 1
    assert len(by_reason.get("clipping", [])) == 1

    kept = by_reason.get(None, [])
    splits = {r.split for r in kept}
    assert "nuisance" in splits
    assert {"train", "val", "test"} <= splits  # off segment has split=None (excluded)

    off_rows = [r for r in kept if r.machine_state == "off"]
    assert len(off_rows) == 1
    assert off_rows[0].split is None  # excluded from AE splits, but not rejected


def test_full_pipeline_train_quantize_calibrate_export(manifest_path: Path, tmp_path: Path):
    model_dir = tmp_path / "model"
    metadata = train_node(manifest_path, NODE, "dense_ae", model_dir, epochs=5)
    assert metadata["train_vectors"] > 0
    assert (model_dir / "model.keras").exists()
    assert (model_dir / "normalization.json").exists()

    tflite_path = model_dir / "model.tflite"
    quant_result = quantize(model_dir, manifest_path, tflite_path)
    assert tflite_path.exists()
    assert quant_result["size_bytes"] > 0

    baseline = calibrate_node(manifest_path, NODE, model_dir)
    assert baseline["n_samples"] > 0
    assert baseline["baseline_mad"] >= 0

    manifest_json = export_model_json(model_dir, tflite_path, fp32_int8_auc_delta=0.0)
    assert manifest_json["config_hash"] == fc.CONFIG_HASH
    assert manifest_json["vector_dim"] == fc.VECTOR_DIM
    assert manifest_json["quantization"]["input_dtype"] == "int8"
    assert (model_dir / "model.json").exists()

    norm_bin_path = export_normalization_bin(model_dir)
    assert norm_bin_path.exists()
    assert norm_bin_path.stat().st_size == fc.VECTOR_DIM * 4 * 2  # mean + std, float32


def test_evaluate_separates_normal_from_synthetic_anomaly(manifest_path: Path, tmp_path: Path):
    model_dir = tmp_path / "model"
    train_node(manifest_path, NODE, "dense_ae", model_dir, epochs=10)
    tflite_path = model_dir / "model.tflite"
    quantize(model_dir, manifest_path, tflite_path)

    rng = np.random.default_rng(99)
    normal_dir, anomaly_dir = tmp_path / "normal", tmp_path / "anomaly"
    normal_dir.mkdir()
    anomaly_dir.mkdir()
    for i in range(10):
        _write_wav(normal_dir / f"n{i}.wav", _hum(3.0, rng=rng))
        # A synthetic "anomaly": added broadband noise + a second, higher-frequency tone —
        # a rough stand-in for a bearing fault's added spectral energy.
        anomalous = _hum(3.0, rng=rng, amplitude=0.15)
        extra = (0.3 * np.sin(2 * np.pi * 2000 * np.arange(3 * SR) / SR) * 32767)
        anomalous = np.clip(anomalous.astype(np.int32) + extra.astype(np.int32), -32767, 32767)
        _write_wav(anomaly_dir / f"a{i}.wav", anomalous.astype(np.int16))

    norm = Normalization.from_dict(json.loads((model_dir / "normalization.json").read_text()))
    scorer = load_trained_scorer(model_dir)
    normal_paths = sorted(str(p) for p in normal_dir.glob("*.wav"))
    anomaly_paths = sorted(str(p) for p in anomaly_dir.glob("*.wav"))

    result = evaluate_clips(scorer, norm, normal_paths, anomaly_paths)
    assert 0.0 <= result["auc"] <= 1.0
    assert result["auc"] > 0.5  # the injected tone should be at least somewhat detectable

    delta = fp32_int8_delta(model_dir, tflite_path, normal_paths, anomaly_paths)
    assert delta["delta"] < 0.5  # loose sanity bound; the real gate (<=0.01) needs real data
