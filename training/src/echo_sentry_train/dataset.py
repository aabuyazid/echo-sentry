"""Loads a manifest (clean.py) and builds feature-vector arrays for training/eval —
comprehensive-plan.md §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from echo_sentry_train import frontend_config as fc
from echo_sentry_train.clean import ManifestRow, read_manifest
from echo_sentry_train.features import pcm_to_vectors


def _load_pcm_int16(path: str) -> np.ndarray:
    data, _sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    return data


def vectors_for_rows(rows: list[ManifestRow]) -> np.ndarray:
    """Concatenated feature vectors across every WAV named in `rows`."""
    chunks = []
    for row in rows:
        pcm = _load_pcm_int16(row.path)
        vecs = pcm_to_vectors(pcm)
        if vecs.size:
            chunks.append(vecs)
    if not chunks:
        return np.empty((0, fc.VECTOR_DIM), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def load_split(manifest_path: Path, node: str, split: str) -> np.ndarray:
    rows = [r for r in read_manifest(manifest_path)
            if r.node == node and r.split == split and r.reject_reason is None]
    return vectors_for_rows(rows)


@dataclass
class Normalization:
    mean: np.ndarray  # (VECTOR_DIM,)
    std: np.ndarray  # (VECTOR_DIM,)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        return (vectors - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Normalization:
        return cls(mean=np.array(d["mean"], dtype=np.float32),
                    std=np.array(d["std"], dtype=np.float32))


def fit_normalization(train_vectors: np.ndarray, *, eps: float = 1e-6) -> Normalization:
    """Per-mel-band mean/std computed on the TRAINING split only —
    comprehensive-plan.md §4.3: computing over the full dataset leaks test statistics.
    """
    mean = train_vectors.mean(axis=0)
    std = train_vectors.std(axis=0) + eps
    return Normalization(mean=mean.astype(np.float32), std=std.astype(np.float32))
