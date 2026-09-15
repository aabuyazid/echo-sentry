from __future__ import annotations

import numpy as np
import pytest
from echo_sentry_train import frontend_config as fc
from echo_sentry_train.features import compute_log_mel_frames, frames_to_vectors, pcm_to_vectors


def test_compute_log_mel_frames_shape():
    rng = np.random.default_rng(0)
    pcm = (rng.normal(0, 0.1, size=fc.SAMPLE_RATE_HZ) * 32767).astype(np.int16)
    frames = compute_log_mel_frames(pcm)
    assert frames.ndim == 2
    assert frames.shape[1] == fc.MEL_CHANNELS
    assert frames.shape[0] > 0


def test_compute_log_mel_frames_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        compute_log_mel_frames(np.zeros(100, dtype=np.float32))


def test_compute_log_mel_frames_rejects_non_1d():
    with pytest.raises(ValueError):
        compute_log_mel_frames(np.zeros((10, 10), dtype=np.int16))


def test_frames_to_vectors_shape_and_stride():
    mel = fc.MEL_CHANNELS
    frames = np.arange(10 * mel, dtype=np.float32).reshape(10, mel)
    vectors = frames_to_vectors(frames)

    fpv, stride = fc.FRAMES_PER_VECTOR, fc.VECTOR_STRIDE
    expected_n = ((10 - fpv) // stride) + 1
    assert vectors.shape == (expected_n, fpv * mel)
    # first vector is exactly the first fpv frames, flattened
    np.testing.assert_array_equal(vectors[0], frames[:fpv].reshape(-1))


def test_frames_to_vectors_empty_when_too_few_frames():
    mel = fc.MEL_CHANNELS
    frames = np.zeros((fc.FRAMES_PER_VECTOR - 1, mel), dtype=np.float32)
    vectors = frames_to_vectors(frames)
    assert vectors.shape == (0, fc.VECTOR_DIM)


def test_pcm_to_vectors_end_to_end():
    rng = np.random.default_rng(1)
    pcm = (rng.normal(0, 0.1, size=2 * fc.SAMPLE_RATE_HZ) * 32767).astype(np.int16)
    vectors = pcm_to_vectors(pcm)
    assert vectors.shape[1] == fc.VECTOR_DIM
    assert vectors.shape[0] > 0
