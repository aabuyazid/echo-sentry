from __future__ import annotations

import numpy as np
from echo_sentry_train.dataset import Normalization, fit_normalization


def test_fit_normalization_produces_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=(1000, 8)).astype(np.float32)
    norm = fit_normalization(x)
    normalized = norm.apply(x)

    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-2)


def test_normalization_roundtrips_through_dict():
    norm = Normalization(mean=np.array([1.0, 2.0], dtype=np.float32),
                          std=np.array([0.5, 1.5], dtype=np.float32))
    restored = Normalization.from_dict(norm.to_dict())
    np.testing.assert_array_equal(norm.mean, restored.mean)
    np.testing.assert_array_equal(norm.std, restored.std)


def test_fit_normalization_avoids_division_by_zero_on_constant_column():
    x = np.zeros((100, 4), dtype=np.float32)
    norm = fit_normalization(x)
    normalized = norm.apply(x)
    assert np.all(np.isfinite(normalized))
