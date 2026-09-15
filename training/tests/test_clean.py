from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from echo_sentry_train.clean import (
    check_clipping,
    check_impulse,
    check_integrity,
    classify_machine_state,
    logbook_overlaps,
)


def test_check_integrity_flags_nonzero_drop_delta():
    assert check_integrity({"i2s_drops_delta": 0}) is None
    assert check_integrity({"i2s_drops_delta": 1}) == "integrity"
    assert check_integrity({}) is None  # missing field treated as zero


def test_check_clipping_flags_saturated_signal():
    quiet = np.full(16000, 0.1, dtype=np.float32)
    assert check_clipping(quiet) is None

    clipped = np.full(16000, 0.999, dtype=np.float32)
    assert check_clipping(clipped) == "clipping"


def test_check_clipping_tolerates_a_few_saturated_samples():
    pcm = np.full(16000, 0.1, dtype=np.float32)
    pcm[:1] = 0.999  # well under the 0.01% threshold at this length
    assert check_clipping(pcm) is None


def test_classify_machine_state_off_vs_normal():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1e-5, size=16000).astype(np.float32)
    loud = rng.normal(0, 0.2, size=16000).astype(np.float32)

    assert classify_machine_state(quiet, off_rms_dbfs=-60.0) == "off"
    assert classify_machine_state(loud, off_rms_dbfs=-60.0) == "normal"


def test_check_impulse_detects_a_spike_in_otherwise_steady_signal():
    rng = np.random.default_rng(1)
    steady = rng.normal(0, 0.05, size=16000).astype(np.float32)
    assert check_impulse(steady) is False

    spiky = steady.copy()
    spiky[8000:8010] = 0.9
    assert check_impulse(spiky) is True


def test_logbook_overlaps_respects_window():
    logbook = [{"timestamp_utc": "2026-09-15T12:00:00Z"}]
    base = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC).timestamp()

    assert logbook_overlaps(base + 10, logbook, window_s=60) is True
    assert logbook_overlaps(base + 3600, logbook, window_s=60) is False
