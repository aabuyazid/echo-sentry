from __future__ import annotations

import random

from echo_sentry_hub.config import DetectorConfig
from echo_sentry_hub.detector import MIN_BASELINE_SAMPLES, Detector


def feed(detector: Detector, node: str, err_mean: float, ts: int, seq: int):
    """Mirrors __main__.Hub._handle_score's insert -> process -> writeback sequence,
    since cross-node common-mode rejection reads peer z values back out of the score
    table rather than from Detector's in-memory state.
    """
    detector.store.insert_score(
        node=node, ts=ts, seq=seq, err_mean=err_mean, err_p95=err_mean, err_max=err_mean,
        rms_dbfs=-40.0, model="m", cfg="c", fw="1.0", raw="{}",
    )
    result = detector.process_score(node=node, err_mean=err_mean, ts=ts)
    detector.store.update_score_detector_outputs(
        detector.store.conn.execute(
            "SELECT id FROM score WHERE node=? AND seq=?", (node, seq)
        ).fetchone()["id"],
        z=result.z, z_corrected=result.z_corrected, z_ewma=result.ewma, cusum=result.cusum,
        state=result.state,
    )
    return result


def seed_baseline(detector: Detector, node: str, *, n: int = 200, mean: float = 1.0,
                   spread: float = 0.05, start_ts: int = 0) -> None:
    rng = random.Random(42)
    for i in range(n):
        detector.store.insert_score(
            node=node, ts=start_ts + i, seq=i, err_mean=mean + rng.uniform(-spread, spread),
            err_p95=mean, err_max=mean, rms_dbfs=-40.0, model="m", cfg="c", fw="1.0", raw="{}",
        )
    detector.recalibrate_baseline(node, now_ts=start_ts + n + 1)


def test_bootstrapping_with_no_baseline_returns_zero(store):
    detector = Detector(store, DetectorConfig(), ["es-01"])
    result = feed(detector, "es-01", err_mean=0.5, ts=1, seq=1)
    assert result.bootstrapping is True
    assert result.z == 0.0
    assert result.state == "NORMAL"


def test_recalibrate_baseline_requires_minimum_samples(store):
    detector = Detector(store, DetectorConfig(), ["es-01"])
    seed_baseline(detector, "es-01", n=MIN_BASELINE_SAMPLES - 1)
    state = store.get_detector_state("es-01")
    assert state is None or state.baseline_median is None

    seed_baseline(detector, "es-01", n=MIN_BASELINE_SAMPLES + 50, start_ts=10_000)
    state = store.get_detector_state("es-01")
    assert state is not None
    assert state.baseline_median is not None
    assert state.baseline_mad is not None


def test_normal_score_stays_normal(store):
    cfg = DetectorConfig(min_nodes_for_common_mode=99)  # disable common-mode for this test
    detector = Detector(store, cfg, ["es-01"])
    seed_baseline(detector, "es-01", n=300, mean=1.0, spread=0.05)

    result = feed(detector, "es-01", err_mean=1.0, ts=100_000, seq=9000)
    assert result.bootstrapping is False
    assert result.state == "NORMAL"
    assert abs(result.z) < 3.0


def test_sustained_spike_escalates_to_alarm(store):
    cfg = DetectorConfig(min_nodes_for_common_mode=99, ewma_alpha=0.5)  # fast EWMA for the test
    detector = Detector(store, cfg, ["es-01"])
    seed_baseline(detector, "es-01", n=300, mean=1.0, spread=0.02)

    seq = 9000
    ts = 100_000
    last_state = "NORMAL"
    # A large, sustained deviation should eventually trip WATCH then ALARM.
    for _ in range(20):
        result = feed(detector, "es-01", err_mean=5.0, ts=ts, seq=seq)
        last_state = result.state
        seq += 1
        ts += cfg.score_interval_s
        if last_state == "ALARM":
            break

    assert last_state == "ALARM"


def test_alarm_recovers_to_watch_then_normal(store):
    cfg = DetectorConfig(min_nodes_for_common_mode=99, ewma_alpha=0.5, watch_exit_sustain_s=0)
    detector = Detector(store, cfg, ["es-01"])
    seed_baseline(detector, "es-01", n=300, mean=1.0, spread=0.02)

    seq, ts = 9000, 100_000
    for _ in range(20):
        result = feed(detector, "es-01", err_mean=5.0, ts=ts, seq=seq)
        seq += 1
        ts += cfg.score_interval_s
        if result.state == "ALARM":
            break
    assert result.state == "ALARM"

    for _ in range(10):
        result = feed(detector, "es-01", err_mean=1.0, ts=ts, seq=seq)
        seq += 1
        ts += cfg.score_interval_s
        if result.state == "NORMAL":
            break

    assert result.state == "NORMAL"


def test_common_mode_rejection_suppresses_simultaneous_room_wide_spike(store):
    nodes = [f"es-0{i}" for i in range(1, 6)]
    cfg = DetectorConfig(min_nodes_for_common_mode=3, ewma_alpha=0.5)
    detector = Detector(store, cfg, nodes)
    for n in nodes:
        seed_baseline(detector, n, n=300, mean=1.0, spread=0.02, start_ts=0)

    # All five nodes spike together — an environmental cause, not five simultaneous faults.
    seq, ts = 9000, 200_000
    last_states = {}
    for _round in range(15):
        for n in nodes:
            last_states[n] = feed(detector, n, err_mean=5.0, ts=ts, seq=seq).state
        seq += 1
        ts += cfg.score_interval_s

    assert all(state != "ALARM" for state in last_states.values()), last_states


def test_lone_node_spike_still_alarms_with_common_mode_enabled(store):
    nodes = [f"es-0{i}" for i in range(1, 6)]
    cfg = DetectorConfig(min_nodes_for_common_mode=3, ewma_alpha=0.5)
    detector = Detector(store, cfg, nodes)
    for n in nodes:
        seed_baseline(detector, n, n=300, mean=1.0, spread=0.02, start_ts=0)

    seq, ts = 9000, 200_000
    last_state = "NORMAL"
    for _round in range(20):
        for n in nodes:
            err = 5.0 if n == "es-01" else 1.0
            result = feed(detector, n, err_mean=err, ts=ts, seq=seq)
            if n == "es-01":
                last_state = result.state
        seq += 1
        ts += cfg.score_interval_s
        if last_state == "ALARM":
            break

    assert last_state == "ALARM"


def test_cusum_flags_slow_drift_below_hysteresis_threshold(store):
    # A drift too small to ever trip watch_enter should still accumulate in CUSUM.
    cfg = DetectorConfig(min_nodes_for_common_mode=99, ewma_alpha=0.02, watch_enter=10.0,
                          cusum_k=0.2, cusum_h=3.0)
    detector = Detector(store, cfg, ["es-01"])
    seed_baseline(detector, "es-01", n=300, mean=1.0, spread=0.02)

    seq, ts = 9000, 100_000
    degraded = False
    for _ in range(200):
        result = feed(detector, "es-01", err_mean=1.06, ts=ts, seq=seq)  # small, persistent bias
        assert result.state == "NORMAL"  # never big enough to trip the hysteresis machine
        if result.degradation_trend:
            degraded = True
            break
        seq += 1
        ts += cfg.score_interval_s

    assert degraded
