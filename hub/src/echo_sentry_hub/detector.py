"""The six-stage anomaly decision pipeline — comprehensive-plan.md §5.

Pure, synchronous, and independent of MQTT/asyncio so it can be unit-tested and
driven by tools/replay.py without a broker in the loop. `ingest.py` calls
`Detector.process_score()` once per incoming score; `alerting.py` reacts to the
`DetectorResult` it returns.

Stage order (each feeds the next):
  1. Robust baseline calibration (median/MAD -> z)      -- recalibrate_baseline()
  2. Rolling median filter (kills single-clip impulses)  -- process_score()
  3. Cross-node common-mode rejection                    -- process_score()
  4. EWMA smoothing                                      -- process_score()
  5. Hysteresis state machine with N-of-M persistence     -- process_score()
  6. CUSUM for slow degradation                          -- process_score()
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal

from echo_sentry_hub.config import DetectorConfig
from echo_sentry_hub.store import DetectorState, Store

MAD_TO_SIGMA = 0.6745  # rescales MAD to be comparable to a Gaussian standard deviation
MIN_BASELINE_SAMPLES = 100  # below this, treat the node as still bootstrapping

DetectorStateName = Literal["NORMAL", "WATCH", "ALARM"]


@dataclass
class DetectorResult:
    node: str
    ts: int
    bootstrapping: bool  # True until enough baseline history exists to compute z
    z: float
    z_corrected: float
    ewma: float
    cusum: float
    state: DetectorStateName
    prev_state: DetectorStateName
    common_mode_applied: bool
    degradation_trend: bool  # CUSUM tripped h — routes to digest, never pages directly
    reporting_nodes: int = 0
    transitions: list[str] = field(default_factory=list)  # e.g. ["WATCH->ALARM"]

    @property
    def state_changed(self) -> bool:
        return self.state != self.prev_state


class Detector:
    def __init__(self, store: Store, config: DetectorConfig, node_ids: list[str]) -> None:
        self.store = store
        self.config = config
        self.node_ids = node_ids

    # ---- Stage 1: baseline calibration -------------------------------------

    def recalibrate_baseline(self, node: str, now_ts: int) -> bool:
        """Recompute median/MAD over the trailing baseline window. Call nightly per node.

        Returns True if the baseline was (re)computed, False if there still isn't enough
        confirmed-normal history to do so (comprehensive-plan.md §5 Stage 1).
        """
        since = now_ts - self.config.baseline_window_days * 86400
        samples = self.store.baseline_sample(node=node, since_ts=since, until_ts=now_ts)
        if len(samples) < MIN_BASELINE_SAMPLES:
            return False

        median = statistics.median(samples)
        mad = statistics.median(abs(x - median) for x in samples)

        state = self._load_state(node)
        state.baseline_median = median
        state.baseline_mad = mad
        state.baseline_updated = now_ts
        self.store.save_detector_state(state)
        return True

    # ---- Stages 2-6: per-score processing ----------------------------------

    def process_score(self, *, node: str, err_mean: float, ts: int) -> DetectorResult:
        cfg = self.config
        state = self._load_state(node)
        prev_state_name: DetectorStateName = state.state  # type: ignore[assignment]

        # Stage 1 (lookup): if no baseline yet, this node is bootstrapping.
        if state.baseline_median is None or not state.baseline_mad:
            self.store.save_detector_state(state)
            return DetectorResult(
                node=node, ts=ts, bootstrapping=True, z=0.0, z_corrected=0.0,
                ewma=state.ewma, cusum=state.cusum, state=prev_state_name,
                prev_state=prev_state_name, common_mode_applied=False,
                degradation_trend=False,
            )

        z_raw = MAD_TO_SIGMA * (err_mean - state.baseline_median) / state.baseline_mad

        # Stage 2: rolling median filter over this node's own recent z values.
        median_buf = [*state.median_buf, z_raw][-cfg.median_filter_window :]
        z_med = statistics.median(median_buf)

        # Stage 3: cross-node common-mode rejection.
        peers = self.store.latest_scores_by_node(
            as_of_ts=ts, max_age_s=2 * cfg.score_interval_s
        )
        peers.pop(node, None)
        all_z = {**peers, node: z_med}
        common_mode_applied = len(all_z) >= cfg.min_nodes_for_common_mode
        if common_mode_applied:
            cross_median = statistics.median(all_z.values())
            z_corrected = z_med - cross_median
        else:
            z_corrected = z_med

        # Stage 4: EWMA smoothing.
        ewma = cfg.ewma_alpha * z_corrected + (1 - cfg.ewma_alpha) * state.ewma

        # Stage 6: CUSUM (independent of the hysteresis state machine below).
        cusum = max(0.0, state.cusum + z_corrected - cfg.cusum_k)
        degradation_trend = cusum >= cfg.cusum_h

        # Stage 5: hysteresis state machine with N-of-M persistence.
        hysteresis_buf = [*state.hysteresis_buf, int(ewma >= cfg.alarm_enter)]
        hysteresis_buf = hysteresis_buf[-cfg.alarm_enter_m :]
        n_of_m_met = sum(hysteresis_buf) >= cfg.alarm_enter_n

        new_state, watch_exit_since, transitions = self._transition(
            prev_state_name, ewma, n_of_m_met, ts, state.watch_exit_since,
        )

        state.baseline_updated = state.baseline_updated  # unchanged here
        state.ewma = ewma
        state.cusum = cusum
        state.state = new_state
        state.state_entered = ts if new_state != prev_state_name else state.state_entered
        state.watch_exit_since = watch_exit_since
        state.median_buf = median_buf
        state.hysteresis_buf = hysteresis_buf
        state.last_score_ts = ts
        self.store.save_detector_state(state)

        return DetectorResult(
            node=node, ts=ts, bootstrapping=False, z=z_med, z_corrected=z_corrected,
            ewma=ewma, cusum=cusum, state=new_state, prev_state=prev_state_name,
            common_mode_applied=common_mode_applied, degradation_trend=degradation_trend,
            reporting_nodes=len(all_z), transitions=transitions,
        )

    def _transition(
        self, current: DetectorStateName, ewma: float, n_of_m_met: bool, ts: int,
        watch_exit_since: int | None,
    ) -> tuple[DetectorStateName, int | None, list[str]]:
        cfg = self.config
        new_state: DetectorStateName = current
        transitions: list[str] = []

        if current == "NORMAL":
            if ewma >= cfg.watch_enter:
                new_state = "WATCH"
            watch_exit_since = None

        elif current == "WATCH":
            if ewma >= cfg.alarm_enter and n_of_m_met:
                new_state = "ALARM"
                watch_exit_since = None
            elif ewma < cfg.watch_exit:
                watch_exit_since = watch_exit_since or ts
                if ts - watch_exit_since >= cfg.watch_exit_sustain_s:
                    new_state = "NORMAL"
                    watch_exit_since = None
            else:
                watch_exit_since = None  # back above the exit threshold; reset the timer

        elif current == "ALARM":
            if ewma < cfg.alarm_exit:
                new_state = "WATCH"
            watch_exit_since = None

        if new_state != current:
            transitions.append(f"{current}->{new_state}")

        return new_state, watch_exit_since, transitions

    # ---- helpers ------------------------------------------------------------

    def _load_state(self, node: str) -> DetectorState:
        state = self.store.get_detector_state(node)
        if state is not None:
            return state
        return DetectorState(
            node=node, baseline_median=None, baseline_mad=None, baseline_updated=None,
            ewma=0.0, cusum=0.0, state="NORMAL", state_entered=None, watch_exit_since=None,
            median_buf=[], hysteresis_buf=[], last_score_ts=None,
        )
