"""Alert lifecycle: dedup, cooldown, escalation, auto-resolve — comprehensive-plan.md §6.

Consumes a detector.DetectorResult, decides what (if anything) changes in the `alert`
table, and returns a list of AlertEvent for the caller (__main__.py's wiring) to hand
to forwarder.py / sinks. This module never talks to MQTT, HTTP, or a sink directly —
that separation is what makes it unit-testable against a plain Store.

Two independent conditions are tracked per node:
  - "anomaly"            — driven by the Stage-5 hysteresis state machine; ALARM pages,
                            WATCH/NORMAL never does.
  - "degradation_trend"  — driven by Stage-6 CUSUM; always digest-only, never pages
                            directly (comprehensive-plan.md §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from echo_sentry_hub.config import AlertingConfig
from echo_sentry_hub.detector import DetectorResult
from echo_sentry_hub.store import Store

EventKind = Literal["opened", "updated", "closed", "digest_queued", "digest_closed"]


@dataclass
class AlertEvent:
    kind: EventKind
    alert_id: int
    node: str
    condition: str
    severity: str
    ts: int
    peak: float


class AlertManager:
    def __init__(self, store: Store, config: AlertingConfig) -> None:
        self.store = store
        self.config = config

    def handle_result(self, result: DetectorResult) -> list[AlertEvent]:
        if result.bootstrapping:
            return []
        events: list[AlertEvent] = []
        events += self._handle_anomaly(result)
        events += self._handle_degradation(result)
        return events

    # ---- primary condition: hysteresis ALARM -----------------------------------

    def _handle_anomaly(self, result: DetectorResult) -> list[AlertEvent]:
        condition = "anomaly"
        open_alert = self.store.get_open_alert(node=result.node, condition=condition)

        if result.state == "ALARM":
            if open_alert is not None:
                self.store.bump_alert_peak(open_alert.id, result.ewma)
                return [AlertEvent("updated", open_alert.id, result.node, condition,
                                    "alarm", result.ts, result.ewma)]

            if self._in_cooldown(result.node, condition, result.ts):
                return []  # suppressed — re-tripped too soon after the last one closed

            alert_id = self.store.open_alert(
                node=result.node, condition=condition, severity="alarm",
                opened_ts=result.ts, peak_z=result.ewma,
            )
            return [AlertEvent("opened", alert_id, result.node, condition, "alarm",
                                result.ts, result.ewma)]

        # WATCH or NORMAL: auto-resolve any open ALARM.
        if open_alert is not None:
            self.store.close_alert(open_alert.id, result.ts)
            return [AlertEvent("closed", open_alert.id, result.node, condition, "alarm",
                                result.ts, result.ewma)]
        return []

    # ---- secondary condition: CUSUM degradation trend (digest-only) ---------------

    def _handle_degradation(self, result: DetectorResult) -> list[AlertEvent]:
        condition = "degradation_trend"
        open_alert = self.store.get_open_alert(node=result.node, condition=condition)

        if result.degradation_trend:
            if open_alert is not None:
                self.store.bump_alert_peak(open_alert.id, result.cusum)
                return []
            alert_id = self.store.open_alert(
                node=result.node, condition=condition, severity="digest",
                opened_ts=result.ts, peak_z=result.cusum,
            )
            return [AlertEvent("digest_queued", alert_id, result.node, condition, "digest",
                                result.ts, result.cusum)]

        if open_alert is not None:
            self.store.close_alert(open_alert.id, result.ts)
            return [AlertEvent("digest_closed", open_alert.id, result.node, condition,
                                "digest", result.ts, result.cusum)]
        return []

    # ---- feedback loop: Teams Confirm / False alarm buttons -----------------------

    def record_verdict(self, alert_id: int, *, verdict: str, verdict_by: str,
                        verdict_ts: int) -> None:
        """Writes an operator's verdict back to the alert row.

        Feeds the evidence flywheel (comprehensive-plan.md §3.4): a "false_alarm" verdict
        is exactly the signal tools/label.py needs to grow the nuisance set, and a
        "confirmed" verdict is what eventually gives us a real in-situ ROC curve.
        """
        self.store.set_alert_verdict(
            alert_id, verdict=verdict, verdict_by=verdict_by, verdict_ts=verdict_ts
        )

    # ---- cooldown -----------------------------------------------------------------

    def _in_cooldown(self, node: str, condition: str, now_ts: int) -> bool:
        recent = self.store.most_recent_alert(node=node, condition=condition)
        if recent is None or recent.closed_ts is None:
            return False
        return (now_ts - recent.closed_ts) < self.config.cooldown_s
