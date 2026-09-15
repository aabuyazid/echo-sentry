from __future__ import annotations

from echo_sentry_hub.alerting import AlertManager
from echo_sentry_hub.config import AlertingConfig
from echo_sentry_hub.detector import DetectorResult


def make_result(state, prev_state="NORMAL", ts=1000, degradation_trend=False, ewma=7.0):
    return DetectorResult(
        node="es-01", ts=ts, bootstrapping=False, z=ewma, z_corrected=ewma, ewma=ewma,
        cusum=0.0, state=state, prev_state=prev_state, common_mode_applied=True,
        degradation_trend=degradation_trend,
    )


def test_alarm_opens_one_alert_and_updates_on_repeat(store):
    manager = AlertManager(store, AlertingConfig())

    events = manager.handle_result(make_result("ALARM", "WATCH", ts=1000))
    assert len(events) == 1
    assert events[0].kind == "opened"
    alert_id = events[0].alert_id

    events = manager.handle_result(make_result("ALARM", "ALARM", ts=1010, ewma=8.0))
    assert len(events) == 1
    assert events[0].kind == "updated"
    assert events[0].alert_id == alert_id


def test_recovery_closes_the_alert(store):
    manager = AlertManager(store, AlertingConfig())
    manager.handle_result(make_result("ALARM", "WATCH", ts=1000))

    events = manager.handle_result(make_result("WATCH", "ALARM", ts=2000, ewma=4.5))
    assert len(events) == 1
    assert events[0].kind == "closed"

    alert = store.most_recent_alert(node="es-01", condition="anomaly")
    assert alert is not None
    assert alert.closed_ts == 2000


def test_cooldown_suppresses_reopen(store):
    cfg = AlertingConfig(cooldown_s=3600)
    manager = AlertManager(store, cfg)

    manager.handle_result(make_result("ALARM", "WATCH", ts=1000))
    manager.handle_result(make_result("NORMAL", "ALARM", ts=1100))  # closes it

    # Re-trips within the cooldown window: must not open a second alert or notify.
    events = manager.handle_result(make_result("ALARM", "WATCH", ts=1200))
    assert events == []

    # Past the cooldown window, it's allowed to reopen.
    events = manager.handle_result(make_result("ALARM", "WATCH", ts=1100 + 3601))
    assert len(events) == 1
    assert events[0].kind == "opened"


def test_degradation_trend_tracked_independently_of_anomaly_condition(store):
    manager = AlertManager(store, AlertingConfig())

    events = manager.handle_result(make_result("NORMAL", "NORMAL", ts=1000,
                                                 degradation_trend=True))
    assert len(events) == 1
    assert events[0].kind == "digest_queued"
    assert events[0].condition == "degradation_trend"

    # Still open even though the primary "anomaly" condition never fires.
    open_anomaly = store.get_open_alert(node="es-01", condition="anomaly")
    assert open_anomaly is None
    open_digest = store.get_open_alert(node="es-01", condition="degradation_trend")
    assert open_digest is not None


def test_bootstrapping_result_produces_no_events(store):
    manager = AlertManager(store, AlertingConfig())
    result = DetectorResult(
        node="es-01", ts=1, bootstrapping=True, z=0.0, z_corrected=0.0, ewma=0.0,
        cusum=0.0, state="NORMAL", prev_state="NORMAL", common_mode_applied=False,
        degradation_trend=False,
    )
    assert manager.handle_result(result) == []


def test_record_verdict_writes_back_to_alert(store):
    manager = AlertManager(store, AlertingConfig())
    events = manager.handle_result(make_result("ALARM", "WATCH", ts=1000))
    alert_id = events[0].alert_id

    manager.record_verdict(alert_id, verdict="false_alarm", verdict_by="ops@example.com",
                            verdict_ts=1500)

    alert = store.most_recent_alert(node="es-01", condition="anomaly")
    assert alert is not None
    assert alert.verdict == "false_alarm"
    assert alert.verdict_by == "ops@example.com"
