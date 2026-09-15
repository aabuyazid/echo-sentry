from __future__ import annotations


def test_insert_score_dedups_on_node_seq(store):
    id1 = store.insert_score(
        node="es-01", ts=1000, seq=1, err_mean=0.1, err_p95=0.2, err_max=0.3,
        rms_dbfs=-40.0, model="m", cfg="c", fw="1.0", raw="{}",
    )
    id2 = store.insert_score(
        node="es-01", ts=1005, seq=1, err_mean=0.9, err_p95=0.9, err_max=0.9,
        rms_dbfs=-10.0, model="m", cfg="c", fw="1.0", raw="{}",
    )
    assert id1 is not None
    assert id2 is None  # duplicate (node, seq) — QoS1 redelivery is a no-op


def test_baseline_sample_excludes_open_alert_window(store):
    for i in range(5):
        store.insert_score(
            node="es-01", ts=1000 + i, seq=i, err_mean=0.1, err_p95=0.2, err_max=0.3,
            rms_dbfs=-40.0, model="m", cfg="c", fw="1.0", raw="{}",
        )
    store.open_alert(node="es-01", condition="anomaly", severity="alarm", opened_ts=1002,
                      peak_z=7.0)
    sample = store.baseline_sample(node="es-01", since_ts=1000, until_ts=1010)
    # ts 1002, 1003, 1004 fall inside the still-open alert window and must be excluded
    assert len(sample) == 2


def test_outbox_roundtrip(store):
    outbox_id = store.enqueue_outbox(sink="console", payload='{"x":1}', created_ts=100)
    pending = store.pending_outbox()
    assert len(pending) == 1
    assert pending[0].id == outbox_id

    store.mark_outbox_sent(outbox_id, sent_ts=200)
    assert store.pending_outbox() == []


def test_detector_state_persists_roundtrip(store):
    from echo_sentry_hub.store import DetectorState

    state = DetectorState(
        node="es-01", baseline_median=1.0, baseline_mad=0.5, baseline_updated=100,
        ewma=2.0, cusum=0.3, state="WATCH", state_entered=90, watch_exit_since=None,
        median_buf=[1.0, 2.0], hysteresis_buf=[0, 1],
        last_score_ts=100,
    )
    store.save_detector_state(state)
    loaded = store.get_detector_state("es-01")
    assert loaded is not None
    assert loaded.state == "WATCH"
    assert loaded.median_buf == [1.0, 2.0]
    assert loaded.hysteresis_buf == [0, 1]
