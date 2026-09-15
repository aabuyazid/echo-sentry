from __future__ import annotations

import pytest
from echo_sentry_hub.forwarder import Forwarder


class FlakySink:
    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.received: list[dict] = []

    async def send(self, payload: dict) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient failure")
        self.received.append(payload)


class AlwaysFailsSink:
    name = "broken"

    async def send(self, payload: dict) -> None:
        raise RuntimeError("permanently broken")


@pytest.mark.asyncio
async def test_forwarder_retries_and_eventually_delivers(store):
    sink = FlakySink(fail_times=2)
    forwarder = Forwarder(store, {"flaky": sink}, max_attempts=5)
    store.enqueue_outbox(sink="flaky", payload='{"a":1}', created_ts=1)

    await forwarder.drain_once()

    assert sink.received == [{"a": 1}]
    assert store.pending_outbox() == []


@pytest.mark.asyncio
async def test_forwarder_marks_failed_row_still_pending(store):
    forwarder = Forwarder(store, {"broken": AlwaysFailsSink()}, max_attempts=2)
    outbox_id = store.enqueue_outbox(sink="broken", payload="{}", created_ts=1)

    await forwarder.drain_once()

    pending = store.pending_outbox()
    assert len(pending) == 1
    assert pending[0].id == outbox_id
    assert pending[0].attempts == 1
    assert pending[0].last_error is not None


@pytest.mark.asyncio
async def test_forwarder_skips_unknown_sink_without_crashing(store):
    forwarder = Forwarder(store, {}, max_attempts=1)
    store.enqueue_outbox(sink="nonexistent", payload="{}", created_ts=1)

    await forwarder.drain_once()  # must not raise

    assert len(store.pending_outbox()) == 1  # left pending, not lost
