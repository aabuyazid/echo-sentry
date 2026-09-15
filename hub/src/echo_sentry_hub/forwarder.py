"""Store-and-forward drain loop — comprehensive-plan.md §6.

Every outbound notification is written to the `outbox` table before delivery is
attempted (alerting.py's events get turned into outbox rows by __main__.py). This
loop drains it continuously, retrying failed sends with exponential backoff, so a
transient Ethernet outage (the hub's only egress — comprehensive-plan.md §2.1) delays
delivery instead of losing it. No duplicate sends: a row is marked `sent_ts` only
after `Sink.send()` returns without raising.
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from echo_sentry_hub.sinks.base import Sink
from echo_sentry_hub.store import Store

log = structlog.get_logger(__name__)


class Forwarder:
    def __init__(self, store: Store, sinks: dict[str, Sink], *, poll_interval_s: float = 2.0,
                 max_attempts: int = 3) -> None:
        self.store = store
        self.sinks = sinks
        self.poll_interval_s = poll_interval_s
        self.max_attempts = max_attempts
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            await self.drain_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)
            except TimeoutError:
                pass

    async def drain_once(self) -> None:
        rows = await asyncio.to_thread(self.store.pending_outbox, limit=50)
        for row in rows:
            sink = self.sinks.get(row.sink)
            if sink is None:
                log.warning("forwarder.unknown_sink", sink=row.sink, outbox_id=row.id)
                continue
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_attempts),
                    wait=wait_exponential(multiplier=1, min=1, max=30),
                    reraise=True,
                ):
                    with attempt:
                        payload = json.loads(row.payload)
                        await sink.send(payload)
                await asyncio.to_thread(self.store.mark_outbox_sent, row.id, int(time.time()))
            except Exception as exc:  # noqa: BLE001 - any sink failure must be recorded, not swallowed
                log.warning("forwarder.send_failed", sink=row.sink, outbox_id=row.id,
                            error=str(exc))
                await asyncio.to_thread(self.store.mark_outbox_failed, row.id, str(exc))
