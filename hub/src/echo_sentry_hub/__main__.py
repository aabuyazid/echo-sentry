"""Entrypoint — wires ingest -> detector -> alerting -> outbox -> forwarder -> sinks,
plus the evidence HTTP intake, per comprehensive-plan.md §2.4.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time

import structlog
from aiohttp import web

from echo_sentry_hub.alerting import AlertEvent, AlertManager
from echo_sentry_hub.config import Settings
from echo_sentry_hub.detector import Detector
from echo_sentry_hub.evidence import build_app
from echo_sentry_hub.forwarder import Forwarder
from echo_sentry_hub.ingest import MqttIngest
from echo_sentry_hub.models import ScorePayload
from echo_sentry_hub.sinks.base import Sink
from echo_sentry_hub.sinks.console import ConsoleSink
from echo_sentry_hub.sinks.powerbi import PowerBiSink
from echo_sentry_hub.sinks.teams import TeamsSink
from echo_sentry_hub.store import Store

log = structlog.get_logger(__name__)

# Conditions that never page directly — see comprehensive-plan.md §6. They still get
# recorded to the console sink (if enabled) for visibility; a real weekly-digest job
# would query `alert WHERE condition = 'degradation_trend'` on its own schedule.
_DIGEST_ONLY_CONDITIONS = {"degradation_trend"}


def configure_logging(level: str) -> None:
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
    )


class Hub:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = Store(settings.db_path)
        self.detector = Detector(self.store, settings.detector, settings.node_ids)
        self.alerting = AlertManager(self.store, settings.alerting)
        self.ingest = MqttIngest(settings.mqtt, settings.node_ids)

        self.sinks: dict[str, Sink] = {}
        if settings.teams.workflow_url:
            self.sinks["teams"] = TeamsSink(settings.teams)
        if settings.powerbi.push_url:
            self.sinks["powerbi"] = PowerBiSink(settings.powerbi)
        if settings.dev_sink or not self.sinks:
            self.sinks["console"] = ConsoleSink()

        self.forwarder = Forwarder(self.store, self.sinks)
        self._stop = asyncio.Event()

    async def run(self) -> None:
        self.store.connect()
        await self.ingest.start()

        app = build_app(self.store, self.settings.evidence)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(
            runner, self.settings.evidence.listen_host, self.settings.evidence.listen_port
        )
        await site.start()
        log.info(
            "hub.evidence_listening",
            host=self.settings.evidence.listen_host,
            port=self.settings.evidence.listen_port,
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass  # non-POSIX dev environments (e.g. Windows)

        tasks = [
            asyncio.create_task(self._consume_loop(), name="consume"),
            asyncio.create_task(self._baseline_loop(), name="baseline"),
            asyncio.create_task(self.forwarder.run(), name="forwarder"),
        ]
        try:
            await self._stop.wait()
        finally:
            self.forwarder.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await runner.cleanup()
            await self.ingest.stop()
            self.store.close()

    def request_stop(self) -> None:
        self._stop.set()

    async def _consume_loop(self) -> None:
        while True:
            message = await self.ingest.queue.get()
            if message.kind == "score":
                assert isinstance(message.payload, ScorePayload)
                await self._handle_score(message.node, message.payload)
            # health/status/evidence-announce messages are validated on receipt (see
            # ingest.py) but don't yet drive detector state; dead-man detection today
            # relies on MQTT's last-will "offline" status plus gaps in `seq`.

    async def _handle_score(self, node: str, payload: ScorePayload) -> None:
        expected = self.settings.expected_config_hash
        quarantined = expected is not None and payload.cfg != expected
        raw = payload.model_dump_json()

        score_id = await asyncio.to_thread(
            self.store.insert_score,
            node=node, ts=payload.t, seq=payload.seq, err_mean=payload.err_mean,
            err_p95=payload.err_p95, err_max=payload.err_max, rms_dbfs=payload.rms_dbfs,
            model=payload.model, cfg=payload.cfg, fw=payload.fw, raw=raw,
            quarantined=quarantined,
            quarantine_reason="cfg_mismatch" if quarantined else None,
        )
        if score_id is None:
            return  # duplicate delivery (QoS1 redelivery) — already stored, nothing to do
        if quarantined:
            log.warning("hub.score_quarantined", node=node, cfg=payload.cfg, expected=expected)
            return

        result = await asyncio.to_thread(
            self.detector.process_score, node=node, err_mean=payload.err_mean, ts=payload.t
        )
        await asyncio.to_thread(
            self.store.update_score_detector_outputs, score_id,
            z=result.z, z_corrected=result.z_corrected, z_ewma=result.ewma,
            cusum=result.cusum, state=result.state,
        )

        events = await asyncio.to_thread(self.alerting.handle_result, result)
        for event in events:
            await self._enqueue_notifications(event)

    async def _enqueue_notifications(self, event: AlertEvent) -> None:
        payload = {
            "kind": event.kind, "alert_id": event.alert_id, "node": event.node,
            "condition": event.condition, "severity": event.severity, "ts": event.ts,
            "peak": event.peak,
        }
        body = json.dumps(payload)
        target_sinks = list(self.sinks)
        if event.condition in _DIGEST_ONLY_CONDITIONS:
            target_sinks = [s for s in target_sinks if s == "console"]
        for sink_name in target_sinks:
            await asyncio.to_thread(
                self.store.enqueue_outbox, sink=sink_name, payload=body, created_ts=event.ts
            )

    async def _baseline_loop(self) -> None:
        """Recalibrates each node's baseline (comprehensive-plan.md §5, Stage 1).

        Checked hourly rather than strictly at midnight — recalibrate_baseline() is
        cheap and idempotent, so the exact cadence doesn't matter as long as it's
        regular; comprehensive-plan.md's "nightly" is a recommendation, not a
        correctness requirement.
        """
        while True:
            now = int(time.time())
            for node in self.settings.node_ids:
                await asyncio.to_thread(self.detector.recalibrate_baseline, node, now)
            await asyncio.sleep(3600)


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    hub = Hub(settings)
    asyncio.run(hub.run())


if __name__ == "__main__":
    main()
