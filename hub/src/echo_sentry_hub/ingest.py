"""MQTT ingest — comprehensive-plan.md §2.2(b) and §2.4.

paho-mqtt runs its own network thread (`loop_start()`); its callbacks fire there, not
on the asyncio loop, so they only decode + validate a payload and hand a typed message
across with `loop.call_soon_threadsafe()`. Everything downstream (detector, alerting,
store) runs on the asyncio event loop in __main__.py.

paho-mqtt 2.x requires `CallbackAPIVersion.VERSION2` in the Client constructor —
v1-style callback signatures raise immediately at connect time.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import paho.mqtt.client as mqtt
import structlog
from pydantic import ValidationError

from echo_sentry_hub.config import MqttConfig
from echo_sentry_hub.models import EvidencePayload, HealthPayload, ScorePayload, StatusPayload

log = structlog.get_logger(__name__)

_PAYLOAD_MODELS: dict[str, type] = {
    "score": ScorePayload,
    "health": HealthPayload,
    "status": StatusPayload,
    "evidence": EvidencePayload,
}


@dataclass
class IngestMessage:
    kind: str  # "score" | "health" | "status" | "evidence"
    node: str
    payload: ScorePayload | HealthPayload | StatusPayload | EvidencePayload


class MqttIngest:
    def __init__(self, config: MqttConfig, node_ids: list[str]) -> None:
        self.config = config
        self.node_ids = set(node_ids)
        self.queue: asyncio.Queue[IngestMessage] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if config.username:
            self._client.username_pw_set(config.username, config.password or "")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._client.connect(self.config.host, self.config.port, keepalive=self.config.keepalive_s)
        self._client.loop_start()

    async def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish_cmd(self, node: str, cmd_payload: dict[str, Any]) -> None:
        topic = f"{self.config.topic_prefix}/cmd/{node}"
        self._client.publish(topic, json.dumps(cmd_payload), qos=1)

    # ---- paho-mqtt callbacks (run on paho's network thread) --------------------

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties=None) -> None:  # noqa: ANN001, ARG002
        if reason_code != 0:
            log.error("mqtt.connect_failed", reason_code=str(reason_code))
            return
        prefix = self.config.topic_prefix
        for kind, qos in (("score", 1), ("health", 0), ("status", 1), ("evidence", 1)):
            client.subscribe(f"{prefix}/+/{kind}", qos=qos)
        log.info("mqtt.connected", host=self.config.host, port=self.config.port)

    def _on_disconnect(  # noqa: ANN001, ARG002
        self, client, userdata, disconnect_flags, reason_code, properties=None
    ) -> None:
        log.warning("mqtt.disconnected", reason_code=str(reason_code))

    def _on_message(self, client, userdata, msg) -> None:  # noqa: ANN001, ARG002
        try:
            node, kind = msg.topic.split("/")[-2], msg.topic.split("/")[-1]
            body = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
            log.warning("mqtt.malformed_message", topic=msg.topic, error=str(exc))
            return

        model_cls = _PAYLOAD_MODELS.get(kind)
        if model_cls is None:
            return  # not one of our topics (or a cmd echo) — ignore

        try:
            payload = model_cls.model_validate(body)
        except ValidationError as exc:
            log.warning("mqtt.invalid_payload", topic=msg.topic, node=node, error=str(exc))
            return

        message = IngestMessage(kind=kind, node=node, payload=payload)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, message)
