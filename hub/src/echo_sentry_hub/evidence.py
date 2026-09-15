"""HTTP evidence intake — comprehensive-plan.md §2.4 / §3.4.

Nodes POST an evidence clip (WAV) here after a fast-path threshold excursion (the
"evidence flywheel", comprehensive-plan.md §3.4). This is the one path where PCM
crosses the node -> Pi boundary in production, and it stops here: nothing forwards
audio any further — Teams and Power BI only ever see an evidence *reference*
(comprehensive-plan.md §2.1, "the room boundary is the privacy boundary").
"""

from __future__ import annotations

import time
import uuid

import structlog
from aiohttp import web

from echo_sentry_hub.config import EvidenceConfig
from echo_sentry_hub.store import Store

log = structlog.get_logger(__name__)

_MAX_TRIGGER_LEN = 64
_MAX_CLIP_ID_LEN = 32


def build_app(store: Store, config: EvidenceConfig) -> web.Application:
    app = web.Application(client_max_size=config.max_upload_bytes)
    app["store"] = store
    app["config"] = config
    app.router.add_post("/evidence/{node}", handle_upload)
    app.router.add_get("/healthz", handle_healthz)
    return app


async def handle_healthz(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_upload(request: web.Request) -> web.Response:
    store: Store = request.app["store"]
    config: EvidenceConfig = request.app["config"]
    node = request.match_info["node"]

    reader = await request.multipart()
    wav_bytes: bytes | None = None
    trigger = "unknown"
    clip_id = uuid.uuid4().hex[:12]

    field = await reader.next()
    while field is not None:
        if field.name == "clip":
            wav_bytes = await field.read(decode=False)
        elif field.name == "trigger":
            trigger = (await field.read(decode=False)).decode("utf-8", "replace")[:_MAX_TRIGGER_LEN]
        elif field.name == "clip_id":
            clip_id = (await field.read(decode=False)).decode("utf-8", "replace")[:_MAX_CLIP_ID_LEN]
        field = await reader.next()

    if wav_bytes is None:
        return web.json_response({"error": "missing 'clip' field"}, status=400)

    node_dir = config.storage_dir / node
    node_dir.mkdir(parents=True, exist_ok=True)
    path = node_dir / f"{clip_id}.wav"
    path.write_bytes(wav_bytes)

    now = int(time.time())
    evidence_id = store.insert_evidence(
        node=node, ts=now, path=str(path), num_bytes=len(wav_bytes), trigger=trigger,
    )

    open_alert = store.get_open_alert(node=node, condition="anomaly")
    if open_alert is not None:
        store.set_alert_evidence(open_alert.id, str(path))
        store.link_evidence_to_alert(evidence_id, open_alert.id)

    log.info("evidence.received", node=node, clip_id=clip_id, bytes=len(wav_bytes))
    return web.json_response({"evidence_id": evidence_id, "path": str(path)})
