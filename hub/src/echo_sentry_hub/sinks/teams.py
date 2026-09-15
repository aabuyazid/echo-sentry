"""Microsoft Teams sink — Power Automate Workflow HTTP trigger, Adaptive Card payload.

Classic O365 Incoming Webhook connectors completed retirement in 2026 — see
comprehensive-plan.md §6. Set up: create a Power Automate flow with an "When a HTTP
request is received" trigger that posts the received JSON into a Teams channel via
"Post adaptive card in a chat or channel", and put that trigger URL in
teams.workflow_url (config.py) / ECHO_SENTRY_TEAMS__WORKFLOW_URL.

Verdict buttons ("Confirm" / "False alarm") require a publicly reachable endpoint for
Teams to call back into — the hub itself is not internet-reachable (outbound HTTPS
only, comprehensive-plan.md §2.1). If `verdict_webhook_base_url` is configured (e.g. a
small Azure Function or Power Automate "Respond to a PowerApp or flow" bridge you've
stood up separately), the card includes Action.Http buttons that post to it. If not,
the card is informational only and operators record verdicts with tools/label.py
against the evidence clip instead — functionally equivalent for the evidence flywheel
(comprehensive-plan.md §3.4), just not from inside Teams.
"""

from __future__ import annotations

import httpx

from echo_sentry_hub.config import TeamsSinkConfig

_SEVERITY_COLOR = {"alarm": "Attention", "digest": "Warning", "resolved": "Good"}


class TeamsSink:
    name = "teams"

    def __init__(self, config: TeamsSinkConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def send(self, payload: dict) -> None:
        if not self.config.workflow_url:
            raise RuntimeError("teams.workflow_url is not configured")
        card = build_adaptive_card(payload, verdict_webhook_base_url=None)
        resp = await self._client.post(self.config.workflow_url, json=card)
        resp.raise_for_status()


def build_adaptive_card(payload: dict, *, verdict_webhook_base_url: str | None) -> dict:
    kind = payload.get("kind", "opened")
    node = payload["node"]
    condition = payload["condition"]
    resolved_kinds = ("closed", "digest_closed")
    severity = "resolved" if kind in resolved_kinds else payload.get("severity", "alarm")
    title = {
        "opened": f"⚠️ Anomaly — {node}",
        "updated": f"⚠️ Anomaly continuing — {node}",
        "closed": f"✅ Resolved — {node}",
        "digest_queued": f"📈 Degradation trend flagged — {node}",
        "digest_closed": f"📉 Degradation trend cleared — {node}",
    }.get(kind, f"Echo Sentry — {node}")

    facts = [
        {"title": "Node", "value": node},
        {"title": "Condition", "value": condition},
        {"title": "Peak score (z)", "value": f"{payload.get('peak', 0.0):.2f}"},
        {"title": "Time (UTC)", "value": str(payload.get("ts"))},
    ]
    if payload.get("evidence_url"):
        facts.append({"title": "Evidence clip", "value": payload["evidence_url"]})

    body: list[dict] = [
        {"type": "TextBlock", "text": title, "weight": "Bolder", "size": "Medium",
         "color": _SEVERITY_COLOR.get(severity, "Default"), "wrap": True},
        {"type": "FactSet", "facts": facts},
    ]

    actions = []
    if verdict_webhook_base_url and kind in ("opened", "updated"):
        for verdict, label in (("confirmed", "Confirm"), ("false_alarm", "False alarm")):
            actions.append({
                "type": "Action.Http",
                "title": label,
                "method": "POST",
                "url": f"{verdict_webhook_base_url}/alerts/{payload['alert_id']}/verdict",
                "body": f'{{"verdict": "{verdict}"}}',
                "headers": [{"name": "Content-Type", "value": "application/json"}],
            })

    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
    }
    if actions:
        card["actions"] = actions

    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": card,
        }],
    }
