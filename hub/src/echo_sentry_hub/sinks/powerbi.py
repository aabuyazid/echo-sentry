"""Power BI sink — streaming push dataset REST API.

A view, not storage: push datasets cap out around 10k rows/hr and 75 columns
(comprehensive-plan.md §6). SQLite remains the system of record; this sink pushes a
compact per-event row so the dashboard stays live without ever being asked to hold
history. The push URL already embeds the dataset key, so there's no separate secret
to manage here — config/env-supplied by convention (see comprehensive-plan.md §2.4,
"Open items").
"""

from __future__ import annotations

import httpx

from echo_sentry_hub.config import PowerBiSinkConfig


class PowerBiSink:
    name = "powerbi"

    def __init__(self, config: PowerBiSinkConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def send(self, payload: dict) -> None:
        if not self.config.push_url:
            raise RuntimeError("powerbi.push_url is not configured")
        row = build_row(payload)
        resp = await self._client.post(self.config.push_url, json={"rows": [row]})
        resp.raise_for_status()


def build_row(payload: dict) -> dict:
    return {
        "Timestamp": payload.get("ts"),
        "Node": payload["node"],
        "Condition": payload["condition"],
        "Kind": payload.get("kind", ""),
        "Severity": payload.get("severity", ""),
        "PeakScore": payload.get("peak", 0.0),
    }
