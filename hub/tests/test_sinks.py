from __future__ import annotations

from echo_sentry_hub.sinks.powerbi import build_row
from echo_sentry_hub.sinks.teams import build_adaptive_card


def test_build_adaptive_card_opened_has_no_actions_without_webhook():
    payload = {"kind": "opened", "alert_id": 1, "node": "es-01", "condition": "anomaly",
               "severity": "alarm", "ts": 1000, "peak": 7.2}
    card = build_adaptive_card(payload, verdict_webhook_base_url=None)
    content = card["attachments"][0]["content"]
    assert "actions" not in content
    assert any("es-01" in block.get("text", "") for block in content["body"]
               if block["type"] == "TextBlock")


def test_build_adaptive_card_opened_includes_verdict_buttons_when_configured():
    payload = {"kind": "opened", "alert_id": 42, "node": "es-01", "condition": "anomaly",
               "severity": "alarm", "ts": 1000, "peak": 7.2}
    card = build_adaptive_card(payload, verdict_webhook_base_url="https://example.invalid")
    content = card["attachments"][0]["content"]
    assert len(content["actions"]) == 2
    assert all("42" in a["url"] for a in content["actions"])


def test_build_adaptive_card_closed_has_no_actions():
    payload = {"kind": "closed", "alert_id": 1, "node": "es-01", "condition": "anomaly",
               "severity": "alarm", "ts": 1000, "peak": 7.2}
    card = build_adaptive_card(payload, verdict_webhook_base_url="https://example.invalid")
    content = card["attachments"][0]["content"]
    assert "actions" not in content


def test_powerbi_row_shape():
    payload = {"kind": "opened", "node": "es-01", "condition": "anomaly", "severity": "alarm",
               "ts": 1000, "peak": 7.2}
    row = build_row(payload)
    assert row["Node"] == "es-01"
    assert row["PeakScore"] == 7.2
