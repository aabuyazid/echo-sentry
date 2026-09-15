from __future__ import annotations

import pytest
from echo_sentry_hub.models import ScorePayload, StatusPayload
from pydantic import ValidationError


def test_score_payload_parses_valid_json():
    payload = ScorePayload.model_validate({
        "v": 1, "node": "es-03", "t": 1789435200, "seq": 40213,
        "err_mean": 0.0412, "err_p95": 0.0981, "err_max": 0.1533,
        "rms_dbfs": -38.2, "bands": [1, 2, 3], "drops": 0,
        "model": "ae-2026w37-a1b2c3", "cfg": "f7e9d2", "fw": "1.2.0",
    })
    assert payload.node == "es-03"
    assert payload.seq == 40213


def test_score_payload_rejects_negative_error():
    with pytest.raises(ValidationError):
        ScorePayload.model_validate({
            "node": "es-01", "t": 1, "seq": 1, "err_mean": -0.1, "err_p95": 0.1,
            "err_max": 0.1, "rms_dbfs": -40.0, "model": "m", "cfg": "c", "fw": "1.0",
        })


def test_score_payload_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        ScorePayload.model_validate({"node": "es-01", "t": 1, "seq": 1})


def test_status_payload_rejects_invalid_status_literal():
    with pytest.raises(ValidationError):
        StatusPayload.model_validate({"node": "es-01", "t": 1, "status": "sleeping"})
