"""Pydantic schemas for every MQTT payload, per comprehensive-plan.md §2.2(b).

These are the wire contract with firmware. Validation is intentionally strict —
a malformed or out-of-range payload is a bug worth surfacing loudly (see
ingest.py), not something to coerce and hope for the best on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

NodeStatus = Literal["online", "offline"]
EvidenceTrigger = Literal["fast_path_threshold", "manual", "cmd_capture"]
CmdName = Literal["set_threshold", "capture", "ota", "reboot"]


class ScorePayload(BaseModel):
    """echo-sentry/v1/<node>/score — published every detector.score_interval_s."""

    v: int = 1
    node: str
    t: int  # unix seconds, SNTP-synced on-device
    seq: int  # monotonic per-node counter; gaps reveal drops timestamps would hide
    err_mean: float
    err_p95: float
    err_max: float
    rms_dbfs: float
    bands: list[int] = Field(default_factory=list)  # coarse spectral summary, diagnostics only
    drops: int = 0  # cumulative I2S drop counter
    model: str  # e.g. "ae-2026w37-a1b2c3" — training run identifier
    cfg: str  # frontend.yaml config hash; must match the hub's or the score is quarantined
    fw: str  # firmware version

    @field_validator("err_mean", "err_p95", "err_max")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("reconstruction error cannot be negative")
        return v


class HealthPayload(BaseModel):
    """echo-sentry/v1/<node>/health — retained, published every 60s."""

    v: int = 1
    node: str
    t: int
    heap_free_bytes: int
    heap_min_free_bytes: int
    rssi_dbm: int
    uptime_s: int
    i2s_drops: int
    clock_skew_ms: int = 0  # vs. the Pi's chrony NTP server


class StatusPayload(BaseModel):
    """echo-sentry/v1/<node>/status — retained; also the MQTT last-will payload."""

    v: int = 1
    node: str
    t: int
    status: NodeStatus


class EvidencePayload(BaseModel):
    """echo-sentry/v1/<node>/evidence — announces a clip uploaded over HTTP to evidence.py."""

    v: int = 1
    node: str
    t: int
    trigger: EvidenceTrigger
    clip_id: str  # correlates with the multipart upload's filename
    bytes: int
    pre_roll_s: float = 15.0
    post_roll_s: float = 15.0


class SetThresholdCmd(BaseModel):
    cmd: Literal["set_threshold"] = "set_threshold"
    threshold: float


class CaptureCmd(BaseModel):
    cmd: Literal["capture"] = "capture"
    duration_s: float = 30.0


class OtaCmd(BaseModel):
    cmd: Literal["ota"] = "ota"
    url: str


class RebootCmd(BaseModel):
    cmd: Literal["reboot"] = "reboot"
