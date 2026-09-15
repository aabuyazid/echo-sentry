"""Hub configuration.

Layered, lowest to highest precedence: defaults -> /etc/echo-sentry/config.toml (or
$ECHO_SENTRY_CONFIG_FILE) -> environment variables (ECHO_SENTRY_*, "__" for nesting,
e.g. ECHO_SENTRY_DETECTOR__EWMA_ALPHA=0.03) -> constructor kwargs (used by tests).

See comprehensive-plan.md §2.4 and §5 for what each of these controls.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_CONFIG_FILE = Path("/etc/echo-sentry/config.toml")


class MqttConfig(BaseModel):
    host: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    topic_prefix: str = "echo-sentry/v1"
    keepalive_s: int = 30


class DetectorConfig(BaseModel):
    """Tuning constants for the six-stage pipeline in comprehensive-plan.md §5.

    The defaults here are the plan's starting values, not the tuned ones — run
    tools/replay.py against two weeks of live history before trusting them in
    production (comprehensive-plan.md §5, "Tuning procedure").
    """

    # Stage 1 — robust baseline calibration
    baseline_window_days: int = 14
    baseline_recalibrate_hour_utc: int = 2  # nightly, low-traffic hour

    # Stage 2 — rolling median filter (in samples, at the node's score cadence)
    median_filter_window: int = 5

    # Stage 3 — cross-node common-mode rejection
    min_nodes_for_common_mode: int = 3

    # Stage 4 — EWMA smoothing
    ewma_alpha: float = 0.02

    # Stage 5 — hysteresis state machine
    watch_enter: float = 3.0
    alarm_enter: float = 6.0
    alarm_enter_n: int = 5  # N-of-M persistence
    alarm_enter_m: int = 10
    alarm_exit: float = 4.0
    watch_exit: float = 2.5
    watch_exit_sustain_s: int = 1800  # 30 min

    # Stage 6 — CUSUM for slow degradation
    cusum_k: float = 0.5
    cusum_h: float = 5.0

    # Dead-man detection
    score_interval_s: int = 10
    offline_after_missed: int = 3  # consecutive missed intervals -> node_offline

    # Machine-off suppression: below this rms_dbfs, don't score at all.
    # None disables suppression until calibrate.py has learned a per-node floor.
    machine_off_rms_dbfs: float | None = None


class AlertingConfig(BaseModel):
    cooldown_s: int = 4 * 3600
    degradation_digest_only: bool = True  # CUSUM condition never pages, only digests


class TeamsSinkConfig(BaseModel):
    workflow_url: str | None = None  # Power Automate Workflow HTTP-trigger URL


class PowerBiSinkConfig(BaseModel):
    push_url: str | None = None  # streaming push dataset REST endpoint (includes key)


class EvidenceConfig(BaseModel):
    listen_host: str = "0.0.0.0"  # noqa: S104 - intentional: node-network-only listener
    listen_port: int = 8765
    storage_dir: Path = Path("/var/lib/echo-sentry/evidence")
    max_upload_bytes: int = 4 * 1024 * 1024  # ~30s of 16kHz/16-bit mono is under 1MB


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ECHO_SENTRY_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    node_ids: list[str] = Field(default_factory=lambda: [f"es-0{i}" for i in range(1, 6)])

    db_path: Path = Path("/var/lib/echo-sentry/hub.db")
    log_level: str = "INFO"
    dev_sink: bool = False  # when true, also log every event to the console sink

    # Set from the same config/frontend.yaml hash firmware and training embed
    # (tools/gen_config.py's ECHO_SENTRY_CONFIG_HASH / CONFIG_HASH). A score whose
    # `cfg` field disagrees is quarantined rather than scored — comprehensive-plan.md
    # §5, "Model/config guard". None (the default) disables the check for local dev
    # against tools/fake_node.py, which doesn't know the real hash.
    expected_config_hash: str | None = None

    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    teams: TeamsSinkConfig = Field(default_factory=TeamsSinkConfig)
    powerbi: PowerBiSinkConfig = Field(default_factory=PowerBiSinkConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_path = Path(os.environ.get("ECHO_SENTRY_CONFIG_FILE", str(DEFAULT_CONFIG_FILE)))
        return (
            init_settings,
            env_settings,
            _TomlConfigSource(settings_cls, toml_path),
            dotenv_settings,
            file_secret_settings,
        )


class _TomlConfigSource(PydanticBaseSettingsSource):
    """Reads config from a TOML file; absent file (the common dev case) yields {}."""

    def __init__(self, settings_cls: type[BaseSettings], toml_path: Path) -> None:
        super().__init__(settings_cls)
        self._toml_path = toml_path

    def get_field_value(self, field, field_name):  # noqa: ANN001, ANN201 - base class shape
        raise NotImplementedError  # unused; we override __call__ directly

    def __call__(self) -> dict:
        if not self._toml_path.exists():
            return {}
        with self._toml_path.open("rb") as f:
            return tomllib.load(f)
