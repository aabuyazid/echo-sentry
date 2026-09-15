"""SQLite system of record — comprehensive-plan.md §2.4.

One connection, WAL mode, guarded by a lock so it's safe to drive from multiple
asyncio.to_thread() callers. At the hub's actual throughput (0.5 msg/s x 5 nodes) this
has enormous headroom; the lock exists for correctness, not performance.

Every score's detector outputs (z, z_corrected, z_ewma, cusum) are written inline
rather than recomputed on read — see the module docstring rationale in
comprehensive-plan.md §2.4: replaying and re-tuning threshold constants only makes
sense if you can see exactly what the detector saw the first time.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS score (
    id INTEGER PRIMARY KEY,
    node TEXT NOT NULL,
    ts INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    err_mean REAL NOT NULL,
    err_p95 REAL NOT NULL,
    err_max REAL NOT NULL,
    rms_dbfs REAL NOT NULL,
    z REAL,
    z_corrected REAL,
    z_ewma REAL,
    cusum REAL,
    state TEXT,
    model TEXT,
    cfg TEXT,
    fw TEXT,
    quarantined INTEGER NOT NULL DEFAULT 0,
    quarantine_reason TEXT,
    raw TEXT,
    UNIQUE(node, seq)
);
CREATE INDEX IF NOT EXISTS idx_score_node_ts ON score(node, ts);

CREATE TABLE IF NOT EXISTS detector_state (
    node TEXT PRIMARY KEY,
    baseline_median REAL,
    baseline_mad REAL,
    baseline_updated INTEGER,
    ewma REAL NOT NULL DEFAULT 0,
    cusum REAL NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'NORMAL',
    state_entered INTEGER,
    watch_exit_since INTEGER,
    median_buf TEXT NOT NULL DEFAULT '[]',
    hysteresis_buf TEXT NOT NULL DEFAULT '[]',
    last_score_ts INTEGER
);

CREATE TABLE IF NOT EXISTS alert (
    id INTEGER PRIMARY KEY,
    node TEXT NOT NULL,
    condition TEXT NOT NULL,
    severity TEXT NOT NULL,
    opened_ts INTEGER NOT NULL,
    closed_ts INTEGER,
    peak_z REAL,
    evidence_path TEXT,
    verdict TEXT,
    verdict_by TEXT,
    verdict_ts INTEGER
);
CREATE INDEX IF NOT EXISTS idx_alert_node_condition ON alert(node, condition, opened_ts);

CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY,
    sink TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_ts INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    sent_ts INTEGER
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox(sent_ts, created_ts);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    node TEXT NOT NULL,
    ts INTEGER NOT NULL,
    path TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    trigger TEXT,
    alert_id INTEGER,
    label TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_label ON evidence(label);
"""


@dataclass
class DetectorState:
    node: str
    baseline_median: float | None
    baseline_mad: float | None
    baseline_updated: int | None
    ewma: float
    cusum: float
    state: str
    state_entered: int | None
    watch_exit_since: int | None
    median_buf: list[float]
    hysteresis_buf: list[int]
    last_score_ts: int | None


@dataclass
class Alert:
    id: int
    node: str
    condition: str
    severity: str
    opened_ts: int
    closed_ts: int | None
    peak_z: float | None
    evidence_path: str | None
    verdict: str | None
    verdict_by: str | None
    verdict_ts: int | None


@dataclass
class OutboxRow:
    id: int
    sink: str
    payload: str
    created_ts: int
    attempts: int
    last_error: str | None


class Store:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store.connect() must be called first")
        return self._conn

    # ---- score ----------------------------------------------------------

    def insert_score(self, *, node: str, ts: int, seq: int, err_mean: float, err_p95: float,
                      err_max: float, rms_dbfs: float, model: str, cfg: str, fw: str,
                      raw: str, quarantined: bool = False,
                      quarantine_reason: str | None = None) -> int | None:
        """Returns the new row id, or None if (node, seq) was already stored (dedup)."""
        with self._lock:
            try:
                cur = self.conn.execute(
                    """INSERT INTO score
                       (node, ts, seq, err_mean, err_p95, err_max, rms_dbfs, model, cfg, fw,
                        quarantined, quarantine_reason, raw)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (node, ts, seq, err_mean, err_p95, err_max, rms_dbfs, model, cfg, fw,
                     int(quarantined), quarantine_reason, raw),
                )
                self.conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                return None

    def update_score_detector_outputs(self, score_id: int, *, z: float, z_corrected: float,
                                       z_ewma: float, cusum: float, state: str) -> None:
        with self._lock:
            self.conn.execute(
                """UPDATE score SET z=?, z_corrected=?, z_ewma=?, cusum=?, state=?
                   WHERE id=?""",
                (z, z_corrected, z_ewma, cusum, state, score_id),
            )
            self.conn.commit()

    def latest_scores_by_node(self, *, as_of_ts: int, max_age_s: int) -> dict[str, float]:
        """Most recent z_corrected (pre-common-mode 'z') per node within a freshness window.

        Used for cross-node common-mode rejection (comprehensive-plan.md §5 Stage 3) —
        deliberately returns the *uncorrected* z, since correcting with an already-corrected
        peer value would be circular.
        """
        with self._lock:
            rows = self.conn.execute(
                """SELECT node, z, ts FROM score
                   WHERE ts >= ? AND ts <= ? AND z IS NOT NULL AND quarantined = 0
                   ORDER BY ts DESC""",
                (as_of_ts - max_age_s, as_of_ts),
            ).fetchall()
        latest: dict[str, float] = {}
        for row in rows:
            latest.setdefault(row["node"], row["z"])
        return latest

    def baseline_sample(self, *, node: str, since_ts: int, until_ts: int) -> list[float]:
        """err_mean values usable for Stage-1 median/MAD calibration.

        Excludes any score inside an open alert window or explicitly quarantined, so a
        live fault doesn't poison its own baseline.
        """
        with self._lock:
            rows = self.conn.execute(
                """SELECT s.err_mean FROM score s
                   WHERE s.node = ? AND s.ts >= ? AND s.ts <= ? AND s.quarantined = 0
                     AND NOT EXISTS (
                       SELECT 1 FROM alert a
                       WHERE a.node = s.node AND a.opened_ts <= s.ts
                         AND (a.closed_ts IS NULL OR a.closed_ts >= s.ts)
                     )""",
                (node, since_ts, until_ts),
            ).fetchall()
        return [row["err_mean"] for row in rows]

    def scores_since(self, *, node: str, since_ts: int) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM score WHERE node = ? AND ts >= ? ORDER BY ts ASC",
                (node, since_ts),
            ).fetchall()

    # ---- detector_state ---------------------------------------------------

    def get_detector_state(self, node: str) -> DetectorState | None:
        import json

        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM detector_state WHERE node = ?", (node,)
            ).fetchone()
        if row is None:
            return None
        return DetectorState(
            node=row["node"],
            baseline_median=row["baseline_median"],
            baseline_mad=row["baseline_mad"],
            baseline_updated=row["baseline_updated"],
            ewma=row["ewma"],
            cusum=row["cusum"],
            state=row["state"],
            state_entered=row["state_entered"],
            watch_exit_since=row["watch_exit_since"],
            median_buf=json.loads(row["median_buf"]),
            hysteresis_buf=json.loads(row["hysteresis_buf"]),
            last_score_ts=row["last_score_ts"],
        )

    def save_detector_state(self, state: DetectorState) -> None:
        import json

        with self._lock:
            self.conn.execute(
                """INSERT INTO detector_state
                     (node, baseline_median, baseline_mad, baseline_updated, ewma, cusum,
                      state, state_entered, watch_exit_since, median_buf, hysteresis_buf,
                      last_score_ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(node) DO UPDATE SET
                     baseline_median=excluded.baseline_median,
                     baseline_mad=excluded.baseline_mad,
                     baseline_updated=excluded.baseline_updated,
                     ewma=excluded.ewma,
                     cusum=excluded.cusum,
                     state=excluded.state,
                     state_entered=excluded.state_entered,
                     watch_exit_since=excluded.watch_exit_since,
                     median_buf=excluded.median_buf,
                     hysteresis_buf=excluded.hysteresis_buf,
                     last_score_ts=excluded.last_score_ts""",
                (
                    state.node, state.baseline_median, state.baseline_mad,
                    state.baseline_updated, state.ewma, state.cusum, state.state,
                    state.state_entered, state.watch_exit_since,
                    json.dumps(state.median_buf), json.dumps(state.hysteresis_buf),
                    state.last_score_ts,
                ),
            )
            self.conn.commit()

    def all_detector_states(self) -> list[DetectorState]:
        with self._lock:
            nodes = [r["node"] for r in self.conn.execute("SELECT node FROM detector_state")]
        return [s for n in nodes if (s := self.get_detector_state(n)) is not None]

    # ---- alert --------------------------------------------------------------

    def open_alert(self, *, node: str, condition: str, severity: str, opened_ts: int,
                    peak_z: float) -> int:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO alert (node, condition, severity, opened_ts, peak_z)
                   VALUES (?, ?, ?, ?, ?)""",
                (node, condition, severity, opened_ts, peak_z),
            )
            self.conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_open_alert(self, *, node: str, condition: str) -> Alert | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT * FROM alert WHERE node=? AND condition=? AND closed_ts IS NULL
                   ORDER BY opened_ts DESC LIMIT 1""",
                (node, condition),
            ).fetchone()
        return self._row_to_alert(row) if row else None

    def most_recent_alert(self, *, node: str, condition: str) -> Alert | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT * FROM alert WHERE node=? AND condition=?
                   ORDER BY opened_ts DESC LIMIT 1""",
                (node, condition),
            ).fetchone()
        return self._row_to_alert(row) if row else None

    def bump_alert_peak(self, alert_id: int, z: float) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE alert SET peak_z = MAX(COALESCE(peak_z, ?), ?) WHERE id = ?",
                (z, z, alert_id),
            )
            self.conn.commit()

    def close_alert(self, alert_id: int, closed_ts: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE alert SET closed_ts = ? WHERE id = ?", (closed_ts, alert_id)
            )
            self.conn.commit()

    def set_alert_evidence(self, alert_id: int, evidence_path: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE alert SET evidence_path = ? WHERE id = ?", (evidence_path, alert_id)
            )
            self.conn.commit()

    def set_alert_verdict(self, alert_id: int, *, verdict: str, verdict_by: str,
                           verdict_ts: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE alert SET verdict=?, verdict_by=?, verdict_ts=? WHERE id=?",
                (verdict, verdict_by, verdict_ts, alert_id),
            )
            self.conn.commit()

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"], node=row["node"], condition=row["condition"],
            severity=row["severity"], opened_ts=row["opened_ts"], closed_ts=row["closed_ts"],
            peak_z=row["peak_z"], evidence_path=row["evidence_path"], verdict=row["verdict"],
            verdict_by=row["verdict_by"], verdict_ts=row["verdict_ts"],
        )

    # ---- outbox (store-and-forward) ------------------------------------------

    def enqueue_outbox(self, *, sink: str, payload: str, created_ts: int) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO outbox (sink, payload, created_ts) VALUES (?, ?, ?)",
                (sink, payload, created_ts),
            )
            self.conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def pending_outbox(self, *, limit: int = 50) -> list[OutboxRow]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT * FROM outbox WHERE sent_ts IS NULL
                   ORDER BY created_ts ASC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            OutboxRow(id=r["id"], sink=r["sink"], payload=r["payload"],
                      created_ts=r["created_ts"], attempts=r["attempts"],
                      last_error=r["last_error"])
            for r in rows
        ]

    def mark_outbox_sent(self, outbox_id: int, sent_ts: int) -> None:
        with self._lock:
            self.conn.execute("UPDATE outbox SET sent_ts = ? WHERE id = ?", (sent_ts, outbox_id))
            self.conn.commit()

    def mark_outbox_failed(self, outbox_id: int, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE id = ?",
                (error, outbox_id),
            )
            self.conn.commit()

    # ---- evidence -------------------------------------------------------------

    def insert_evidence(self, *, node: str, ts: int, path: str, num_bytes: int,
                         trigger: str | None, alert_id: int | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO evidence (node, ts, path, bytes, trigger, alert_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (node, ts, path, num_bytes, trigger, alert_id),
            )
            self.conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def unlabeled_evidence(self, *, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM evidence WHERE label IS NULL ORDER BY ts ASC LIMIT ?", (limit,)
            ).fetchall()

    def set_evidence_label(self, evidence_id: int, label: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE evidence SET label = ? WHERE id = ?", (label, evidence_id)
            )
            self.conn.commit()

    def link_evidence_to_alert(self, evidence_id: int, alert_id: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE evidence SET alert_id = ? WHERE id = ?", (alert_id, evidence_id)
            )
            self.conn.commit()


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"))


def iter_rows_as_dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]
