#!/usr/bin/env python3
"""Replay historical scores through a candidate detector config — comprehensive-plan.md
§5, "Tuning procedure".

Never mutates the source database (opened read-only). Every stored score's err_mean is
replayed, in timestamp order, through a *fresh* in-memory Store + Detector using the
given config overrides — this is what lets you grid-search (ewma_alpha, median window,
thresholds, N-of-M, cusum k/h) against real history in seconds instead of guessing.

Usage:
    uv run --project hub python tools/replay.py --db /path/to/hub.db
    uv run --project hub python tools/replay.py --db hub.db --ewma-alpha 0.03 --alarm-enter 5.0
    uv run --project hub python tools/replay.py --db hub.db --node es-01 --node es-02
"""

from __future__ import annotations

import argparse
import sqlite3

from echo_sentry_hub.alerting import AlertingConfig, AlertManager
from echo_sentry_hub.config import DetectorConfig
from echo_sentry_hub.detector import Detector
from echo_sentry_hub.store import Store


def load_source_rows(db_path: str, node_filter: list[str] | None) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT node, ts, seq, err_mean FROM score WHERE quarantined = 0 ORDER BY ts ASC"
    ).fetchall()
    conn.close()
    if node_filter:
        rows = [r for r in rows if r["node"] in node_filter]
    return rows


def replay(rows: list[sqlite3.Row], detector_config: DetectorConfig,
           alerting_config: AlertingConfig, node_ids: list[str],
           recalibrate_every_s: int = 86400) -> dict:
    store = Store(":memory:")
    store.connect()
    detector = Detector(store, detector_config, node_ids)
    alerting = AlertManager(store, alerting_config)

    last_recal: dict[str, int] = {}
    events: list[tuple[str, int, str, str]] = []

    for row in rows:
        node, ts, seq, err_mean = row["node"], row["ts"], row["seq"], row["err_mean"]
        store.insert_score(
            node=node, ts=ts, seq=seq, err_mean=err_mean, err_p95=err_mean, err_max=err_mean,
            rms_dbfs=-40.0, model="replay", cfg="replay", fw="replay", raw="{}",
        )
        if ts - last_recal.get(node, -recalibrate_every_s) >= recalibrate_every_s:
            if detector.recalibrate_baseline(node, ts):
                last_recal[node] = ts

        result = detector.process_score(node=node, err_mean=err_mean, ts=ts)
        score_id = store.conn.execute(
            "SELECT id FROM score WHERE node=? AND seq=?", (node, seq)
        ).fetchone()["id"]
        store.update_score_detector_outputs(
            score_id, z=result.z, z_corrected=result.z_corrected, z_ewma=result.ewma,
            cusum=result.cusum, state=result.state,
        )
        for event in alerting.handle_result(result):
            events.append((node, ts, event.kind, event.condition))

    store.close()

    alarms_opened = sum(1 for e in events if e[2] == "opened" and e[3] == "anomaly")
    digests_opened = sum(1 for e in events if e[2] == "digest_queued")
    return {"events": events, "alarms_opened": alarms_opened, "digests_opened": digests_opened,
            "scores_replayed": len(rows)}


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    """Adds one --<field-name-with-dashes> flag per DetectorConfig field."""
    for name, info in DetectorConfig.model_fields.items():
        flag = "--" + name.replace("_", "-")
        arg_type = info.annotation if info.annotation in (int, float, bool) else str
        parser.add_argument(flag, type=arg_type, default=None,
                             help=f"override DetectorConfig.{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="path to the hub's hub.db")
    parser.add_argument("--node", action="append", default=None,
                         help="restrict replay to this node (repeatable)")
    _add_config_overrides(parser)
    args = parser.parse_args()

    overrides = {
        name: getattr(args, name)
        for name in DetectorConfig.model_fields
        if getattr(args, name) is not None
    }
    detector_config = DetectorConfig(**overrides)

    rows = load_source_rows(args.db, args.node)
    if not rows:
        print("No scores found in the source database (with the given --node filter).")
        return

    node_ids = sorted({r["node"] for r in rows})
    result = replay(rows, detector_config, AlertingConfig(), node_ids)

    print(f"replayed {result['scores_replayed']} scores across {len(node_ids)} node(s)")
    print(f"  ALARM opens:            {result['alarms_opened']}")
    print(f"  degradation_trend opens: {result['digests_opened']}")
    if result["alarms_opened"] == 0:
        print("  -> zero alerts on this history: a good sign this config is safe to deploy,")
        print("     per comprehensive-plan.md §5's tuning procedure.")
    print()
    for node, ts, kind, condition in result["events"]:
        print(f"  {ts}  {node:8s}  {kind:15s}  {condition}")


if __name__ == "__main__":
    main()
