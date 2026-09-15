#!/usr/bin/env python3
"""Review unlabeled evidence clips and assign a label — comprehensive-plan.md §3.4.

Usage:
    uv run --project hub python tools/label.py --db /var/lib/echo-sentry/hub.db

For each unlabeled clip: prints its metadata, plays it (macOS `afplay`; elsewhere the
path is printed for manual playback), then asks for a label. Labels feed the evidence
flywheel: `normal` extends a node's confirmed-normal baseline, `nuisance` grows the
rejection-tuning set (comprehensive-plan.md §3.3), and `suspicious` / `confirmed-fault`
are the closest thing to a real in-situ ROC curve this pilot can produce without
seeding faults.
"""

from __future__ import annotations

import argparse
import platform
import subprocess

from echo_sentry_hub.store import Store

LABELS = ["normal", "nuisance", "suspicious", "confirmed-fault", "skip"]


def play(path: str) -> None:
    if platform.system() == "Darwin":
        subprocess.run(["afplay", path], check=False)
    else:
        print(f"  (playback not automated on this platform — open manually: {path})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    store = Store(args.db)
    store.connect()

    clips = store.unlabeled_evidence(limit=args.limit)
    if not clips:
        print("No unlabeled evidence clips.")
        store.close()
        return

    print(f"{len(clips)} unlabeled clip(s).\n")
    for row in clips:
        print(f"--- evidence #{row['id']} | node={row['node']} | trigger={row['trigger']} "
              f"| {row['bytes']} bytes | {row['path']}")
        if not args.no_play:
            play(row["path"])
        choice = input(f"  label [{'/'.join(LABELS)}] (default: skip): ").strip().lower()
        if not choice or choice == "skip":
            continue
        if choice not in LABELS:
            print(f"  unrecognized label {choice!r}, skipping")
            continue
        store.set_evidence_label(row["id"], choice)
        print(f"  -> labeled {choice}")

    store.close()


if __name__ == "__main__":
    main()
