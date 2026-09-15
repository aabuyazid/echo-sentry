"""Training entrypoint — comprehensive-plan.md §4.3.

One model per node (five machines with different acoustics shouldn't share one
autoencoder — the DCASE "section" concept exists for exactly this reason). Normal
data only; the model never sees an anomaly.

Usage:
    uv run --project training python -m echo_sentry_train.train \\
        --manifest manifest.csv --node es-01 --arch dense_ae --out models/es-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from echo_sentry_train import frontend_config as fc
from echo_sentry_train.dataset import fit_normalization, load_split
from echo_sentry_train.model import ARCHS, PcaBaseline, set_seed

ALL_ARCHS = {**ARCHS, "pca": PcaBaseline}


def train_node(manifest_path: Path, node: str, arch: str, out_dir: Path, *,
                epochs: int = 100, batch_size: int = 512, seed: int = 42) -> dict:
    set_seed(seed)
    x_train = load_split(manifest_path, node, "train")
    x_val = load_split(manifest_path, node, "val")
    if len(x_train) == 0:
        raise ValueError(f"no training vectors for node {node!r} — check the manifest")

    norm = fit_normalization(x_train)
    x_train_n = norm.apply(x_train)
    x_val_n = norm.apply(x_val) if len(x_val) else None

    scorer = ALL_ARCHS[arch]()
    if arch == "pca":
        scorer.fit(x_train_n)
    else:
        scorer.fit(x_train_n, x_val=x_val_n, epochs=epochs, batch_size=batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(scorer, "model"):
        scorer.model.save(out_dir / "model.keras")
    (out_dir / "normalization.json").write_text(json.dumps(norm.to_dict()))

    metadata = {
        "node": node, "arch": arch, "config_hash": fc.CONFIG_HASH,
        "train_vectors": int(len(x_train)), "val_vectors": int(len(x_val)),
        "vector_dim": fc.VECTOR_DIM, "seed": seed,
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--arch", default="dense_ae", choices=list(ALL_ARCHS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = train_node(args.manifest, args.node, args.arch, args.out,
                           epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
