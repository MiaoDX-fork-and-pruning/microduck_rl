#!/usr/bin/env python3
"""Train a single shared G0 actor on exact canonical teacher rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_generalist_bc as bc
from mjlab_microduck.generalist_schema import validate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gated-adapter", action="store_true")
    parser.add_argument("--film", action="store_true")
    parser.add_argument("--unbounded", action="store_true",
                        help="leave the shared action head unbounded for diagnostics")
    parser.add_argument("--critical-ticks", type=int, default=0,
                        help="oversample the first N ticks of every canonical segment")
    args = parser.parse_args()
    if args.gated_adapter == args.film:
        raise SystemExit("choose exactly one of --gated-adapter or --film")
    if args.critical_ticks < 0:
        raise SystemExit("--critical-ticks must be non-negative")
    with np.load(args.data, allow_pickle=False) as payload:
        x = np.asarray(payload["inputs"], dtype=np.float32)
        y = np.asarray(payload["actions"], dtype=np.float32)
        trajectory_ids = np.asarray(payload.get("trajectory_ids", np.arange(len(x))), dtype=np.int64)
        segment_ids = np.asarray(payload.get("segment_ids", np.arange(len(x))))
    validate_batch(x, y)
    if trajectory_ids.shape != (len(x),) or segment_ids.shape != (len(x),):
        raise ValueError("canonical data metadata must align with samples")
    bucket_labels = None
    if args.critical_ticks:
        offsets = np.zeros(len(x), dtype=np.int64)
        for trajectory in np.unique(trajectory_ids):
            selected = np.flatnonzero(trajectory_ids == trajectory)
            offsets[selected] = np.arange(len(selected))
        bucket_labels = (offsets < args.critical_ticks).astype(np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "dataset.npz", inputs=x, actions=y,
                        trajectory_ids=trajectory_ids, segment_ids=segment_ids)
    metrics = bc.train(
        x, y, args.output, args.epochs, args.seed,
        balance=True, bounded=not args.unbounded, gated_adapter=args.gated_adapter,
        film=args.film, trajectory_ids=trajectory_ids, bucket_labels=bucket_labels,
    )
    manifest = {
        "schema": "generalist-v0-canonical-teacher",
        "schema_version": 1,
        "data": str(args.data),
        "samples": int(len(x)),
        "trajectories": int(len(np.unique(trajectory_ids))),
        "segments": sorted({str(item) for item in segment_ids}),
        "critical_ticks": args.critical_ticks,
        "metrics": metrics,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
