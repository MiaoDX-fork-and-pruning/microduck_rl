#!/usr/bin/env python3
"""Train one shared gated adapter on base traces plus a student-state shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.generalist_schema import validate_batch

import train_generalist_bc as bc


def _base_trajectory_ids(trace_root: Path) -> tuple[np.ndarray, dict]:
    sources = (("stand", "velstand_flat"), ("locomotion", "velocity_flat"), ("sit_stand", "sitstand_flat"))
    ids: list[np.ndarray] = []
    source_rows: list[str] = []
    next_id = 0
    for _, name in sources:
        for path in sorted((trace_root / name).glob("*.npz")):
            length = len(np.load(path, allow_pickle=False)["observation"])
            ids.append(np.full(length, next_id, dtype=np.int64))
            source_rows.append(str(path))
            next_id += 1
    if not ids:
        raise FileNotFoundError(f"no base traces under {trace_root}")
    return np.concatenate(ids), {"base_trajectory_sources": source_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--student-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    x, y, manifest = bc.collect(args.trace_root)
    base_ids, source_manifest = _base_trajectory_ids(args.trace_root)
    if len(base_ids) != len(x):
        raise ValueError(f"base trajectory/sample mismatch: {len(base_ids)} != {len(x)}")
    with np.load(args.student_states, allow_pickle=False) as shard:
        extra_x = np.asarray(shard["inputs"], dtype=np.float32)
        extra_y = np.asarray(shard["actions"], dtype=np.float32)
        segment_ids = np.asarray(shard["segment_ids"]).astype(str)
    validate_batch(extra_x, extra_y)
    unique_segments, extra_ids = np.unique(segment_ids, return_inverse=True)
    extra_ids = extra_ids.astype(np.int64) + int(base_ids.max()) + 1
    x = np.concatenate((x, extra_x))
    y = np.concatenate((y, extra_y))
    trajectory_ids = np.concatenate((base_ids, extra_ids))
    metrics = bc.train(
        x, y, args.output, args.epochs, args.seed,
        balance=True, bounded=True, gated_adapter=True,
        trajectory_ids=trajectory_ids,
    )
    np.savez_compressed(args.output / "dataset.npz", inputs=x, actions=y,
                        trajectory_ids=trajectory_ids)
    manifest.update({
        "samples": int(len(x)),
        "input_dim": 71,
        "action_dim": 14,
        "seed": args.seed,
        "extra_data": str(args.student_states),
        "student_state_segments": [str(item) for item in unique_segments],
        **source_manifest,
        "metrics": metrics,
    })
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
