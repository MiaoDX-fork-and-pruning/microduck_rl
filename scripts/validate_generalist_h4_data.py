#!/usr/bin/env python3
"""Validate canonical H4 data and frozen teacher identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.generalist_schema import validate_batch
from mjlab_microduck.generalist_temporal import build_windows, validate_batch as validate_h4_batch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(data_path: Path, teacher_manifest_path: Path) -> dict:
    manifest = json.loads(teacher_manifest_path.read_text())
    teachers = {}
    for item in manifest["teachers"]:
        path = Path(item["artifacts"]["onnx"])
        actual = sha256(path)
        expected = item["sha256"]["onnx"]
        teachers[item["id"]] = {"path": str(path), "sha256": actual, "passed": actual == expected}
    with np.load(data_path, allow_pickle=False) as payload:
        x = np.asarray(payload["inputs"], dtype=np.float32)
        y = np.asarray(payload["actions"], dtype=np.float32)
        trajectories = np.asarray(payload["trajectory_ids"])
        segments = np.asarray(payload["segment_ids"])
    validate_batch(x, y)
    if trajectories.shape != (len(x),) or segments.shape != (len(x),):
        raise ValueError("trajectory and segment metadata must align with samples")
    h4 = build_windows(x, segments)
    validate_h4_batch(h4, y)
    if not np.array_equal(h4, build_windows(x, segments)):
        raise ValueError("H4 replay is not deterministic")
    labels = x[:, 48:54].argmax(axis=1)
    names = ("stand", "locomotion", "sit_stand")
    counts = {name: int(np.sum(labels == index)) for index, name in enumerate(names)}
    passed = len(np.unique(trajectories)) == 7 and all(counts.values()) and all(t["passed"] for t in teachers.values())
    return {
        "schema": "generalist-g0-h4-data-validation", "version": 1, "passed": bool(passed),
        "data": str(data_path), "data_sha256": sha256(data_path), "samples": len(x),
        "trajectories": int(len(np.unique(trajectories))), "segments": int(len(np.unique(segments))),
        "input_dim": int(h4.shape[1]), "action_dim": int(y.shape[1]),
        "behavior_samples": counts, "finite": bool(np.isfinite(h4).all() and np.isfinite(y).all()),
        "action_outside_unit_fraction": float(np.mean(np.abs(y) > 1.0)), "teachers": teachers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, default=Path("docs/plans/generalist-g0-teacher-manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.data, args.teacher_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
