#!/usr/bin/env python3
"""Compare native VELSTAND labels with FrozenG0Teachers reconstruction.

The input NPZ is intentionally simple: ``observations`` is [N,61] and
``native_actions`` is [N,14], captured from the immutable specialist battery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mjlab_microduck.generalist_schema import make_conditioned_observation
from mjlab_microduck.generalist_teachers import FrozenG0Teachers, compare_action_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/specialists"))
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    native = np.asarray(data["native_actions"], dtype=np.float32)
    command = np.asarray(data["command"], dtype=np.float32) if "command" in data else np.zeros((len(observations), 13), np.float32)
    conditioned = make_conditioned_observation(observations, command, "stand")
    teachers = FrozenG0Teachers(args.artifact_root)
    with torch.no_grad():
        reconstructed = teachers(torch.from_numpy(conditioned))
    report = compare_action_batches(torch.from_numpy(native), reconstructed)
    report.update({"behavior": "VELSTAND", "observation_dim": 71, "action_dim": 14})
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
