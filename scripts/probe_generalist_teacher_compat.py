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
    observation_key = "observations" if "observations" in data else "observation"
    action_key = "native_actions" if "native_actions" in data else "raw_action"
    if observation_key not in data or action_key not in data:
        raise ValueError("input must contain observations/native_actions or canonical observation/raw_action")
    observations = np.asarray(data[observation_key], dtype=np.float32)
    native = np.asarray(data[action_key], dtype=np.float32)
    command = np.asarray(data["command" if "command" in data else "requested_command"], dtype=np.float32) if ("command" in data or "requested_command" in data) else np.zeros((len(observations), 13), np.float32)
    conditioned = make_conditioned_observation(observations, command, "stand")
    teachers = FrozenG0Teachers(args.artifact_root)
    with torch.no_grad():
        reconstructed = teachers(torch.from_numpy(conditioned))
    report = compare_action_batches(torch.from_numpy(native), reconstructed)
    report.update({"behavior": "VELSTAND", "observation_dim": 71, "action_dim": 14})
    report.update({"input": str(args.input), "samples": len(observations),
                   "source_observation_key": observation_key,
                   "source_action_key": action_key})
    output = args.input.with_name(args.input.stem + "-compatibility.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
