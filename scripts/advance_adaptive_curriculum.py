#!/usr/bin/env python3
"""Consume a frozen capability battery and persist the next adaptive stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjlab_microduck.tasks.adaptive_curriculum import (
    ADAPTIVE_AXIS_CONFIGS,
    CapabilityGate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage-file", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    battery = json.loads(args.battery.read_text(encoding="utf-8"))
    metrics = battery.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("battery report is missing metrics")

    gate = CapabilityGate(
        tuple(ADAPTIVE_AXIS_CONFIGS),
        critical_buckets=("zero", "forward", "lateral", "yaw", "turn-left", "turn-right"),
        ema_alpha=0.25,
        preservation_tolerance=0.05,
    )
    if args.state.exists():
        gate.load_state_dict(json.loads(args.state.read_text(encoding="utf-8")))
    transition = gate.update(args.step, {str(k): float(v) for k, v in metrics.items()}, checkpoint=args.checkpoint, seed=args.seed)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(gate.state_dict(), indent=2) + "\n", encoding="utf-8")
    stages = {axis: gate.stage_value(axis) for axis in gate.axis_order}
    args.stage_file.parent.mkdir(parents=True, exist_ok=True)
    args.stage_file.write_text(json.dumps({"stages": stages}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"transition": None if transition is None else transition.as_dict(), "stages": stages}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
