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
from mjlab_microduck.evaluation.capability import resolve_enabled_axes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stage-file", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--axis-mode", choices=("all_static", "com", "head_com", "composed"), default="composed")
    args = parser.parse_args()

    battery = json.loads(args.battery.read_text(encoding="utf-8"))
    metrics = battery.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("battery report is missing metrics")

    enabled_axes = resolve_enabled_axes(args.axis_mode)
    if not enabled_axes:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "axis_mode": args.axis_mode,
            "enabled_axes": [],
            "states": {},
            "best_metrics": {},
            "trace": [],
        }
        args.state.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        args.stage_file.parent.mkdir(parents=True, exist_ok=True)
        args.stage_file.write_text(json.dumps({"axis_mode": args.axis_mode, "stages": {}}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"transition": None, "stages": {}}, indent=2))
        return 0
    gate = CapabilityGate(
        tuple(axis for axis in ADAPTIVE_AXIS_CONFIGS if axis.name in enabled_axes),
        critical_buckets=("zero", "forward", "lateral", "yaw", "turn-left", "turn-right"),
        ema_alpha=0.25,
        preservation_tolerance=0.05,
        axis_mode=args.axis_mode,
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
