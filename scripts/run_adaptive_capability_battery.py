#!/usr/bin/env python3
"""Run the frozen six-bucket capability battery for an adaptive checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_specialist_action_battery import command_cases, run_case  # noqa: E402


def run_battery(onnx_path: Path, output: Path, seed: int, duration_seconds: float, smoke: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    cases = []
    for offset, case in enumerate(command_cases("adaptive_velocity", smoke)):
        report, trace = run_case(
            "adaptive_velocity", onnx_path, case, seed + offset, duration_seconds, smoke=smoke
        )
        report["bucket"] = case["bucket"]
        cases.append(report)
        metrics[case["bucket"]] = 1.0 if report["passed"] else 0.0
        trace_path = output / f"{case['bucket']}.npz"
        import numpy as np

        np.savez_compressed(trace_path, **trace)
        report["trace"] = str(trace_path)
    payload = {
        "schema_version": 1,
        "battery": "adaptive_velocity_six_bucket",
        "checkpoint": str(onnx_path),
        "seed": seed,
        "duration_seconds": duration_seconds,
        "metrics": metrics,
        "score": min(metrics.values()) if metrics else 0.0,
        "cases": cases,
    }
    (output / "capability.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    payload = run_battery(args.onnx, args.output, args.seed, args.duration_seconds, args.smoke)
    print(json.dumps(payload, indent=2))
    return 0 if payload["score"] >= 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
