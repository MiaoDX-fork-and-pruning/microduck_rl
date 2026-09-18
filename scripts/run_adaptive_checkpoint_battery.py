#!/usr/bin/env python3
"""Export a runner checkpoint, run the frozen battery, and bind provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from mjlab_microduck.evaluation.capability import CapabilityReport
from mjlab_microduck.evaluation.capability import BUCKETS

EXPECTED_BATTERY_STEPS = 300


def _validate_battery_traces(battery_dir: Path, payload: dict) -> None:
    """Require the frozen six-bucket trace ABI before accepting a report."""
    cases = {str(case.get("bucket")): case for case in payload.get("cases", [])}
    if set(cases) != set(BUCKETS):
        raise ValueError("battery report must contain exactly six canonical buckets")
    for bucket in BUCKETS:
        case = cases[bucket]
        if int(case.get("steps", -1)) != EXPECTED_BATTERY_STEPS:
            raise ValueError(f"{bucket} trace has unexpected step count")
        if case.get("finite_61d_14d") is not True:
            raise ValueError(f"{bucket} trace failed the 61D/14D finite ABI")
        trace_path = battery_dir / f"{bucket}.npz"
        if not trace_path.exists():
            raise FileNotFoundError(f"missing {bucket} trace: {trace_path}")
        import numpy as np

        with np.load(trace_path, allow_pickle=False) as trace:
            for name, shape in (
                ("observation", (EXPECTED_BATTERY_STEPS, 61)),
                ("raw_action", (EXPECTED_BATTERY_STEPS, 14)),
                ("applied_action", (EXPECTED_BATTERY_STEPS, 14)),
            ):
                if name not in trace:
                    raise ValueError(f"{bucket} trace is missing {name}")
                array = np.asarray(trace[name])
                if array.shape != shape:
                    raise ValueError(f"{bucket} trace {name} shape {array.shape} != {shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"{bucket} trace {name} contains non-finite values")


def run_checkpoint_battery(
    *,
    checkpoint: Path,
    task_id: str,
    axis_mode: str,
    seed_set_id: str,
    evaluation_seed: int,
    output: Path,
    export_device: str = "cuda:0",
) -> dict:
    """Export one checkpoint, run the frozen battery, and bind both artifacts."""
    if not seed_set_id.strip():
        raise ValueError("seed_set_id must be non-empty")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    source_sha = os.environ.get("MICRODUCK_SOURCE_SHA")
    if not source_sha:
        raise ValueError("MICRODUCK_SOURCE_SHA is required for evaluator provenance")
    source = Path(__file__).resolve().parents[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx = output.parent / f"{checkpoint.parent.name}.onnx"
    battery_dir = output.parent / "battery"

    # One environment is enough to construct/export the policy and keeps the
    # evaluator from competing with the training allocation on CloudML.
    subprocess.run(
        [
            sys.executable,
            str(source / "scripts/export.py"),
            task_id,
            "--checkpoint-file",
            str(checkpoint),
            "--onnx-file",
            str(onnx),
            "--num-envs",
            "1",
            "--device",
            export_device,
        ],
        check=True,
    )
    if not onnx.exists():
        raise FileNotFoundError(f"export did not write ONNX artifact: {onnx}")

    battery_result = subprocess.run(
        [
            sys.executable,
            str(source / "scripts/run_adaptive_capability_battery.py"),
            "--onnx",
            str(onnx),
            "--output",
            str(battery_dir),
            "--seed",
            str(evaluation_seed),
            "--duration-seconds",
            "6.0",
            "--task-id",
            task_id,
            "--source-sha",
            source_sha,
            "--axis-mode",
            axis_mode,
        ],
        check=False,
    )
    report_path = battery_dir / "capability.json"
    if battery_result.returncode not in (0, 2):
        raise subprocess.CalledProcessError(battery_result.returncode, battery_result.args)
    if not report_path.exists():
        raise FileNotFoundError(f"battery did not write capability report: {report_path}")

    # Exit code 2 means a valid report whose aggregate capability gate failed;
    # this is a training signal for the runner, not an evaluator crash.
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    CapabilityReport.from_dict(payload)
    if payload.get("seed") != evaluation_seed:
        raise ValueError("battery report evaluation seed mismatch")
    _validate_battery_traces(battery_dir, payload)

    metadata = dict(payload["metadata"])
    metadata.update(
        {
            # Keep this string byte-for-byte identical to the checkpoint path
            # passed to the runner validator; hash the actual file contents.
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed_set_id": seed_set_id,
            "evaluation_seed": evaluation_seed,
            "exported_onnx": str(onnx),
            "exported_onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        }
    )
    payload["metadata"] = metadata
    payload["exported_onnx"] = str(onnx)
    payload["exported_onnx_sha256"] = metadata["exported_onnx_sha256"]
    # Re-validate after provenance rebinding so malformed or forged reports
    # cannot reach the adaptive gate.
    CapabilityReport.from_dict(payload)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--axis-mode", required=True)
    parser.add_argument("--seed-set-id", required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--export-device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_checkpoint_battery(
        checkpoint=args.checkpoint,
        task_id=args.task_id,
        axis_mode=args.axis_mode,
        seed_set_id=args.seed_set_id,
        evaluation_seed=args.evaluation_seed,
        output=args.output,
        export_device=args.export_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
