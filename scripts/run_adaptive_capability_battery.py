#!/usr/bin/env python3
"""Run the frozen six-bucket capability battery for an adaptive checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_specialist_action_battery import command_cases, run_case  # noqa: E402
from mjlab_microduck.evaluation.capability import (  # noqa: E402
    DEFAULT_INSTANTANEOUS_CAPS,
    DEFAULT_TRACKING_METRIC,
    DEFAULT_TRACKING_TAU_S,
    DEFAULT_THRESHOLDS,
    TRACKING_METRIC_SIGNED_EMA,
    build_capability_report,
    tracking_error_metrics,
)


def _report_metadata(
    onnx_path: Path,
    seed: int,
    *,
    task_id: str,
    source_sha: str,
    axis_mode: str,
    config_hash: str,
) -> dict[str, object]:
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    return {
        "task_id": task_id,
        "source_sha": source_sha,
        "evaluator_config_sha256": config_hash,
        "checkpoint": str(onnx_path),
        "checkpoint_sha256": digest,
        "policy_format": "onnx",
        "seed_set_id": f"adaptive-default-{seed}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _raw_case(
    report: dict,
    trace: dict,
    *,
    tracking_metric: str = DEFAULT_TRACKING_METRIC,
    tracking_tau_s: float = DEFAULT_TRACKING_TAU_S,
) -> dict[str, object]:
    requested = trace["requested_command"]
    linear = trace["trunk_linear_velocity_m_s"]
    angular = trace["trunk_angular_velocity_rad_s"]
    command = requested[:, :3].astype(float)
    achieved = linear[:, :2].astype(float)
    angular_target = command[:, 2]
    angular_achieved = angular[:, 2].astype(float)
    bucket = report["bucket"]
    raw: dict[str, object] = {
        "survival_fraction": float(
            len(trace["trunk_tilt_rad"]) / max(1, report["steps"])
        ),
        "tilt_p95_rad": float(np.percentile(trace["trunk_tilt_rad"], 95))
        if len(trace["trunk_tilt_rad"])
        else None,
        "episode_length_mean": float(report["steps"] / 50.0),
        "action_magnitude_mean": float(np.mean(np.abs(trace["applied_action"])))
        if len(trace["applied_action"])
        else None,
    }
    if bucket == "zero":
        raw["zero_drift_m"] = float(
            np.linalg.norm(np.asarray(report["world_displacement_m"], dtype=float)[:2])
        )
    elif bucket in {"yaw", "turn-left", "turn-right"}:
        samplewise, signed_ema = tracking_error_metrics(
            angular_achieved,
            angular_target,
            dt=1.0 / 50.0,
            tau_s=tracking_tau_s,
        )
        raw["angular_tracking_error_rad_s"] = float(
            signed_ema if tracking_metric == TRACKING_METRIC_SIGNED_EMA else samplewise
        )
        if tracking_metric == TRACKING_METRIC_SIGNED_EMA:
            raw["angular_tracking_error_samplewise_rad_s"] = float(samplewise)
    else:
        axis = 0 if bucket == "forward" else 1
        samplewise, signed_ema = tracking_error_metrics(
            achieved[:, axis],
            command[:, axis],
            dt=1.0 / 50.0,
            tau_s=tracking_tau_s,
        )
        raw["tracking_error_m_s"] = float(
            signed_ema if tracking_metric == TRACKING_METRIC_SIGNED_EMA else samplewise
        )
        if tracking_metric == TRACKING_METRIC_SIGNED_EMA:
            raw["tracking_error_samplewise_m_s"] = float(samplewise)
    return raw


def run_battery(
    onnx_path: Path,
    output: Path,
    seed: int,
    duration_seconds: float,
    smoke: bool,
    *,
    task_id: str,
    source_sha: str,
    axis_mode: str,
) -> dict:
    tracking_metric = TRACKING_METRIC_SIGNED_EMA
    tracking_tau_s = DEFAULT_TRACKING_TAU_S
    output.mkdir(parents=True, exist_ok=True)
    raw_metrics: dict[str, dict[str, object]] = {}
    cases = []
    for offset, case in enumerate(command_cases("adaptive_velocity", smoke)):
        report, trace = run_case(
            "adaptive_velocity",
            onnx_path,
            case,
            seed + offset,
            duration_seconds,
            smoke=smoke,
        )
        report["bucket"] = case["bucket"]
        cases.append(report)
        raw_metrics[case["bucket"]] = _raw_case(
            report,
            trace,
            tracking_metric=tracking_metric,
            tracking_tau_s=tracking_tau_s,
        )
        trace_path = output / f"{case['bucket']}.npz"
        import numpy as np

        np.savez_compressed(trace_path, **trace)
        report["trace"] = str(trace_path)
    config = {
        "name": "adaptive_velocity_six_bucket_v2",
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "tracking_metric": tracking_metric,
        "tracking_metric_tau_s": tracking_tau_s,
        "instantaneous_caps": dict(DEFAULT_INSTANTANEOUS_CAPS),
    }
    import hashlib

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    capability = build_capability_report(
        raw_metrics,
        metadata=_report_metadata(
            onnx_path,
            seed,
            task_id=task_id,
            source_sha=source_sha,
            axis_mode=axis_mode,
            config_hash=config_hash,
        ),
        axis_mode=axis_mode,
        evaluator_config=config,
    )
    payload = {
        **capability.payload,
        "schema_version": 2,
        "battery": "adaptive_velocity_six_bucket",
        "checkpoint": str(onnx_path),
        "seed": seed,
        "duration_seconds": duration_seconds,
        "metrics": capability.metrics,
        "score": capability.payload["aggregate"]["lower_tail_score"],
        "cases": cases,
        "seed_manifest": {
            "version": "adaptive-battery-seed-v1",
            "seed_set_id": f"adaptive-default-{seed}",
            "gate_seed": seed,
            "consumed_case_seeds": {
                case["bucket"]: seed + offset
                for offset, case in enumerate(command_cases("adaptive_velocity", smoke))
            },
            "sources": ["reset_qpos_noise", "reset_qvel_noise"],
        },
    }
    (output / "capability.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task-id", default="Mjlab-Velocity-Flat-Adaptive-MicroDuck")
    parser.add_argument("--source-sha", default="unknown")
    parser.add_argument(
        "--axis-mode",
        choices=("all_static", "com", "head_com", "composed"),
        default="composed",
    )
    args = parser.parse_args()
    payload = run_battery(
        args.onnx,
        args.output,
        args.seed,
        args.duration_seconds,
        args.smoke,
        task_id=args.task_id,
        source_sha=args.source_sha,
        axis_mode=args.axis_mode,
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["aggregate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
