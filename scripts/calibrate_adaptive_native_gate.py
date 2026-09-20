#!/usr/bin/env python3
"""Summarize canonical native evidence; do not invent gate calibration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np

from mjlab_microduck.evaluation.capability import DEFAULT_THRESHOLDS, canonical_sha256
from assemble_adaptive_phase2a_manifest import FIXED_TASK, sha256, validate_native
from mjlab_microduck.evaluation.capability import BUCKETS
from mjlab_microduck.tasks.adaptive_curriculum import ADAPTIVE_AXIS_CONFIGS, CapabilityGate


def replay_native_curves(reports: list[tuple[Path, dict]]) -> dict:
    """Replay observed checkpoint curves without calling sparse samples calibrated."""
    curves: dict[int, list[tuple[int, Path, dict]]] = {}
    for path, payload in reports:
        match = re.fullmatch(r"fixed-s(\d+)-i(\d+)", path.parent.name)
        if match:
            seed, index = map(int, match.groups())
            curves.setdefault(seed, []).append((index, path, payload))
    trials = []
    for seed, curve in sorted(curves.items()):
        for alpha in (0.1, 0.25, 0.5, 1.0):
            gate = CapabilityGate(ADAPTIVE_AXIS_CONFIGS, critical_buckets=BUCKETS,
                                  ema_alpha=alpha, preservation_tolerance=0.05)
            windows = []
            for index, path, payload in sorted(curve):
                metrics = {name: float(bucket["score"]) for name, bucket in payload["buckets"].items()}
                step = (index + 1) * 24
                decision = gate.decide(step, metrics, checkpoint=payload["metadata"]["checkpoint"], seed=seed)
                windows.append({"checkpoint_index": index, "env_step": step,
                                "source_report": str(path), "report_sha256": sha256(path),
                                "lower_tail_score": min(metrics.values()),
                                "outcome": decision.outcome.value})
            trials.append({"training_seed": seed, "ema_alpha": alpha,
                           "windows": windows, "transition_count": len(gate.trace)})
    return {"status": "diagnostic_only" if trials else "no_curves",
            "axes": [asdict(axis) for axis in ADAPTIVE_AXIS_CONFIGS],
            "preservation_tolerance": 0.05, "trials": trials,
            "limitation": "Sparse checkpoint replay does not measure gate-window noise, stage responses, or preservation after advancement; it cannot identify EMA/dwell/tolerance values."}


def calibrate(root: Path, output: Path) -> dict:
    reports = sorted(root.glob("*/native_capability.json"))
    errors, accepted, excluded = [], [], []
    native_reports = []
    values: dict[str, list[float]] = {}
    if not reports:
        errors.append(f"no native reports under {root}")
    for path in reports:
        try:
            payload = validate_native(path)
            if payload["metadata"]["task_id"] != FIXED_TASK:
                excluded.append({"path": str(path), "reason": "not canonical fixed control"})
                continue
            accepted.append({"path": str(path), "sha256": sha256(path), "evaluator_config_sha256": payload["metadata"]["evaluator_config_sha256"]})
            native_reports.append((path, payload))
            for bucket in payload["buckets"].values():
                for name, value in bucket["raw"].items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        values.setdefault(name, []).append(float(value))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
    # Cross-sectional checkpoint quantiles cannot identify a temporal smoothing
    # or dwell policy. Require a subsequent gate replay experiment and rationale.
    missing = ["ema_alpha", "dwell", "preservation_tolerance", "curriculum_thresholds"]
    config = {
        "name": "adaptive_native_gate_calibration_v2",
        "status": "incomplete", "source": str(root), "report_count": len(reports),
        "accepted_report_count": len(accepted), "source_reports": accepted, "excluded_reports": excluded,
        "errors": errors, "missing_calibration": missing, "calibrated_parameters": {},
        "units": {"tracking_error_m_s": "m/s", "angular_tracking_error_rad_s": "rad/s", "zero_drift_m": "m", "tilt_p95_rad": "rad", "survival_fraction": "fraction"},
        "native_quantiles": {name: {"count": len(samples), "p50": float(np.percentile(samples, 50)), "p95": float(np.percentile(samples, 95))} for name, samples in values.items()},
        "product_thresholds": dict(DEFAULT_THRESHOLDS),
        "product_gate_score_minimum": 0.8,
        "product_gate_error_limits": {name: value * 0.2 for name, value in DEFAULT_THRESHOLDS.items()},
        "temporal_gate_replay": replay_native_curves(native_reports),
        "decision": "insufficient evidence for calibration; product thresholds unchanged; temporal gate replay and parameter rationale required",
    }
    config["config_sha256"] = canonical_sha256(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n")
    return config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = calibrate(args.reports, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
