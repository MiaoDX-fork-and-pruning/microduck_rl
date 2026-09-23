#!/usr/bin/env python3
"""Verify the matched adaptive hip-yaw limit-proximity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BUCKETS = ("zero", "forward", "lateral", "yaw", "turn-left", "turn-right")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def manifest(path: Path) -> dict[str, str]:
    return {entry["path"]: entry["sha256"] for entry in load(path)["files"]}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--treatment-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = load(args.control / "campaign-result.json")
    treatment = load(args.treatment / "campaign-result.json")
    control_cfg = load(args.control / "campaign-config.json")
    treatment_cfg = load(args.treatment / "campaign-config.json")
    for field in (
        "branch", "seed", "iterations", "resume", "num_envs", "gate_interval",
        "gate_cohort_size", "gate_distribution", "gate_seed", "heldout_seed",
        "start_completed_iterations", "task_id", "axis_mode",
        "action_rate_relief_scope", "sensor_reset_fraction",
    ):
        check(control_cfg.get(field) == treatment_cfg.get(field), f"A/B mismatch: {field}")
    check(control_cfg["iterations"] == 1250 and control_cfg["start_completed_iterations"] == 1000,
          "unexpected training window")
    check(control_cfg["num_envs"] == 4096 and control_cfg["gate_interval"] == 250,
          "unexpected matched budget")
    check(control["completed_iterations"] == treatment["completed_iterations"] == 1250,
          "training did not complete the requested budget")
    check(control["segment_training_transitions"] == treatment["segment_training_transitions"] == 24_576_000,
          "segment transition budget mismatch")

    control_files = manifest(args.control_source / "source-manifest.json")
    treatment_files = manifest(args.treatment_source / "source-manifest.json")
    check(control_files.keys() == treatment_files.keys(), "source file sets differ")
    changed = [path for path in control_files if control_files[path] != treatment_files[path]]
    check(changed == ["src/mjlab_microduck/tasks/microduck_adaptive_velocity_env_cfg.py"],
          f"unexpected source differences: {changed}")
    control_source = (args.control_source / changed[0]).read_text()
    treatment_source = (args.treatment_source / changed[0]).read_text()
    check("hip_yaw_limit_proximity" not in control_source, "control contains treatment term")
    check("hip_yaw_limit_proximity" in treatment_source, "treatment term missing")
    check("weight=LATERAL_DRIVE_HIP_YAW_LIMIT_WEIGHT" in treatment_source,
          "treatment weight wiring missing")
    check("LATERAL_DRIVE_HIP_YAW_LIMIT_WEIGHT = -0.5" in treatment_source,
          "treatment weight changed")
    check("LATERAL_DRIVE_HIP_YAW_LIMIT_MARGIN_RAD = 0.15" in treatment_source,
          "treatment margin changed")

    cases: dict[str, dict] = {}
    for label, result in (("control", control), ("treatment", treatment)):
        check(result["status"] == "evaluated", f"{label} was not evaluated")
        checkpoint = Path(result["checkpoint"])
        check(digest(checkpoint) == result["checkpoint_sha256"], f"{label} checkpoint hash mismatch")
        report = Path(result["heldout_report"])
        check(digest(report) == result["heldout_report_sha256"], f"{label} native report hash mismatch")
        transfer = Path(result["cpu_transfer_report"])
        check(digest(transfer) == result["cpu_transfer_report_sha256"], f"{label} CPU report hash mismatch")
        native = load(report)
        check(tuple(native["buckets"]) == BUCKETS, f"{label} bucket schema mismatch")
        check(native["metadata"]["task_id"] == control_cfg["task_id"], f"{label} task mismatch")
        check(native["seed_manifest"] == load(Path(control["heldout_report"]))["seed_manifest"],
              f"{label} native seed manifest mismatch")
        cases[label] = {
            "native": {
                "aggregate": native["aggregate"],
                "scores": {name: native["buckets"][name]["score"] for name in BUCKETS},
                "report_sha256": digest(report),
            },
            "cpu": {
                "aggregate": load(transfer)["aggregate"],
                "report_sha256": digest(transfer),
            },
            "checkpoint": {"path": str(checkpoint), "sha256": digest(checkpoint)},
        }

    native_delta = {
        name: cases["treatment"]["native"]["scores"][name]
        - cases["control"]["native"]["scores"][name]
        for name in BUCKETS
    }
    control_tail = cases["control"]["native"]["aggregate"]["lower_tail_score"]
    treatment_tail = cases["treatment"]["native"]["aggregate"]["lower_tail_score"]
    delta_tail = treatment_tail - control_tail
    accepted = bool(delta_tail >= 0.05 or all(value >= 0.80 for value in cases["treatment"]["native"]["scores"].values()))
    result = {
        "schema_version": 1,
        "status": "verified",
        "decision": "reject_treatment" if not accepted else "retain_candidate",
        "contract": {
            "branch": control_cfg["branch"],
            "seed": control_cfg["seed"],
            "start_completed_iterations": control_cfg["start_completed_iterations"],
            "completed_iterations": control["completed_iterations"],
            "segment_training_transitions": control["segment_training_transitions"],
            "num_envs": control_cfg["num_envs"],
            "gate_interval": control_cfg["gate_interval"],
            "gate_seed": control_cfg["gate_seed"],
            "heldout_seed": control_cfg["heldout_seed"],
        },
        "source": {
            "control_manifest_sha256": digest(args.control_source / "source-manifest.json"),
            "treatment_manifest_sha256": digest(args.treatment_source / "source-manifest.json"),
            "changed_files": changed,
            "control_has_hip_yaw_term": False,
            "treatment_has_hip_yaw_term": True,
            "treatment_weight": -0.5,
            "treatment_margin_rad": 0.15,
        },
        "runs": cases,
        "native_delta_treatment_minus_control": native_delta,
        "lower_tail": {
            "control": control_tail,
            "treatment": treatment_tail,
            "delta": delta_tail,
            "retention_threshold": 0.05,
            "all_six_threshold": 0.80,
        },
        "accepted": accepted,
        "limitations": [
            "Both branches start from the same checkpoint and consume the same declared seed sets.",
            "GPU Warp rollout execution is not bitwise deterministic; this is a matched-budget A/B, not a deterministic replay.",
            "The treatment improves the native lower tail by less than 0.05 and does not reach all-six mastery.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
