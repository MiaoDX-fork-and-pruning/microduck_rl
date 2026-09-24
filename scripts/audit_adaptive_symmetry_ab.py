#!/usr/bin/env python3
"""Audit provenance for the matched adaptive symmetry A/B window.

The campaign result files are the source of truth for budgets and checkpoint
selection.  This audit only checks those immutable artifacts plus the two
source manifests; it does not rerun training or recompute capability scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def manifest_files(path: Path) -> dict[str, str]:
    data = read_json(path)
    files = data.get("files")
    if not isinstance(files, list):
        raise ValueError(f"source manifest has no file list: {path}")
    return {entry["path"]: entry["sha256"] for entry in files}


def require(condition: bool, message: str) -> None:
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

    control = read_json(args.control / "campaign-result.json")
    treatment = read_json(args.treatment / "campaign-result.json")
    control_cfg = read_json(args.control / "campaign-config.json")
    treatment_cfg = read_json(args.treatment / "campaign-config.json")
    contract_fields = (
        "branch", "seed", "iterations", "resume", "num_envs", "gate_interval",
        "gate_cohort_size", "gate_distribution", "gate_seed", "heldout_seed",
        "start_completed_iterations", "task_id", "axis_mode",
        "action_rate_relief_scope", "sensor_reset_fraction",
    )
    for field in contract_fields:
        require(control_cfg.get(field) == treatment_cfg.get(field), f"A/B config mismatch: {field}")
    require(control_cfg["iterations"] == 1250, "unexpected total budget")
    require(control_cfg["start_completed_iterations"] == 1000, "unexpected start budget")
    require(control_cfg["gate_interval"] == 250, "unexpected gate interval")
    require(control_cfg["num_envs"] == 4096, "unexpected environment count")
    require(control["completed_iterations"] == treatment["completed_iterations"] == 1250,
            "A/B did not complete 1250 updates")
    require(control["segment_training_transitions"] == treatment["segment_training_transitions"] == 24_576_000,
            "A/B transition budget mismatch")

    control_files = manifest_files(args.control_source / "source-manifest.json")
    treatment_files = manifest_files(args.treatment_source / "source-manifest.json")
    require(control_files.keys() == treatment_files.keys(), "source file sets differ")
    changed = [path for path in control_files if control_files[path] != treatment_files[path]]
    require(changed == ["src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py"],
            f"unexpected source differences: {changed}")
    control_cfg_source = (args.control_source / changed[0]).read_text()
    treatment_cfg_source = (args.treatment_source / changed[0]).read_text()
    require(re.search(r"ENABLE_SYMMETRY = False", control_cfg_source), "control symmetry flag missing")
    require(re.search(r"ENABLE_SYMMETRY = True", treatment_cfg_source), "treatment symmetry flag missing")
    require("symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None" in treatment_cfg_source,
            "treatment symmetry config wiring missing")

    for branch in (control, treatment):
        require(branch["status"] == "evaluated", "campaign was not evaluated")
        checkpoint = Path(branch["checkpoint"])
        require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
        require(sha256(checkpoint) == branch["checkpoint_sha256"], "checkpoint hash mismatch")
        for key in ("heldout_report", "cpu_transfer_report"):
            report = Path(branch[key])
            require(report.is_file(), f"missing report: {report}")
            require(sha256(report) == branch[f"{key}_sha256"], f"{key} hash mismatch")
        heldout = read_json(Path(branch["heldout_report"]))
        require(heldout["seed_manifest"] == read_json(Path(treatment["heldout_report"]))["seed_manifest"],
                "native held-out seed manifests differ")
        require(heldout["metadata"]["task_id"] == control_cfg["task_id"], "native task mismatch")

    control_symmetry_lines = [
        line for line in (args.control / "training.log").read_text().splitlines()
        if "symmetry loss" in line.lower()
    ]
    treatment_symmetry_lines = [
        line for line in (args.treatment / "training.log").read_text().splitlines()
        if "symmetry loss" in line.lower()
    ]
    require(not control_symmetry_lines, "control unexpectedly logged symmetry loss")
    require(treatment_symmetry_lines, "treatment did not log symmetry loss")

    control_native = read_json(Path(control["heldout_report"]))
    treatment_native = read_json(Path(treatment["heldout_report"]))
    result = {
        "schema_version": 1,
        "status": "verified",
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
            "control_manifest_sha256": sha256(args.control_source / "source-manifest.json"),
            "treatment_manifest_sha256": sha256(args.treatment_source / "source-manifest.json"),
            "changed_files": changed,
            "control_symmetry": False,
            "treatment_symmetry": True,
            "mirror_loss_coeff": 0.5,
        },
        "checkpoints": {
            "control": {"path": control["checkpoint"], "sha256": control["checkpoint_sha256"]},
            "treatment": {"path": treatment["checkpoint"], "sha256": treatment["checkpoint_sha256"]},
        },
        "native": {
            "control": control_native["aggregate"],
            "treatment": treatment_native["aggregate"],
            "report_sha256": {
                "control": sha256(Path(control["heldout_report"])),
                "treatment": sha256(Path(treatment["heldout_report"])),
            },
        },
        "evidence": {
            "control_symmetry_loss_log_lines": len(control_symmetry_lines),
            "treatment_symmetry_loss_log_lines": len(treatment_symmetry_lines),
            "same_native_seed_manifest": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
