#!/usr/bin/env python3
"""Verify the pinned official fall-recovery contract without copying its scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


PINNED_COMMIT = "590b986bd8c0d50ae02cb3ea2f59c463b6828168"


def run(command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    checkout = args.official_checkout.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"official checkout is {commit}, expected {PINNED_COMMIT}")

    tests = [
        run(["cargo", "test", "-p", "duck-control", "fall::tests", "--", "--nocapture"], checkout),
        run(["cargo", "test", "-p", "robotd", "the_limp_fall_pose_ramp_ends_at_the_standing_pose", "--", "--nocapture"], checkout),
        run(["cargo", "test", "-p", "robotd", "limp_fall_ships_on_and_refuses_nothing", "--", "--nocapture"], checkout),
    ]
    standup = json.loads(args.standup_report.read_text(encoding="utf-8"))
    primary = next(case for case in standup["cases"] if case["id"] == "sit_to_stand")
    probe = next(case for case in standup["cases"] if case["id"] == "prone_recovery_probe")
    official_source = (checkout / "robotd/src/main.rs").read_text(encoding="utf-8")
    has_external_replay = "--replay" in official_source or "NdjsonIo" in official_source

    checks = {
        "official_commit_pinned": True,
        "official_fall_predictor_tests": tests[0]["passed"],
        "official_pose_ramp_test": tests[1]["passed"],
        "official_limp_fall_default_test": tests[2]["passed"],
        "standup_primary_handoff_passed": primary["passed"],
        "persistent_sensor_injection_bridge": has_external_replay,
    }
    report = {
        "schema_version": 1,
        "official_repository": "https://github.com/pollen-robotics/microduck.git",
        "official_commit": commit,
        "recovery_contract": [
            "fall_predictor",
            "limp_until_landed",
            "pose_to_standing",
            "hand_back_to_standing_policy",
        ],
        "official_defaults": {
            "predictor_tilt_z": -0.90,
            "predictor_predicted_z": -0.5,
            "predictor_lookahead_ms": 300,
            "predictor_debounce_ms": 60,
            "landed_still_rate_rad_s": 1.0,
            "landed_still_ms": 200,
            "limp_max_ms": 1500,
            "pose_ms": 600,
            "pose_gain": 160,
        },
        "tests": tests,
        "mujoco_policy_evidence": {
            "standup_primary": primary,
            "prone_recovery_probe": probe,
        },
        "checks": checks,
        "p1_passed": all(checks.values()),
        "blocker": None if all(checks.values()) else (
            "Pinned robotd has no persistent external sensor-frame injection seam; "
            "official decisions cannot yet be driven end-to-end by MuJoCo without "
            "duplicating the scheduler."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--standup-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = verify(make_parser().parse_args())
    print(json.dumps(report, indent=2))
    return 0 if report["p1_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
