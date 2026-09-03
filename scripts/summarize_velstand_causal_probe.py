#!/usr/bin/env python3
"""Write a bounded Phase-B diagnosis from a closed-loop evaluator report."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compatibility", type=Path, required=True)
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    compatibility = json.loads(args.compatibility.read_text())
    rollout = json.loads(args.rollout.read_text())
    stand = next((p for p in rollout.get("profiles", []) if p.get("behavior") == "stand"), None)
    if stand is None:
        raise ValueError("rollout report has no VELSTAND/stand profile")
    phase_a = bool(compatibility.get("passed"))
    phase_b = bool(stand.get("passed"))
    diagnosis = "adapter_teacher_semantics" if not phase_a else (
        "state_distribution_coverage_or_model_capacity" if not phase_b else "closed_loop_pass")
    report = {
        "schema": "generalist-g0-velstand-causal-probe-diagnosis",
        "phase_a": {"passed": phase_a, "report": str(args.compatibility),
                    "max_abs": compatibility.get("max_abs"), "mean_abs": compatibility.get("mean_abs")},
        "phase_b": {"passed": phase_b, "report": str(args.rollout),
                    "finite": stand.get("finite"), "max_tilt_rad": stand.get("max_tilt_rad"),
                    "safety_gate_rad": stand.get("stability_gate_rad")},
        "diagnosis": diagnosis,
        "recommendation": "Do not start merged PPO; repair coverage, initialization, action representation, or capacity and rerun the fixed battery." if phase_a and not phase_b else "Repair teacher semantics before interpreting student results." if not phase_a else "Freeze the student and request a parent-plan increment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
