"""Compare two raw-source trace reports without normalizing or threshold hiding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SIGNALS: tuple[tuple[str, str, str], ...] = (
    ("gyro_adapter", "root_ang_vel_b_adapter", "root_ang_vel_b_adapter"),
    ("gravity_adapter", "projected_gravity_b_adapter", "projected_gravity_b_adapter"),
    ("joint_pos", "joint_pos", "joint_pos"),
    ("joint_vel", "joint_vel", "joint_vel"),
    ("foot_site_pos_w", "foot_site_pos_w", "foot_site_pos_w"),
    ("foot_site_vel_w", "foot_site_vel_w", "foot_site_vel_w"),
    ("contact_found", "contact_found", "contact_found"),
    ("contact_force", "contact_force", "contact_force"),
    ("subtree_angular_momentum", "named_root_angmom", "subtree_angmom_adapter"),
)


def _stats(reference: Any, observed: Any) -> dict[str, object]:
    ref = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if ref.shape != obs.shape:
        return {"shape_match": False, "reference_shape": list(ref.shape), "observed_shape": list(obs.shape)}
    delta = obs - ref
    finite = bool(np.isfinite(ref).all() and np.isfinite(obs).all())
    return {
        "shape_match": True,
        "shape": list(ref.shape),
        "finite": finite,
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "reference_min": float(np.min(ref)) if ref.size else 0.0,
        "reference_max": float(np.max(ref)) if ref.size else 0.0,
    }


def compare(mjlab_path: Path, isaaclab_path: Path, output: Path) -> dict[str, object]:
    mj = json.loads(mjlab_path.read_text())
    isaac = json.loads(isaaclab_path.read_text())
    if mj.get("schema_version") != isaac.get("schema_version"):
        raise ValueError("trace schema versions differ")
    if len(mj["records"]) != len(isaac["records"]):
        raise ValueError("trace lengths differ")
    per_signal: dict[str, dict[str, object]] = {}
    for label, mj_name, isaac_name in SIGNALS:
        reference = [record["raw"].get(mj_name) for record in mj["records"]]
        observed = [record["raw"].get(isaac_name) for record in isaac["records"]]
        if any(value is None for value in reference + observed):
            per_signal[label] = {"available": False}
        else:
            per_signal[label] = {"available": True, **_stats(reference, observed)}
    report = {
        "schema_version": mj["schema_version"],
        "reference_backend": "mjlab",
        "observed_backend": "isaaclab",
        "steps": len(mj["records"]),
        "reset_behavior": {
            "both_traces_start_from_explicit_reset": True,
            "partial_reset_checked": False,
            "note": "This bounded source probe compares a seeded scripted state sequence; reset-state buffers remain covered by the runtime parity probe.",
        },
        "source_notes": {"mjlab": mj.get("source_notes", {}), "isaaclab": isaac.get("source_notes", {})},
        "processing": {"mjlab": mj.get("processing", {}), "isaaclab": isaac.get("processing", {})},
        "signals": per_signal,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjlab", type=Path, required=True)
    parser.add_argument("--isaaclab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.mjlab, args.isaaclab, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
