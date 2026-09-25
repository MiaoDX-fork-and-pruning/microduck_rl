"""Inspect a converted Isaac Sim USD for the Microduck asset contract.

This script intentionally reports the authored USD, rather than the IsaacLab
articulation config.  It is therefore useful for catching importer omissions
(for example, MJCF actuator gains that did not become PhysX drives).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp

from isaaclab_microduck.assets.report import ACTUATED_ORDER


def _value(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    return attribute.Get() if attribute else None


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value]


def inspect_usd(path: Path) -> dict[str, Any]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"USD stage could not be opened: {path}")

    joints: dict[str, dict[str, Any]] = {}
    articulation_roots: list[str] = []
    rigid_bodies = 0
    collision_geometries = 0
    contact_reporters: list[str] = []
    for prim in stage.Traverse():
        schemas = set(prim.GetAppliedSchemas())
        if "PhysicsArticulationRootAPI" in schemas:
            articulation_roots.append(str(prim.GetPath()))
        if "PhysicsRigidBodyAPI" in schemas:
            rigid_bodies += 1
        if "PhysicsCollisionAPI" in schemas:
            collision_geometries += 1
        if "PhysxContactReportAPI" in schemas:
            contact_reporters.append(str(prim.GetPath()))
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        name = prim.GetName()
        joints[name] = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "axis": _value(prim, "physics:axis"),
            "lower_limit_deg": _value(prim, "physics:lowerLimit"),
            "upper_limit_deg": _value(prim, "physics:upperLimit"),
            "drive_stiffness": _value(prim, "drive:angular:physics:stiffness"),
            "drive_damping": _value(prim, "drive:angular:physics:damping"),
            "drive_max_force": _value(prim, "drive:angular:physics:maxForce"),
            "armature": _value(prim, "physxJoint:armature"),
            "joint_friction": _value(prim, "physxJoint:jointFriction"),
        }

    ordered = [joints[name] for name in ACTUATED_ORDER if name in joints]
    missing = [name for name in ACTUATED_ORDER if name not in joints]
    drive_configured = bool(ordered) and all(
        float(item["drive_stiffness"] or 0.0) > 0.0
        or float(item["drive_damping"] or 0.0) > 0.0
        for item in ordered
    )
    return {
        "usd_path": str(path),
        "stage_valid": True,
        "active_prim_count": sum(1 for prim in stage.Traverse() if prim.IsActive()),
        "articulation_roots": articulation_roots,
        "rigid_body_count": rigid_bodies,
        "collision_geometry_count": collision_geometries,
        "contact_reporter_count": len(contact_reporters),
        "contact_reporter_paths": contact_reporters,
        "joint_count": len(joints),
        "actuated_joint_order": list(ACTUATED_ORDER),
        "actuated_joints": ordered,
        "missing_actuated_joints": missing,
        "drive_configured": drive_configured,
        "warnings": (["physx_drives_missing_or_zero"] if not drive_configured else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("--output", type=Path, help="write machine-readable JSON to this path")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    app = SimulationApp({"renderer": "RaytracedLighting", "headless": args.headless})
    try:
        report = inspect_usd(args.usd)
        encoded = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
            print(f"ISAACLAB_USD_INSPECTED:{args.usd} output={args.output}", flush=True)
        else:
            print(encoded, end="")
    finally:
        app.close()


if __name__ == "__main__":
    main()
