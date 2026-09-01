#!/usr/bin/env python3
"""Run deterministic, single-policy ONNX action batteries in CPU MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_policy import DEFAULT_POSE, PolicyInference  # noqa: E402


COMMAND_SPEEDS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)
COMMAND_MODES = ("direct_step", "command_ema")
EMA_ALPHA = 0.08
CONTROL_HZ = 50
PHYSICS_STEPS = 4
TILT_FAILURE_RAD = math.radians(65.0)
TILT_RECOVERED_RAD = math.radians(35.0)

PROFILE_SCENES = {
    # The canonical no-wheel deployment rehearsal uses scene.xml, whose
    # robot include is robot_allcollisions.xml. Keep that scene for all
    # no-wheel specialists; rollers remain an independent model/profile.
    "walk_all_collisions": ROOT / "src/mjlab_microduck/robot/microduck/scene.xml",
    "rollers": ROOT / "src/mjlab_microduck/robot/microduck/scene_rollers.xml",
}

POLICY_PROFILES = {
    "velocity_flat": "walk_all_collisions",
    "velstand_flat": "walk_all_collisions",
    "sitstand_flat": "walk_all_collisions",
    "ground_pick_flat": "walk_all_collisions",
    "ball_kick_flat": "walk_all_collisions",
    "roulade_flat": "walk_all_collisions",
    "standup_flat": "walk_all_collisions",
    "roller_standup": "rollers",
    "velocity_rollers": "rollers",
    "swizzle": "rollers",
    "roller_crouch": "rollers",
    "roller_slope": "rollers",
    "spin": "rollers",
}

LOCOMOTION_POLICIES = {"velocity_flat", "velocity_rollers"}
PHASE_POLICIES = {"ground_pick_flat", "roller_crouch", "spin"}
RECOVERY_POLICIES = {"standup_flat", "roller_standup", "roulade_flat"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_cases(policy_id: str, smoke: bool) -> list[dict[str, Any]]:
    if policy_id in LOCOMOTION_POLICIES:
        speeds = COMMAND_SPEEDS[:1] if smoke else COMMAND_SPEEDS
        modes = COMMAND_MODES[:1] if smoke else COMMAND_MODES
        return [
            {"id": f"vx_{speed:.2f}_{mode}", "speed": speed, "input_mode": mode}
            for speed in speeds
            for mode in modes
        ]
    if policy_id == "sitstand_flat":
        return [{"id": "sit_then_rise", "input_mode": "direct_step"}]
    if policy_id == "standup_flat":
        cases = [{"id": "sit_to_stand", "input_mode": "direct_step", "acceptance": "primary"}]
        if not smoke:
            cases.append({"id": "prone_recovery_probe", "input_mode": "direct_step", "acceptance": "probe"})
        return cases
    return [{"id": "canonical", "input_mode": "direct_step"}]


def apply_command_input(
    requested: np.ndarray, previous: np.ndarray, mode: str, alpha: float = EMA_ALPHA
) -> np.ndarray:
    if mode == "direct_step":
        return requested.copy()
    if mode == "command_ema":
        return previous + alpha * (requested - previous)
    raise ValueError(f"unsupported command input mode: {mode}")


def trunk_metrics(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> dict[str, Any]:
    rotation = data.xmat[body_id].reshape(3, 3)
    up_dot = float(np.clip(rotation[2, 2], -1.0, 1.0))
    position = data.xpos[body_id].copy()
    quaternion = data.xquat[body_id].copy()
    return {
        "position": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "height_m": float(position[2]),
        "tilt_rad": float(math.acos(up_dot)),
    }


def contact_names(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    names: set[str] = set()
    for contact in data.contact[: data.ncon]:
        for geom_id in (contact.geom1, contact.geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            names.add(name or f"geom_{int(geom_id)}")
    return sorted(names)


def requested_command(policy_id: str, case: dict[str, Any], step: int, steps: int) -> np.ndarray:
    command = np.zeros(13, dtype=np.float32)
    if policy_id in LOCOMOTION_POLICIES:
        command[0] = float(case["speed"])
    elif policy_id == "sitstand_flat":
        command[0] = 1.0 if step < steps // 2 else 0.0
    elif policy_id in PHASE_POLICIES:
        phase = step / max(steps - 1, 1)
        command[0] = math.cos(2.0 * math.pi * phase)
        command[1] = math.sin(2.0 * math.pi * phase)
    elif policy_id == "swizzle":
        command[0] = 0.20
    return command


def reset_pose(policy_id: str, case_id: str, model: mujoco.MjModel, data: mujoco.MjData, policy: PolicyInference) -> None:
    mujoco.mj_resetData(model, data)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[free_id])
    height = 0.1385 if POLICY_PROFILES[policy_id] == "rollers" else 0.125
    data.qpos[qadr : qadr + 3] = [0.0, 0.0, height]
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for index, qpos_index in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_index] = DEFAULT_POSE[index]
    if policy_id == "standup_flat" and case_id == "prone_recovery_probe":
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, 0.07]
        data.qpos[qadr + 3 : qadr + 7] = [0.0, 1.0, 0.0, 0.0]
    elif policy_id == "standup_flat":
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, 0.060]
        for index, value in {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
                             10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0}.items():
            data.qpos[policy.joint_qpos_indices[index]] = value
    elif policy_id == "roller_standup":
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, 0.07]
        data.qpos[qadr + 3 : qadr + 7] = [0.0, 1.0, 0.0, 0.0]
    elif policy_id == "roulade_flat":
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, 0.10]
        data.qpos[qadr + 3 : qadr + 7] = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
    data.ctrl[:] = DEFAULT_POSE
    policy.last_action[:] = 0.0
    policy.command[:] = 0.0
    mujoco.mj_forward(model, data)


def classify_case(
    policy_id: str,
    finite: bool,
    displacement: list[float],
    max_tilt: float,
    final_tilt: float,
    recovered: bool,
) -> tuple[bool, str | None]:
    if not finite:
        return False, "nonfinite_state_or_action"
    if final_tilt > TILT_FAILURE_RAD:
        return False, "ended_fallen"
    if policy_id in LOCOMOTION_POLICIES and displacement[0] < 0.005:
        return False, "insufficient_forward_displacement"
    if policy_id in RECOVERY_POLICIES and max_tilt > TILT_FAILURE_RAD and not recovered:
        return False, "did_not_recover"
    return True, None


def _load_policy(model: mujoco.MjModel, data: mujoco.MjData, onnx_path: Path) -> PolicyInference:
    # Match scripts/infer_policy.py's deployment default: projected gravity is
    # the trained 61D observation, while raw accelerometer is opt-in.
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=str(onnx_path),
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    session = ort.InferenceSession(str(onnx_path))
    policy.ort_session = session
    policy.input_name = session.get_inputs()[0].name
    policy.output_name = session.get_outputs()[0].name
    return policy


def run_case(
    policy_id: str,
    onnx_path: Path,
    case: dict[str, Any],
    seed: int,
    duration_s: float,
    smoke: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    np.random.seed(seed)
    profile = POLICY_PROFILES[policy_id]
    model = mujoco.MjModel.from_xml_path(str(PROFILE_SCENES[profile]))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = _load_policy(model, data, onnx_path)
    reset_pose(policy_id, case["id"], model, data, policy)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    start = data.xpos[body_id].copy()
    steps = max(1, round(duration_s * CONTROL_HZ))
    applied_command = np.zeros(13, dtype=np.float32)
    fallen_seen = False
    recovered = False
    max_tilt = 0.0
    finite = True
    records: dict[str, list[Any]] = {
        "requested_command": [], "applied_command": [], "observation": [],
        "raw_action": [], "applied_action": [], "trunk_position": [],
        "trunk_quaternion_wxyz": [], "trunk_tilt_rad": [], "trunk_height_m": [],
        "contacts": [], "recovered": [],
    }

    for step in range(steps):
        requested = requested_command(policy_id, case, step, steps)
        applied_command = apply_command_input(requested, applied_command, case["input_mode"])
        policy.command = applied_command.copy()
        observation = policy.get_observations()
        raw_action = policy.infer()
        applied_action = np.asarray(raw_action, dtype=np.float32).copy()
        finite = bool(
            np.isfinite(observation).all()
            and np.isfinite(raw_action).all()
            and np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
        )
        if not finite:
            break
        policy.apply_action(applied_action)
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(model, data)
        metrics = trunk_metrics(model, data, body_id)
        tilt = float(metrics["tilt_rad"])
        max_tilt = max(max_tilt, tilt)
        fallen_seen = fallen_seen or tilt > TILT_FAILURE_RAD
        recovered = recovered or (fallen_seen and tilt < TILT_RECOVERED_RAD)
        records["requested_command"].append(requested)
        records["applied_command"].append(applied_command.copy())
        records["observation"].append(observation)
        records["raw_action"].append(np.asarray(raw_action, dtype=np.float32))
        records["applied_action"].append(applied_action)
        records["trunk_position"].append(metrics["position"])
        records["trunk_quaternion_wxyz"].append(metrics["quaternion_wxyz"])
        records["trunk_tilt_rad"].append(tilt)
        records["trunk_height_m"].append(metrics["height_m"])
        records["contacts"].append("|".join(contact_names(model, data)))
        records["recovered"].append(recovered)

    final = trunk_metrics(model, data, body_id)
    displacement = (data.xpos[body_id] - start).tolist()
    passed, failure_reason = classify_case(
        policy_id, finite, displacement, max_tilt, float(final["tilt_rad"]), recovered
    )
    trace = {
        key: np.asarray(value, dtype=str if key == "contacts" else None)
        for key, value in records.items()
    }
    report = {
        "id": case["id"],
        "seed": seed,
        "input_mode": case["input_mode"],
        "requested_speed_m_s": case.get("speed"),
        "steps": len(records["raw_action"]),
        "finite_61d_14d": finite and all(
            np.asarray(obs).shape == (61,) for obs in records["observation"]
        ) and all(np.asarray(action).shape == (14,) for action in records["raw_action"]),
        "world_displacement_m": displacement,
        "max_tilt_rad": max_tilt,
        "final_tilt_rad": float(final["tilt_rad"]),
        "final_height_m": float(final["height_m"]),
        "max_abs_raw_action": float(np.max(np.abs(records["raw_action"]))) if records["raw_action"] else None,
        "max_abs_applied_action": float(np.max(np.abs(records["applied_action"]))) if records["applied_action"] else None,
        "contact_geoms": sorted({name for item in records["contacts"] for name in item.split("|") if name}),
        "recovered": recovered,
        "passed": passed,
        "failure_reason": failure_reason,
    }
    if not report["finite_61d_14d"]:
        report["passed"] = False
        report["failure_reason"] = report["failure_reason"] or "abi_or_finiteness_failure"
    elif smoke:
        # Smoke proves artifact loading, ABI, finite inference, stepping, and
        # evidence serialization. Behavior gates require a full-duration case.
        report["passed"] = True
        report["failure_reason"] = None
        report["smoke_observation"] = failure_reason
    return report, trace


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policies = payload.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("manifest.policies must be a non-empty array")
    accepted = [entry for entry in policies if entry.get("accepted") is True]
    unknown = sorted({entry.get("id") for entry in accepted} - POLICY_PROFILES.keys())
    if unknown:
        raise ValueError(f"accepted policies lack a profile: {', '.join(unknown)}")
    return accepted


def run_battery(args: argparse.Namespace) -> dict[str, Any]:
    entries = load_manifest(args.manifest)
    if args.policy:
        entries = [entry for entry in entries if entry["id"] == args.policy]
        if not entries:
            raise ValueError(f"accepted policy not found: {args.policy}")
    reports = []
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        policy_id = entry["id"]
        onnx_path = (ROOT / entry["artifacts"]["onnx"]).resolve()
        expected_hash = entry["sha256"]["onnx"]
        actual_hash = sha256(onnx_path)
        if actual_hash != expected_hash:
            raise ValueError(f"{policy_id}: ONNX hash mismatch")
        policy_dir = args.output / policy_id
        policy_dir.mkdir(parents=True, exist_ok=True)
        case_reports = []
        for index, case in enumerate(command_cases(policy_id, args.smoke)):
            report, trace = run_case(
                policy_id,
                onnx_path,
                case,
                args.seed + index,
                args.duration_seconds,
                smoke=args.smoke,
            )
            report["acceptance"] = case.get("acceptance", "primary")
            if report["acceptance"] == "probe":
                report["passed_probe"] = report["passed"]
                report["passed"] = True
                report["failure_reason"] = None
            trace_path = policy_dir / f"{case['id']}.npz"
            np.savez_compressed(trace_path, **trace)
            report["trace"] = str(trace_path)
            case_reports.append(report)
        policy_report = {
            "policy_id": policy_id,
            "task": entry["task"],
            "profile": POLICY_PROFILES[policy_id],
            "scene": str(PROFILE_SCENES[POLICY_PROFILES[policy_id]]),
            "checkpoint_sha256": entry["sha256"]["checkpoint"],
            "onnx_sha256": actual_hash,
            "parity_report_sha256": entry["sha256"]["parity_report"],
            "observation_dim": 61,
            "action_dim": 14,
            "passed": all(case["passed"] for case in case_reports),
            "cases": case_reports,
        }
        (policy_dir / "report.json").write_text(json.dumps(policy_report, indent=2) + "\n")
        reports.append(policy_report)
    summary = {
        "schema_version": 1,
        "mode": "smoke" if args.smoke else "full",
        "manifest": str(args.manifest),
        "seed": args.seed,
        "control_rate_hz": CONTROL_HZ,
        "physics_steps_per_action": PHYSICS_STEPS,
        "command_ema_alpha": EMA_ALPHA,
        "profiles": {name: str(path) for name, path in PROFILE_SCENES.items()},
        "passed": all(report["passed"] for report in reports),
        "policies": reports,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/specialist_artifact_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=sorted(POLICY_PROFILES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.duration_seconds <= 0 or not math.isfinite(args.duration_seconds):
        raise SystemExit("--duration-seconds must be finite and positive")
    summary = run_battery(args)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
