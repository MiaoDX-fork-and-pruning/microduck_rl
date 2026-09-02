#!/usr/bin/env python3
"""Measure whether velocity_rollers has a fixed or reset-dependent lead leg."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import MICRODUCK_ROLLERS_XML, PolicyInference


VARIANTS = {
    "canonical": (0.0, 0.0),
    "left_leg_ahead": (-0.20, -0.20),
    "right_leg_ahead": (0.20, 0.20),
}


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def run_variant(name: str, offsets: tuple[float, float], duration_s: float, speed: float) -> dict:
    model = mujoco.MjModel.from_xml_path(str(ROOT / MICRODUCK_ROLLERS_XML))
    model.opt.timestep = 0.005
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name and joint_name.startswith("passive_"):
            model.dof_frictionloss[model.jnt_dofadr[joint_id]] = 0.003
    data = mujoco.MjData(model)
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=str(ROOT / "artifacts/specialists/velocity_rollers/policy.onnx"),
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    free_qpos = int(model.jnt_qposadr[free_id])
    data.qpos[free_qpos : free_qpos + 3] = [0.0, 0.0, 0.1385]
    data.qpos[free_qpos + 3 : free_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    for index, qpos_index in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_index] = policy.default_pose[index]
    # Servo layout: left hip pitch is 2 and right hip pitch is 11.
    data.qpos[policy.joint_qpos_indices[2]] += offsets[0]
    data.qpos[policy.joint_qpos_indices[11]] += offsets[1]
    data.ctrl[:] = data.qpos[policy.joint_qpos_indices]
    mujoco.mj_forward(model, data)
    policy.activate_specialist_policy("velocity_rollers", [speed, 0.0, 0.0] + [0.0] * 10)

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    left_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_l_v1")
    right_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ankle_r_v1")
    wheel_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        for joint_name in ("passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel")
    ]
    wheel_dofs = [int(model.jnt_dofadr[joint_id]) for joint_id in wheel_joint_ids]
    start_position = data.xpos[trunk_id].copy()
    start_yaw = math.atan2(
        2.0 * (data.xquat[trunk_id, 0] * data.xquat[trunk_id, 3] + data.xquat[trunk_id, 1] * data.xquat[trunk_id, 2]),
        1.0 - 2.0 * (data.xquat[trunk_id, 2] ** 2 + data.xquat[trunk_id, 3] ** 2),
    )
    lead_delta = []
    left_contacts = 0
    right_contacts = 0
    wheel_speeds = []
    tilts = []
    ticks = int(duration_s * 50)
    for _ in range(ticks):
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(4):
            mujoco.mj_step(model, data)
        policy.last_action = np.asarray(action, dtype=np.float32).copy()
        lead_delta.append(float(data.xpos[left_ankle, 0] - data.xpos[right_ankle, 0]))
        wheel_speeds.append(np.abs(data.qvel[wheel_dofs]).tolist())
        quat = data.xquat[trunk_id]
        tilts.append(float(2.0 * math.acos(np.clip(abs(float(quat[0])), 0.0, 1.0))))
        left_touch = False
        right_touch = False
        for contact in data.contact[: data.ncon]:
            for geom_id in (int(contact.geom1), int(contact.geom2)):
                body_id = int(model.geom_bodyid[geom_id])
                left_touch |= _is_descendant(model, body_id, left_ankle)
                right_touch |= _is_descendant(model, body_id, right_ankle)
        left_contacts += int(left_touch)
        right_contacts += int(right_touch)
    end_yaw = math.atan2(
        2.0 * (data.xquat[trunk_id, 0] * data.xquat[trunk_id, 3] + data.xquat[trunk_id, 1] * data.xquat[trunk_id, 2]),
        1.0 - 2.0 * (data.xquat[trunk_id, 2] ** 2 + data.xquat[trunk_id, 3] ** 2),
    )
    settled = lead_delta[min(100, len(lead_delta)) :]
    return {
        "variant": name,
        "hip_pitch_offsets_rad": list(offsets),
        "duration_s": duration_s,
        "command_x": speed,
        "initial_lead_delta_m": lead_delta[0],
        "settled_mean_lead_delta_m": float(np.mean(settled)),
        "settled_left_ahead_fraction": float(np.mean(np.asarray(settled) > 0.0)),
        "left_contact_fraction": left_contacts / ticks,
        "right_contact_fraction": right_contacts / ticks,
        "mean_abs_wheel_speed_rad_s": np.mean(np.asarray(wheel_speeds), axis=0).tolist(),
        "yaw_change_rad": float(math.atan2(math.sin(end_yaw - start_yaw), math.cos(end_yaw - start_yaw))),
        "world_displacement_m": (data.xpos[trunk_id] - start_position).tolist(),
        "max_tilt_rad": max(tilts),
        "finite": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [run_variant(name, offsets, args.duration_seconds, args.speed) for name, offsets in VARIANTS.items()]
    left_ahead = [result["settled_left_ahead_fraction"] for result in results]
    report = {
        "schema_version": 1,
        "policy_id": "velocity_rollers",
        "variants": results,
        "persistent_fixed_lead": all(value >= 0.8 for value in left_ahead) or all(value <= 0.2 for value in left_ahead),
        "interpretation": "persistent learned lead bias" if all(value >= 0.8 for value in left_ahead) or all(value <= 0.2 for value in left_ahead) else "lead posture depends on reset phase",
        "passed": all(result["finite"] and result["max_tilt_rad"] < math.radians(65) for result in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
