#!/usr/bin/env python3
"""Run a finite, headless MuJoCo battery against an IsaacLab ONNX export.

This is a deployment-side rehearsal, not a replacement for the IsaacLab
training battery.  It uses the canonical six command cases and records body
frame velocity, tilt, finite-state, and ABI evidence without opening a viewer.
MuJoCo and PhysX trajectory equality is intentionally not assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer_policy import DEFAULT_POSE, PolicyInference  # noqa: E402
from scripts.isaaclab.velocity_flat_battery_spec import CASES, CONTROL_HZ, evaluate_case  # noqa: E402


SCENE = ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"
PHYSICS_STEPS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tilt_rad(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> float:
    rotation = data.xmat[body_id].reshape(3, 3)
    return math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))


def _reset(model: mujoco.MjModel, data: mujoco.MjData, policy: PolicyInference) -> None:
    mujoco.mj_resetData(model, data)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[free_id])
    data.qpos[qpos_adr : qpos_adr + 3] = [0.0, 0.0, 0.125]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    for index, qpos_index in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_index] = DEFAULT_POSE[index]
    data.ctrl[:] = DEFAULT_POSE
    policy.last_action[:] = 0.0
    policy.command[:] = 0.0
    mujoco.mj_forward(model, data)


def _run_case(onnx_path: Path, case, seed: int, steps: int) -> dict:
    np.random.seed(seed)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=str(onnx_path),
        new_cmd_obs=True,
        use_projected_gravity=True,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    policy.ort_session = session
    policy.input_name = session.get_inputs()[0].name
    policy.output_name = session.get_outputs()[0].name
    _reset(model, data, policy)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qvel_adr = int(model.jnt_dofadr[free_id])
    initial_position = data.xpos[body_id].copy()
    xy_velocity: list[np.ndarray] = []
    yaw_velocity: list[float] = []
    max_tilt = 0.0
    finite = True

    command = np.zeros(13, dtype=np.float32)
    command[:3] = np.asarray(case.command, dtype=np.float32)
    for _ in range(steps):
        policy.command = command.copy()
        observation = policy.get_observations()
        action = np.asarray(policy.infer(), dtype=np.float32)
        finite = finite and bool(
            np.isfinite(observation).all()
            and observation.shape == (61,)
            and np.isfinite(action).all()
            and action.shape == (14,)
            and np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
        )
        if not finite:
            break
        policy.apply_action(action)
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(model, data)
        quat_wxyz = data.xquat[body_id].astype(np.float32)
        world_linear = data.qvel[qvel_adr : qvel_adr + 3].astype(np.float32)
        body_linear = policy.quat_rotate_inverse(quat_wxyz, world_linear)
        xy_velocity.append(body_linear[:2])
        yaw_velocity.append(float(data.qvel[qvel_adr + 5]))
        max_tilt = max(max_tilt, _tilt_rad(model, data, body_id))

    actual_xy = np.mean(np.asarray(xy_velocity), axis=0) if xy_velocity else np.zeros(2)
    actual_yaw = float(np.mean(yaw_velocity)) if yaw_velocity else 0.0
    reset_fraction = 0.0
    passed, failures = evaluate_case(
        case,
        finite=finite,
        mean_actual_xy=actual_xy,
        mean_actual_yaw=actual_yaw,
        max_tilt=max_tilt,
        reset_fraction=reset_fraction,
    )
    return {
        "name": case.name,
        "command": list(case.command),
        "steps": len(xy_velocity),
        "finite": finite,
        "actor_obs_dim": 61,
        "action_dim": 14,
        "mean_actual_vel_xy_m_s": [float(value) for value in actual_xy],
        "mean_actual_vel_yaw_rad_s": actual_yaw,
        "world_displacement_m": (data.xpos[body_id] - initial_position).tolist(),
        "max_tilt_rad": max_tilt,
        "episode_resets": 0,
        "passed": passed,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX file does not exist: {args.onnx}")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    cases = [_run_case(args.onnx, case, args.seed + index, args.steps) for index, case in enumerate(CASES)]
    report = {
        "schema": "mujoco_onnx_battery.v1",
        "backend": "mujoco",
        "scene": str(SCENE),
        "onnx": str(args.onnx),
        "onnx_sha256": _sha256(args.onnx),
        "seed": args.seed,
        "steps_per_case": args.steps,
        "control_rate_hz": CONTROL_HZ,
        "physics_steps_per_action": PHYSICS_STEPS,
        "friction_bridge": "mjlab_bam_runtime",
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
