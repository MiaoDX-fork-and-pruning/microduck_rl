#!/usr/bin/env python3
"""Capture real specialist fall/recovery states from deterministic MuJoCo rollouts."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from run_specialist_action_battery import (
    CONTROL_HZ,
    DEFAULT_POSE,
    PHYSICS_STEPS,
    POLICY_PROFILES,
    PolicyInference,
    _load_policy,
)


def capture(onnx: Path, scene: Path, seed: int, ticks: int, impulse_tick: int, impulse: float, axis: str):
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be x, y, or z")
    np.random.seed(seed)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = _load_policy(model, data, onnx)
    mujoco.mj_resetData(model, data)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[free_id])
    vadr = int(model.jnt_dofadr[free_id])
    data.qpos[qadr : qadr + 3] = [0.0, 0.0, 0.125]
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for index, qpos_index in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_index] = DEFAULT_POSE[index]
    data.ctrl[:] = DEFAULT_POSE
    policy.last_action[:] = 0.0
    policy.command[:] = 0.0
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    rows: list[dict[str, np.ndarray | float | int | bool]] = []
    perturbation_applied = False
    for tick in range(ticks):
        if tick == impulse_tick:
            angular = np.zeros(3)
            angular["xyz".index(axis)] = impulse
            data.qvel[vadr + 3 : vadr + 6] += angular
            perturbation_applied = True
        observation = policy.get_observations()
        action = np.asarray(policy.infer(), dtype=np.float32)
        policy.apply_action(action)
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        rows.append({
            "tick": tick,
            "observation": observation.copy(),
            "raw_action": action.copy(),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "command": policy.command.copy(),
            "tilt_rad": tilt,
            "height_m": float(data.xpos[body_id, 2]),
            "perturbation_applied": perturbation_applied,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--impulse-tick", type=int, default=200)
    parser.add_argument("--impulse", type=float, default=3.0)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/specialists/velstand_flat/policy.onnx"))
    parser.add_argument("--scene", type=Path, default=Path("src/mjlab_microduck/robot/microduck/scene.xml"))
    args = parser.parse_args()
    if args.impulse_tick < 0 or args.impulse_tick >= args.ticks:
        raise ValueError("impulse tick must be inside rollout")
    rows = capture(args.onnx, args.scene, args.seed, args.ticks, args.impulse_tick, args.impulse, args.axis)
    finite = all(np.isfinite(np.asarray(row["qpos"])).all() and np.isfinite(np.asarray(row["qvel"])).all() for row in rows)
    fallen = [row for row in rows if float(row["tilt_rad"]) > np.deg2rad(40.0)]
    recovered = [row for row in rows if fallen and row["tick"] > fallen[0]["tick"] and float(row["tilt_rad"]) < np.deg2rad(25.0)]
    payload = {
        "schema": "specialist-recovery-trace",
        "version": 1,
        "teacher": "velstand_flat",
        "seed": args.seed,
        "control_hz": CONTROL_HZ,
        "physics_steps_per_action": PHYSICS_STEPS,
        "ticks": args.ticks,
        "impulse_tick": args.impulse_tick,
        "impulse_rad_s": args.impulse,
        "impulse_axis": args.axis,
        "finite": finite,
        "fall_tick": int(fallen[0]["tick"]) if fallen else None,
        "recovery_tick": int(recovered[0]["tick"]) if recovered else None,
        "rows": rows,
    }
    if not fallen:
        raise RuntimeError("perturbation did not produce a real fallen state; adjust only the capture parameters")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: np.asarray([row[key] for row in rows]) for key in rows[0]})
    print({"output": str(args.output), "fall_tick": payload["fall_tick"], "recovery_tick": payload["recovery_tick"], "finite": finite})


if __name__ == "__main__":
    main()
