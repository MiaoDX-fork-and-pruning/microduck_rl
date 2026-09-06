#!/usr/bin/env python3
"""Replay a captured real fall state through the immutable specialist ONNX."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from run_specialist_action_battery import DEFAULT_POSE, PHYSICS_STEPS, _load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/specialists/velstand_flat/policy.onnx"))
    parser.add_argument("--scene", type=Path, default=Path("src/mjlab_microduck/robot/microduck/scene.xml"))
    args = parser.parse_args()
    trace = np.load(args.trace)
    tilt = np.asarray(trace["tilt_rad"])
    fall_candidates = np.flatnonzero(tilt > math.radians(40.0))
    if len(fall_candidates) == 0:
        raise ValueError("trace contains no fallen state")
    source_tick = int(fall_candidates[0])
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = _load_policy(model, data, args.onnx)
    data.qpos[:] = np.asarray(trace["qpos"])[source_tick]
    data.qvel[:] = np.asarray(trace["qvel"])[source_tick]
    data.ctrl[:] = DEFAULT_POSE
    policy.last_action[:] = 0.0
    policy.command = np.asarray(trace["command"])[source_tick].astype(np.float32)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    tilts, heights, actions = [], [], []
    finite = True
    for _ in range(args.ticks):
        observation = policy.get_observations()
        action = np.asarray(policy.infer(), dtype=np.float32)
        finite &= bool(np.isfinite(observation).all() and np.isfinite(action).all() and np.abs(action).max() <= 1.00001)
        actions.append(action.copy())
        policy.apply_action(action)
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        tilts.append(float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        heights.append(float(data.xpos[body_id, 2]))
        finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    payload = {
        "schema": "specialist-recovery-replay",
        "version": 1,
        "trace": str(args.trace),
        "source_tick": source_tick,
        "ticks": args.ticks,
        "finite": finite,
        "initial_tilt_rad": float(tilt[source_tick]),
        "max_tilt_rad": max(tilts, default=None),
        "final_tilt_rad": tilts[-1] if tilts else None,
        "initial_height_m": float(trace["height_m"][source_tick]),
        "final_height_m": heights[-1] if heights else None,
        "max_abs_action": float(np.abs(actions).max()) if actions else None,
        "recovered": bool(any(t < math.radians(25.0) for t in tilts)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
