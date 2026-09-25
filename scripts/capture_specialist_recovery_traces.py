#!/usr/bin/env python3
"""Capture replayable CPU MuJoCo perturbation diagnostics, not acceptance resets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from run_specialist_action_battery import (
    CONTROL_HZ, PHYSICS_STEPS, PROFILE_SCENES, _load_policy, reset_pose,
    sha256, trunk_metrics,
)

TEACHER_ID = "velstand_flat"
SCENE = PROFILE_SCENES["walk_all_collisions"]
TEACHER_MANIFEST = Path("docs/plans/generalist-g0-teacher-manifest.json")
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def model_sha256(model: mujoco.MjModel) -> str:
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def load_teacher(onnx: Path, scene: Path):
    manifest = json.loads(TEACHER_MANIFEST.read_text())
    teacher = next(row for row in manifest["teachers"] if row["id"] == TEACHER_ID)
    if sha256(onnx) != teacher["sha256"]["onnx"]:
        raise ValueError("teacher ONNX does not match the frozen manifest")
    if scene.resolve() != SCENE.resolve():
        raise ValueError("recovery diagnostics require the G0 all-collisions scene")
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy = _load_policy(model, data, onnx)
    if policy.use_delay:
        raise ValueError("this diagnostic contract does not support action delays")
    return model, data, policy


def advance(model, data, policy, action) -> None:
    policy.apply_action(action)
    for _ in range(PHYSICS_STEPS):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def capture(onnx: Path, scene: Path, seed: int, ticks: int,
            impulse_tick: int, impulse: float, axis: str) -> dict:
    if ticks < 1 or not 0 <= impulse_tick < ticks:
        raise ValueError("impulse tick must be inside a positive rollout")
    if axis not in {"x", "y", "z"} or not np.isfinite(impulse):
        raise ValueError("impulse must be finite and axis must be x, y, or z")
    np.random.seed(seed)
    model, data, policy = load_teacher(onnx, scene)
    reset_pose(TEACHER_ID, "canonical", model, data, policy)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    vadr = int(model.jnt_dofadr[free_id])
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    rows = []
    for tick in range(ticks):
        if tick == impulse_tick:
            data.qvel[vadr + 3 + "xyz".index(axis)] += impulse
            mujoco.mj_forward(model, data)
        state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        metrics = trunk_metrics(model, data, body_id)
        row = {
            "tick": tick, "integration_state": state,
            "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "previous_action": policy.last_action.copy(),
            "command": policy.command.copy(),
            "observation": policy.get_observations().copy(),
            "tilt_rad": metrics["tilt_rad"], "height_m": metrics["height_m"],
        }
        action = policy.infer()
        row["raw_action"] = action.copy()
        advance(model, data, policy, action)
        row["next_qpos"] = data.qpos.copy()
        row["next_qvel"] = data.qvel.copy()
        rows.append(row)
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("non-finite recovery trace")
    fallen = np.flatnonzero(arrays["tilt_rad"] > np.deg2rad(40))
    fall = int(fallen[0]) if len(fallen) else None
    recovery = np.flatnonzero(
        (arrays["tick"] > (fall if fall is not None else ticks))
        & (arrays["tilt_rad"] < np.deg2rad(25)) & (arrays["height_m"] > 0.09)
    )
    metadata = {
        "schema": "specialist-recovery-trace", "version": 2,
        "evidence_class": "cpu_perturbation_diagnostic",
        "accepted_reset_equivalence": False,
        "harness": "PolicyInference; position actuators; no training managers",
        "sample_boundary": "pre_action_after_forward_and_impulse",
        "teacher": TEACHER_ID, "onnx": str(onnx.resolve()), "onnx_sha256": sha256(onnx),
        "teacher_manifest_sha256": sha256(TEACHER_MANIFEST),
        "scene": str(scene.resolve()), "model_sha256": model_sha256(model),
        "mujoco_version": mujoco.__version__, "state_spec": int(STATE_SPEC),
        "seed": seed, "control_hz": CONTROL_HZ, "physics_steps_per_action": PHYSICS_STEPS,
        "ticks": ticks, "impulse_tick": impulse_tick, "impulse_rad_s": impulse,
        "impulse_axis": axis, "action_delay": False, "action_filter": False,
        "action_scale": policy.action_scale, "action_clipping": "none",
        "termination": "fixed diagnostic horizon; no automatic resets", "finite": True,
        "raw_action_max_abs": float(np.abs(arrays["raw_action"]).max()),
        "raw_action_within_unit_range": bool(np.abs(arrays["raw_action"]).max() <= 1.00001),
        "fall_tick": fall, "recovery_tick": int(recovery[0]) if len(recovery) else None,
    }
    return {**arrays, "metadata": np.asarray(json.dumps(metadata, sort_keys=True))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--impulse-tick", type=int, default=200); parser.add_argument("--impulse", type=float, default=3.0)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/specialists/velstand_flat/policy.onnx"))
    parser.add_argument("--scene", type=Path, default=SCENE); args = parser.parse_args()
    trace = capture(args.onnx, args.scene, args.seed, args.ticks, args.impulse_tick, args.impulse, args.axis)
    args.output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output, **trace)
    metadata = json.loads(trace["metadata"].item())
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
