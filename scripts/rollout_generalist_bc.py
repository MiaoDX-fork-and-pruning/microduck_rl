#!/usr/bin/env python3
"""Run a conditioned BC checkpoint in the canonical all-collisions MuJoCo scene."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.generalist_schema import make_conditioned_observation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, PolicyInference
TILT_FAILURE_RAD = np.deg2rad(65.0)


def run_case(model, student, teacher_onnx: Path, normalizer_checkpoint: Path, behavior: str, speed: float, ticks: int) -> dict:
    data = mujoco.MjData(model)
    helper = PolicyInference(model, data, walking_onnx_path=str(teacher_onnx), new_cmd_obs=True,
                             use_projected_gravity=True)
    helper.command = np.array([speed, 0.0, 0.0] + [0.0] * 10, dtype=np.float32)
    import torch
    state = torch.load(normalizer_checkpoint, weights_only=False, map_location="cpu")["actor_state_dict"]
    mean = state["obs_normalizer._mean"].numpy().reshape(61)
    std = state["obs_normalizer._std"].numpy().reshape(61)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    free_qadr = int(model.jnt_qposadr[free_id])
    data.qpos[free_qadr:free_qadr + 3] = [0.0, 0.0, 0.125]
    data.qpos[free_qadr + 3:free_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i, qidx in enumerate(helper.joint_qpos_indices):
        data.qpos[qidx] = DEFAULT_POSE[i]
    mujoco.mj_forward(model, data)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    start = data.xpos[trunk].copy()
    tilts, actions = [], []
    finite = True
    for _ in range(ticks):
        legacy = helper.get_observations()
        normalized = (legacy - mean) / np.maximum(std, 1e-6)
        conditioned = make_conditioned_observation(normalized[None, :], helper.command[None, :], behavior)
        with torch.inference_mode():
            action = student(torch.from_numpy(conditioned)).numpy().astype(np.float32).reshape(-1)
        finite = bool(np.isfinite(legacy).all() and np.isfinite(conditioned).all() and np.isfinite(action).all())
        if not finite or action.shape != (14,):
            break
        helper.last_action = action.copy()
        helper.apply_action(action)
        for _ in range(5):
            mujoco.mj_step(model, data)
        quat = data.xquat[trunk]
        tilt = 2.0 * np.arccos(np.clip(abs(float(quat[0])), 0.0, 1.0))
        tilts.append(float(tilt)); actions.append(action)
    displacement = (data.xpos[trunk] - start).tolist()
    stable = bool(max(tilts, default=np.inf) < TILT_FAILURE_RAD)
    displacement_ok = behavior != "locomotion" or displacement[0] >= 0.005
    return {"behavior": behavior, "ticks": len(actions), "finite": finite,
            "world_displacement_m": displacement, "max_tilt_rad": max(tilts, default=None),
            "final_tilt_rad": tilts[-1] if tilts else None,
            "max_abs_action": float(np.max(np.abs(actions))) if actions else None,
            "stability_gate_rad": float(TILT_FAILURE_RAD), "stable": stable,
            "displacement_gate_m": 0.005 if behavior == "locomotion" else None,
            "passed": finite and len(actions) == ticks and stable and displacement_ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("/tmp/p2-walk-bc-100"))
    ap.add_argument("--teacher-onnx", type=Path, default=Path("artifacts/specialists/velocity_flat/policy.onnx"))
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--output", type=Path, default=Path("/tmp/p2-walk-bc-rollout.json"))
    args = ap.parse_args()
    import torch
    bundle = torch.load(args.run / "model.pt", weights_only=False)
    net = torch.nn.Sequential(torch.nn.Linear(71, 512), torch.nn.Tanh(), torch.nn.Linear(512, 256), torch.nn.Tanh(), torch.nn.Linear(256, 128), torch.nn.Tanh(), torch.nn.Linear(128, 14))
    net.load_state_dict(bundle["state_dict"]); net.eval()
    model = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
    model.opt.timestep = 0.005
    report = {"schema": bundle.get("schema"), "schema_version": bundle.get("schema_version"), "ticks": args.ticks,
              "profiles": [run_case(model, net, args.teacher_onnx, Path("artifacts/specialists/velstand_flat/checkpoint.pt"), "stand", 0.0, args.ticks),
                           run_case(model, net, args.teacher_onnx, Path("artifacts/specialists/velocity_flat/checkpoint.pt"), "locomotion", args.speed, args.ticks)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(item["passed"] for item in report["profiles"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
