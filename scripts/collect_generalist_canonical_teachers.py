#!/usr/bin/env python3
"""Collect exact canonical G0 teacher trajectories for shared-policy training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_generalist_g0 as g0
from infer_policy import DEFAULT_POSE, PolicyInference
from mjlab_microduck.generalist_schema import make_conditioned_observation, validate_batch


TEACHERS = {
    "stand": Path("artifacts/specialists/velstand_flat/policy.onnx"),
    "locomotion": Path("artifacts/specialists/velocity_flat/policy.onnx"),
    "sit_stand": Path("artifacts/specialists/sitstand_flat/policy.onnx"),
}


def _reset(model: mujoco.MjModel, data: mujoco.MjData, helper: PolicyInference) -> None:
    mujoco.mj_resetData(model, data)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[free_id])
    data.qpos[qadr:qadr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    for index, qidx in enumerate(helper.joint_qpos_indices):
        data.qpos[qidx] = DEFAULT_POSE[index]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    helper.last_action = np.zeros(14, dtype=np.float32)
    mujoco.mj_forward(model, data)


def _action(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    action = np.asarray(session.run(None, {name: obs[None, :]})[0], dtype=np.float32).reshape(-1)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError("teacher action is not finite [14]")
    return action


def collect(reference_onnx: Path) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
    model.opt.timestep = 0.005
    inputs: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    trajectory_ids: list[int] = []
    segment_ids: list[str] = []
    trajectory = 0
    cases = [(name, segments) for name, segments in g0.BEHAVIOR_SEGMENTS.items()]
    cases += [(f"{source}->{destination}", segments)
              for (source, destination), segments in sorted(g0.EDGE_SEGMENTS.items())]
    for case_name, segments in cases:
        data = mujoco.MjData(model)
        helper = PolicyInference(model, data, walking_onnx_path=str(reference_onnx),
                                 new_cmd_obs=True, use_projected_gravity=True)
        _reset(model, data, helper)
        for segment_index, segment in enumerate(segments):
            behavior = g0.STATE_TO_BEHAVIOR[segment.state]
            session = ort.InferenceSession(str(TEACHERS[behavior]), providers=["CPUExecutionProvider"])
            command = g0._command(segment.state, segment.command_x)
            helper.command = command
            for tick in range(segment.ticks):
                legacy = helper.get_observations()
                conditioned = make_conditioned_observation(
                    legacy[None, :], command[None, :], behavior,
                    phase=np.array([[tick / max(segment.ticks - 1, 1)
                                     if segment.active_transition else 0.0,
                                     float(segment.active_transition)]], dtype=np.float32),
                    posture=np.array([[segment.command_x if segment.state == "SITSTAND" else 0.0]], dtype=np.float32),
                )[0]
                inputs.append(conditioned)
                actions.append(_action(session, legacy))
                trajectory_ids.append(trajectory)
                segment_ids.append(f"{case_name}:{segment_index}:{segment.state}")
                helper.last_action = actions[-1].copy()
                helper.apply_action(actions[-1])
                for _ in range(4):
                    mujoco.mj_step(model, data)
        trajectory += 1
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32)
    validate_batch(x, y)
    return {"inputs": x, "actions": y,
            "trajectory_ids": np.asarray(trajectory_ids, dtype=np.int64),
            "segment_ids": np.asarray(segment_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.reference_onnx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(json.dumps({"samples": int(len(data["inputs"])),
                      "trajectories": int(len(np.unique(data["trajectory_ids"]))),
                      "segments": sorted(set(data["segment_ids"].tolist())),
                      "action_outside_unit_fraction": float(np.mean(np.abs(data["actions"]) > 1.0)),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
