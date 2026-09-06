#!/usr/bin/env python3
"""Collect teacher labels on states visited by one shared G0 student actor.

This is a bounded coverage probe. It does not train or modify the actor; the
resulting shard can be passed to ``train_generalist_bc.py --extra-data``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from mjlab_microduck.generalist_model import build_actor
from mjlab_microduck.generalist_schema import make_conditioned_observation, validate_batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_generalist_g0 as g0
from infer_policy import DEFAULT_POSE, PolicyInference


TEACHER_PATHS = {
    "stand": Path("artifacts/specialists/velstand_flat/policy.onnx"),
    "locomotion": Path("artifacts/specialists/velocity_flat/policy.onnx"),
    "sit_stand": Path("artifacts/specialists/sitstand_flat/policy.onnx"),
}


def _load_student(run: Path):
    import torch

    manifest = json.loads((run / "manifest.json").read_text())
    metadata = manifest.get("metrics", manifest)
    model = build_actor(metadata)
    bundle = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    return model


def _teacher_sessions() -> dict[str, ort.InferenceSession]:
    sessions = {}
    for behavior, path in TEACHER_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        sessions[behavior] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sessions


def _teacher_action(session: ort.InferenceSession, observation: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: observation[None, :]})[0]
    action = np.asarray(result, dtype=np.float32).reshape(-1)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError(f"teacher emitted invalid action shape/values: {action.shape}")
    return action


def _reset(model: mujoco.MjModel, data: mujoco.MjData, helper: PolicyInference) -> None:
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[free_id])
    data.qpos[qadr:qadr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    for index, qidx in enumerate(helper.joint_qpos_indices):
        data.qpos[qidx] = DEFAULT_POSE[index]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    helper.last_action = np.zeros(14, dtype=np.float32)
    mujoco.mj_forward(model, data)


def collect(run: Path, reference_onnx: Path, seed: int = 42, beta: float = 0.9) -> dict[str, np.ndarray]:
    del seed  # Canonical segments are deterministic; retained in the manifest.
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")
    import torch

    student = _load_student(run)
    sessions = _teacher_sessions()
    model = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
    model.opt.timestep = 0.005
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    behavior_ids: list[int] = []
    segment_ids: list[str] = []
    tilt_values: list[float] = []

    for case_name, segments in {
        **g0.BEHAVIOR_SEGMENTS,
        **{f"{source}->{destination}": value for (source, destination), value in g0.EDGE_SEGMENTS.items()},
    }.items():
        data = mujoco.MjData(model)
        helper = PolicyInference(model, data, walking_onnx_path=str(reference_onnx),
                                 new_cmd_obs=True, use_projected_gravity=True)
        _reset(model, data, helper)
        for segment in segments:
            state = segment.state
            behavior = g0.STATE_TO_BEHAVIOR[state]
            command = g0._command(state, segment.command_x)
            helper.command = command
            for tick in range(segment.ticks):
                legacy = helper.get_observations()
                conditioned = make_conditioned_observation(
                    legacy[None, :], command[None, :], behavior,
                    phase=np.array([[tick / max(segment.ticks - 1, 1) if segment.active_transition else 0.0,
                                     float(segment.active_transition)]], dtype=np.float32),
                    posture=np.array([[segment.command_x if state == "SITSTAND" else 0.0]], dtype=np.float32),
                )[0]
                teacher = _teacher_action(sessions[behavior], legacy)
                with torch.inference_mode():
                    student_action = student(torch.from_numpy(conditioned[None, :])).numpy()[0].astype(np.float32)
                if not np.isfinite(student_action).all():
                    raise ValueError(f"student became non-finite in {case_name} tick {tick}")
                xs.append(conditioned)
                ys.append(teacher)
                behavior_ids.append(("stand", "locomotion", "sit_stand").index(behavior))
                segment_ids.append(case_name)
                # Keep the rollout near the teacher manifold while still
                # exposing the student to its own induced observations.
                applied_action = beta * teacher + (1.0 - beta) * student_action
                helper.last_action = applied_action.copy()
                helper.apply_action(applied_action)
                for _ in range(4):
                    mujoco.mj_step(model, data)
                trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
                quat = data.xquat[trunk]
                tilt_values.append(float(2.0 * np.arccos(np.clip(abs(float(quat[0])), 0.0, 1.0))))

    inputs = np.asarray(xs, dtype=np.float32)
    actions = np.asarray(ys, dtype=np.float32)
    validate_batch(inputs, actions)
    return {
        "inputs": inputs,
        "actions": actions,
        "behavior_ids": np.asarray(behavior_ids, dtype=np.int64),
        "segment_ids": np.asarray(segment_ids),
        "tilt_rad": np.asarray(tilt_values, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--reference-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.9,
                        help="teacher mixture weight during state collection")
    args = parser.parse_args()
    data = collect(args.student_run, args.reference_onnx, args.seed, args.beta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    labels = data["behavior_ids"]
    report = {
        "schema": "generalist-g0-student-state-probe",
        "schema_version": 1,
        "student_run": str(args.student_run),
        "reference_onnx": str(args.reference_onnx),
        "seed": args.seed,
        "beta": args.beta,
        "physics_steps_per_control_tick": 4,
        "samples": int(len(labels)),
        "samples_by_behavior": {name: int(np.sum(labels == index)) for index, name in enumerate(("stand", "locomotion", "sit_stand"))},
        "max_visited_tilt_rad": float(np.max(data["tilt_rad"])),
        "visited_tilt_gate_breach_fraction": float(np.mean(data["tilt_rad"] >= np.deg2rad(65.0))),
        "output": str(args.output),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
