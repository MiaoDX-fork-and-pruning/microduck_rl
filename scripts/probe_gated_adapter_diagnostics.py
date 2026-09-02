#!/usr/bin/env python3
"""Diagnose a failed shared G0 gated-adapter actor without changing training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.generalist_model import build_actor
from mjlab_microduck.generalist_schema import make_conditioned_observation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_generalist_g0 as g0
from infer_policy import DEFAULT_POSE, PolicyInference


BEHAVIOR_NAMES = ("stand", "locomotion", "sit_stand")
STATE_TO_TEACHER = {
    "VELSTAND": "velstand_flat",
    "VELOCITY": "velocity_flat",
    "SITSTAND": "sitstand_flat",
}


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


def _load_model(run: Path):
    import torch

    manifest = json.loads((run / "manifest.json").read_text())
    metadata = manifest.get("metrics", manifest)
    model = build_actor(metadata)
    bundle = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    return model, metadata


def _onnx_actions(session, observations: np.ndarray) -> np.ndarray:
    """Run both static-batch and dynamic-batch frozen teacher graphs."""
    input_name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    if shape[0] in (None, "batch"):
        return np.asarray(session.run(None, {input_name: observations})[0], dtype=np.float32)
    return np.concatenate([
        np.asarray(session.run(None, {input_name: row[None, :]})[0], dtype=np.float32)
        for row in observations
    ], axis=0)


def offline_probe(run: Path, teacher_manifest: Path) -> dict:
    import onnxruntime as ort
    import torch

    data = np.load(run / "dataset.npz", allow_pickle=False)
    inputs = data["inputs"].astype(np.float32)
    targets = data["actions"].astype(np.float32)
    model, _ = _load_model(run)
    with torch.inference_mode():
        student = model(torch.from_numpy(inputs)).numpy()
        gate = torch.softmax(model.gate(torch.from_numpy(inputs[:, 48:54])), dim=-1).numpy()

    manifest = json.loads(teacher_manifest.read_text())
    sessions = {
        item["id"]: ort.InferenceSession(item["artifacts"]["onnx"], providers=["CPUExecutionProvider"])
        for item in manifest["teachers"]
    }
    labels = inputs[:, 48:54].argmax(axis=1)
    teacher_names = ("velstand_flat", "velocity_flat", "sitstand_flat")
    legacy = np.concatenate((inputs[:, :48], inputs[:, 54:67]), axis=1)
    teacher_actions = np.empty_like(targets)
    for index, name in enumerate(teacher_names):
        mask = labels == index
        if mask.any():
            session = sessions[name]
            teacher_actions[mask] = _onnx_actions(session, legacy[mask])

    result = {"samples": int(len(inputs)), "by_behavior": {}, "gate": {}}
    for index, name in enumerate(BEHAVIOR_NAMES):
        mask = labels == index
        target = targets[mask]
        pred = student[mask]
        teacher = teacher_actions[mask]
        result["by_behavior"][name] = {
            "samples": int(mask.sum()),
            "student_target_mse": float(np.mean((pred - target) ** 2)),
            "student_teacher_mse": float(np.mean((pred - teacher) ** 2)),
            "teacher_target_max_abs_error": float(np.max(np.abs(teacher - target))),
            "target_outside_unit_fraction": float(np.mean(np.abs(target) > 1.0)),
            "teacher_outside_unit_fraction": float(np.mean(np.abs(teacher) > 1.0)),
            "student_saturated_fraction": float(np.mean(np.abs(pred) >= 0.995)),
            "target_action_abs": _summary(np.abs(target)),
            "student_action_abs": _summary(np.abs(pred)),
        }
        result["gate"][name] = {
            "mean": gate[mask].mean(axis=0).tolist(),
            "std": gate[mask].std(axis=0).tolist(),
            "min": gate[mask].min(axis=0).tolist(),
            "max": gate[mask].max(axis=0).tolist(),
            "entropy_mean": float(np.mean(-np.sum(gate[mask] * np.log(np.maximum(gate[mask], 1e-12)), axis=1))),
        }
    return result


def _canonical_case(model, policy, reference_onnx: Path, state: str,
                    segments: tuple[g0.Segment, ...], train_inputs: np.ndarray) -> dict:
    data = mujoco.MjData(model)
    helper = PolicyInference(model, data, walking_onnx_path=str(reference_onnx),
                             new_cmd_obs=True, use_projected_gravity=True)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[free_id])
    data.qpos[qadr:qadr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    for index, qidx in enumerate(helper.joint_qpos_indices):
        data.qpos[qidx] = DEFAULT_POSE[index]
    mujoco.mj_forward(model, data)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    records = []
    for segment in segments:
        command = g0._command(state, segment.command_x)
        helper.command = command
        behavior = g0.STATE_TO_BEHAVIOR[state]
        for tick in range(segment.ticks):
            legacy = helper.get_observations()
            conditioned = make_conditioned_observation(
                legacy[None, :], command[None, :], behavior,
                phase=np.array([[tick / max(segment.ticks - 1, 1) if segment.active_transition else 0.0,
                                 float(segment.active_transition)]], dtype=np.float32),
                posture=np.array([[segment.command_x if state == "SITSTAND" else 0.0]], dtype=np.float32),
            )
            action = policy(conditioned)
            helper.last_action = action.copy()
            helper.apply_action(action)
            for _ in range(4):
                mujoco.mj_step(model, data)
            quat = data.xquat[trunk]
            tilt = 2.0 * np.arccos(np.clip(abs(float(quat[0])), 0.0, 1.0))
            records.append({
                "state": state,
                "segment_tick": tick,
                "tilt_rad": float(tilt),
                "height_m": float(data.xpos[trunk, 2]),
                "max_abs_action": float(np.max(np.abs(action))),
                "conditioned": conditioned[0],
            })
            if not np.isfinite(action).all():
                break
    conditioned = np.asarray([item.pop("conditioned") for item in records], dtype=np.float32)
    label = g0.STATE_TO_BEHAVIOR[state]
    mask = train_inputs[:, 48:54].argmax(axis=1) == BEHAVIOR_NAMES.index(label)
    train_slice = train_inputs[mask]
    outside = (conditioned < train_slice.min(axis=0)) | (conditioned > train_slice.max(axis=0))
    first_tilt = next((
        {"segment_tick": item["segment_tick"], "tilt_rad": item["tilt_rad"]}
        for item in records if item["tilt_rad"] >= np.deg2rad(65.0)
    ), None)
    return {
        "steps": len(records),
        "first_tilt_gate_breach": first_tilt,
        "max_tilt_rad": max(item["tilt_rad"] for item in records),
        "max_action": max(item["max_abs_action"] for item in records),
        "input_outside_training_minmax_fraction": float(outside.mean()),
        "input_outside_training_minmax_by_feature": outside.mean(axis=0).tolist(),
        "input_mean": conditioned.mean(axis=0).tolist(),
        "input_std": conditioned.std(axis=0).tolist(),
    }


def canonical_probe(run: Path, reference_onnx: Path, train_inputs: np.ndarray) -> dict:
    model = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
    model.opt.timestep = 0.005
    policy = g0.Policy(run=run)
    cases = {}
    for state, segments in g0.BEHAVIOR_SEGMENTS.items():
        cases[state] = _canonical_case(model, policy, reference_onnx, state, segments, train_inputs)
    for edge, segments in g0.EDGE_SEGMENTS.items():
        cases[f"{edge[0]}->{edge[1]}"] = _canonical_case(model, policy, reference_onnx, edge[1], segments, train_inputs)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-onnx", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path,
                        default=Path("docs/plans/generalist-g0-teacher-manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_data = np.load(args.run / "dataset.npz", allow_pickle=False)
    report = {
        "schema": "generalist-g0-gated-adapter-diagnostic",
        "schema_version": 1,
        "run": str(args.run),
        "offline": offline_probe(args.run, args.teacher_manifest),
        "canonical": canonical_probe(args.run, args.reference_onnx, train_data["inputs"].astype(np.float32)),
        "interpretation": {
            "tilt_gate_rad": float(np.deg2rad(65.0)),
            "input_coverage_is_train_minmax_only": True,
            "canonical_cases_use_the_same_50hz_segments_as_evaluate_generalist_g0": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
