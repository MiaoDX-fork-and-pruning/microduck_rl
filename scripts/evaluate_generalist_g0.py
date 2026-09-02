#!/usr/bin/env python3
"""Run the canonical three-behavior/four-edge G0 CPU MuJoCo battery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.generalist_g0_evaluation import STATE_TO_BEHAVIOR, TraceMetrics, make_report
from mjlab_microduck.generalist_model import build_actor
from mjlab_microduck.generalist_schema import make_conditioned_observation
from mjlab_microduck.generalist_transition_graph import LEGAL_EDGES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, PolicyInference


class Policy:
    def __init__(self, run: Path | None, onnx: Path | None):
        if (run is None) == (onnx is None):
            raise ValueError("choose exactly one of --run or --onnx")
        self.backend = "onnx" if onnx else "pytorch"
        if onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            if len(inputs) != 1 or inputs[0].shape[-1] != 71:
                raise ValueError("G0 ONNX must expose one 71D input")
            self.input_name = inputs[0].name
        else:
            import torch
            bundle = torch.load(run / "model.pt", weights_only=False)
            metadata = json.loads((run / "manifest.json").read_text()).get("metrics", {})
            self.model = build_actor(metadata)
            self.model.load_state_dict(bundle["state_dict"])
            self.model.eval()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            return np.asarray(self.session.run(None, {self.input_name: observation})[0][0], dtype=np.float32)
        import torch
        with torch.inference_mode():
            return self.model(torch.from_numpy(observation)).numpy()[0].astype(np.float32)


def _command(state: str) -> np.ndarray:
    command = np.zeros(13, dtype=np.float32)
    if state == "VELOCITY":
        command[0] = 0.2
    elif state == "SITSTAND":
        command[0] = 1.0
    return command


def run_sequence(model, policy: Policy, reference_onnx: Path, states: list[str], ticks: int) -> TraceMetrics:
    data = mujoco.MjData(model)
    helper = PolicyInference(model, data, walking_onnx_path=str(reference_onnx), new_cmd_obs=True,
                             use_projected_gravity=True)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr = int(model.jnt_qposadr[free_id])
    data.qpos[qadr:qadr + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    for i, qidx in enumerate(helper.joint_qpos_indices):
        data.qpos[qidx] = DEFAULT_POSE[i]
    mujoco.mj_forward(model, data)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    metrics = TraceMetrics()
    for state in states:
        command = _command(state)
        helper.command = command
        for _ in range(ticks):
            legacy = helper.get_observations()
            conditioned = make_conditioned_observation(
                legacy[None, :], command[None, :], STATE_TO_BEHAVIOR[state]
            )
            action = policy(conditioned)
            helper.last_action = action.copy()
            helper.apply_action(action)
            for _ in range(5):
                mujoco.mj_step(model, data)
            quat = data.xquat[trunk]
            tilt = 2.0 * np.arccos(np.clip(abs(float(quat[0])), 0.0, 1.0))
            metrics.append(height=data.xpos[trunk, 2], tilt=tilt, position=data.xpos[trunk], action=action)
            if not metrics.finite:
                return metrics
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="BC run directory containing model.pt and manifest.json")
    source.add_argument("--onnx", type=Path, help="exported 71D G0 ONNX policy")
    parser.add_argument("--observation-reference-onnx", type=Path, required=True,
                        help="61D specialist ONNX used by the established observation harness only")
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/generalist-g0/evaluation.json"))
    args = parser.parse_args()
    np.random.seed(args.seed)
    policy = Policy(args.run, args.onnx)
    model = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
    model.opt.timestep = 0.005
    behaviors = []
    for state, behavior in STATE_TO_BEHAVIOR.items():
        metrics = run_sequence(model, policy, args.observation_reference_onnx, [state], args.ticks)
        behaviors.append({"state": state, "behavior": behavior, "metrics": metrics.report()})
    edges = []
    for source_state, destination in sorted(LEGAL_EDGES):
        metrics = run_sequence(model, policy, args.observation_reference_onnx,
                               [source_state, destination], args.ticks)
        edges.append({"from": source_state, "to": destination, "reset_count": 0,
                      "metrics": metrics.report()})
    report = make_report(backend=policy.backend, seed=args.seed, behaviors=behaviors, edges=edges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
