#!/usr/bin/env python3
"""Run the canonical three-behavior/four-edge G0 CPU MuJoCo battery."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.generalist_g0_evaluation import STATE_TO_BEHAVIOR, TraceMetrics, make_report
from mjlab_microduck.generalist_model import build_actor
from mjlab_microduck.generalist_schema import make_conditioned_observation
from mjlab_microduck.generalist_transition_graph import LEGAL_EDGES
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SIT_Z, STAND_Z

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, PolicyInference

CONTROL_HZ = 50
STAND_HEIGHT_MIN_M = STAND_Z * 0.9
SIT_HEIGHT_MAX_M = SIT_Z * 1.1


@dataclass(frozen=True)
class Segment:
    state: str
    command_x: float
    duration_s: float
    active_transition: bool = False
    score: bool = True

    @property
    def ticks(self) -> int:
        return round(self.duration_s * CONTROL_HZ)


BEHAVIOR_SEGMENTS = {
    "VELSTAND": (Segment("VELSTAND", 0.0, 8.0),),
    "VELOCITY": (Segment("VELOCITY", 0.20, 14.0),),
    "SITSTAND": (Segment("SITSTAND", 1.0, 6.0), Segment("SITSTAND", 0.0, 6.0)),
}

EDGE_SEGMENTS = {
    ("VELSTAND", "VELOCITY"): (Segment("VELSTAND", 0.0, 8.0, score=False), Segment("VELOCITY", 0.20, 14.0, True)),
    ("VELOCITY", "VELSTAND"): (Segment("VELOCITY", 0.20, 14.0, score=False), Segment("VELSTAND", 0.0, 8.0, True)),
    ("VELSTAND", "SITSTAND"): (Segment("VELSTAND", 0.0, 8.0, score=False), Segment("SITSTAND", 1.0, 6.0, True)),
    ("SITSTAND", "VELSTAND"): (
        Segment("SITSTAND", 1.0, 6.0, score=False),
        Segment("SITSTAND", 0.0, 6.0, True),
        Segment("VELSTAND", 0.0, 8.0),
    ),
}


class Policy:
    def __init__(self, run: Path | None = None, onnx: Path | None = None,
                 checkpoint: Path | None = None, task: str = "Mjlab-GeneralistG0-DirectPPO-Flat-MicroDuck",
                 device: str | None = None):
        if sum(value is not None for value in (run, onnx, checkpoint)) != 1:
            raise ValueError("choose exactly one of --run, --onnx, or --checkpoint")
        self.backend = "onnx" if onnx else ("rsl_rl" if checkpoint else "pytorch")
        if onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            if len(inputs) != 1 or inputs[0].shape[-1] != 71:
                raise ValueError("G0 ONNX must expose one 71D input")
            self.input_name = inputs[0].name
        elif run:
            import torch
            bundle = torch.load(run / "model.pt", weights_only=False)
            metadata = json.loads((run / "manifest.json").read_text()).get("metrics", {})
            self.model = build_actor(metadata)
            self.model.load_state_dict(bundle["state_dict"])
            self.model.eval()
        else:
            self._load_rsl_checkpoint(checkpoint, task, device)

    def _load_rsl_checkpoint(self, checkpoint: Path, task: str, device: str | None) -> None:
        """Load an rsl_rl actor through the task registry (critic is ignored)."""
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actor_state = payload.get("actor_state_dict", {})
        # The canonical standalone MuJoCo harness supplies raw 71D vectors,
        # whereas rsl_rl inference policies require a grouped TensorDict. For
        # checkpoint diagnostics, reconstruct the saved MLP directly so the
        # observation contract remains explicit and reproducible.
        mlp_keys = [key for key in actor_state if key.startswith("mlp.") and key.endswith(".weight")]
        if {key for key in mlp_keys} >= {"mlp.0.weight", "mlp.2.weight", "mlp.4.weight", "mlp.6.weight"}:
            from torch import nn
            dims = [int(actor_state[f"mlp.{i}.weight"].shape[1]) for i in (0, 2, 4, 6)] + [14]
            layers = []
            for index, (source, target) in enumerate(zip(dims, dims[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(dims) - 2:
                    layers.append(nn.ELU())
            model = nn.Sequential(*layers)
            state = {key: actor_state[f"mlp.{key}"] for key in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias", "6.weight", "6.bias")}
            model.load_state_dict(state, strict=True)
            self.model = model.to(device or "cpu").eval()
            mean = actor_state.get("obs_normalizer._mean")
            std = actor_state.get("obs_normalizer._std")
            self._obs_mean = mean.reshape(-1).numpy() if mean is not None else None
            self._obs_std = std.reshape(-1).numpy() if std is not None else None
            self.backend = "rsl_raw_checkpoint"
            return
        from dataclasses import asdict
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
        import mjlab.tasks  # noqa: F401
        target = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        env_cfg = load_env_cfg(task, play=True)
        env_cfg.scene.num_envs = 1
        env_cfg.seed = 42
        raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=target)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=load_rl_cfg(task).clip_actions)
        agent_cfg = load_rl_cfg(task)
        runner_cls = load_runner_cls(task)
        if runner_cls is None:
            from rsl_rl.runners import OnPolicyRunner
            runner_cls = OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=target)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=target)
        self._runner_env = env
        self._torch_policy = runner.get_inference_policy(device=target)
        self._device = target

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            value = np.asarray(self.session.run(None, {self.input_name: observation})[0][0], dtype=np.float32)
            return value
        import torch
        if self.backend == "rsl_raw_checkpoint":
            if self._obs_mean is not None and self._obs_std is not None:
                observation = (observation - self._obs_mean) / np.maximum(self._obs_std, 1e-6)
            with torch.inference_mode():
                value = self.model(torch.from_numpy(observation[None, :]).to(next(self.model.parameters()).device))
            return value.detach().cpu().numpy()[0].astype(np.float32)
        if self.backend == "rsl_rl":
            with torch.inference_mode():
                value = self._torch_policy({"actor": torch.from_numpy(observation[None, :]).to(self._device)})
            return value.detach().cpu().numpy()[0].astype(np.float32)
        with torch.inference_mode():
            value = self.model(torch.from_numpy(observation)).numpy()[0].astype(np.float32)
            return value


def _command(state: str, command_x: float | None = None) -> np.ndarray:
    command = np.zeros(13, dtype=np.float32)
    command[0] = command_x if command_x is not None else (0.2 if state == "VELOCITY" else 1.0 if state == "SITSTAND" else 0.0)
    return command


BEHAVIOR_GATES = {
    "stand": {"final_height_min_m": STAND_HEIGHT_MIN_M},
    "locomotion": {"displacement_gate_m": 1.0},
    "sit_stand": {
        "height_min_m": STAND_HEIGHT_MIN_M,
        "height_max_m": SIT_HEIGHT_MAX_M,
        "final_height_min_m": STAND_HEIGHT_MIN_M,
    },
}
EDGE_GATES = {
    ("VELSTAND", "VELOCITY"): {"displacement_gate_m": 1.0},
    ("VELOCITY", "VELSTAND"): {"final_height_min_m": STAND_HEIGHT_MIN_M},
    ("VELSTAND", "SITSTAND"): {"final_height_max_m": SIT_HEIGHT_MAX_M},
    ("SITSTAND", "VELSTAND"): {
        "height_max_m": SIT_HEIGHT_MAX_M,
        "final_height_min_m": STAND_HEIGHT_MIN_M,
    },
}


def run_sequence(model, policy: Policy, reference_onnx: Path, segments: tuple[Segment, ...],
                 *, condition_mode: str = "canonical") -> TraceMetrics:
    if condition_mode not in ("canonical", "training_default"):
        raise ValueError(f"unknown condition mode: {condition_mode}")
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
    for segment in segments:
        state = segment.state
        command = _command(state, segment.command_x)
        helper.command = command
        for tick in range(segment.ticks):
            legacy = helper.get_observations()
            kwargs = {}
            if condition_mode == "canonical":
                kwargs = {
                    # G0 phase is a transition timer. Standalone behavior
                    # episodes start with phase=0 and do not invent elapsed
                    # phase that the training environment never supplied.
                    "phase": np.array([[tick / max(segment.ticks - 1, 1) if segment.active_transition else 0.0,
                                         float(segment.active_transition)]], dtype=np.float32),
                    "posture": np.array([[segment.command_x if state == "SITSTAND" else 0.0]], dtype=np.float32),
                }
            conditioned = make_conditioned_observation(
                legacy[None, :], command[None, :], STATE_TO_BEHAVIOR[state], **kwargs
            )
            action = policy(conditioned)
            helper.last_action = action.copy()
            helper.apply_action(action)
            for _ in range(4):
                mujoco.mj_step(model, data)
            # Tilt is loss of the body's world-up alignment. A quaternion's
            # total rotation angle would incorrectly count yaw as falling.
            rotation = data.xmat[trunk].reshape(3, 3)
            tilt = np.arccos(np.clip(float(rotation[2, 2]), -1.0, 1.0))
            if segment.score:
                metrics.append(height=data.xpos[trunk, 2], tilt=tilt, position=data.xpos[trunk], action=action)
            if not metrics.finite:
                return metrics
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="BC run directory containing model.pt and manifest.json")
    source.add_argument("--onnx", type=Path, help="exported 71D G0 ONNX policy")
    source.add_argument("--checkpoint", type=Path, help="rsl_rl model_*.pt checkpoint")
    parser.add_argument("--task", default="Mjlab-GeneralistG0-DirectPPO-Flat-MicroDuck",
                        help="registered task used to construct the rsl_rl runner")
    parser.add_argument("--device", default=None)
    parser.add_argument("--observation-reference-onnx", type=Path, required=True,
                        help="61D specialist ONNX used by the established observation harness only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition-mode", choices=("canonical", "training_default"), default="canonical",
                        help="condition fields used by the evaluator; training_default is a contract probe")
    parser.add_argument("--allow-action-overflow", action="store_true",
                        help="diagnostic control-arm mode for raw specialist outputs")
    parser.add_argument("--output", type=Path, default=Path("artifacts/generalist-g0/evaluation.json"))
    args = parser.parse_args()
    np.random.seed(args.seed)
    policy = Policy(args.run, args.onnx, args.checkpoint, args.task, args.device)
    model = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
    model.opt.timestep = 0.005
    behaviors = []
    for state, behavior in STATE_TO_BEHAVIOR.items():
        metrics = run_sequence(model, policy, args.observation_reference_onnx, BEHAVIOR_SEGMENTS[state], condition_mode=args.condition_mode)
        behaviors.append({"state": state, "behavior": behavior,
                          "metrics": metrics.report(**BEHAVIOR_GATES[behavior], enforce_action_range=not args.allow_action_overflow)})
    edges = []
    for source_state, destination in sorted(LEGAL_EDGES):
        metrics = run_sequence(model, policy, args.observation_reference_onnx,
                               EDGE_SEGMENTS[(source_state, destination)], condition_mode=args.condition_mode)
        edges.append({"from": source_state, "to": destination, "reset_count": 0,
                      "metrics": metrics.report(**EDGE_GATES[(source_state, destination)], enforce_action_range=not args.allow_action_overflow)})
    report = make_report(backend=policy.backend, seed=args.seed, behaviors=behaviors, edges=edges)
    report["condition_mode"] = args.condition_mode
    report["action_range_enforced"] = not args.allow_action_overflow
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
