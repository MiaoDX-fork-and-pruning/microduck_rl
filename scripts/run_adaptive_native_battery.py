#!/usr/bin/env python3
"""Evaluate a checkpoint through the training MJLab/BAM observation/actuator path.

The product tracking metric is a signed 0.5 s EMA of the command-aligned
velocity error.  The trace also keeps the samplewise MAE as a stability
diagnostic and hard cap; this lets a normal alternating gait pass without
letting violent oscillation average away.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np

from mjlab_microduck.evaluation.capability import (
    DEFAULT_INSTANTANEOUS_CAPS,
    DEFAULT_TRACKING_METRIC,
    DEFAULT_TRACKING_TAU_S,
    DEFAULT_THRESHOLDS,
    TRACKING_METRIC_SIGNED_EMA,
    build_capability_report,
    canonical_sha256,
    tracking_error_metrics,
)

# Frozen CPU battery commands, including moving turns (not turn-in-place).
BUCKETS = {
    "zero": (0.0, 0.0, 0.0),
    "forward": (0.12, 0.0, 0.0),
    "lateral": (0.0, 0.12, 0.0),
    "yaw": (0.0, 0.0, 0.8),
    "turn-left": (0.08, 0.0, 0.8),
    "turn-right": (0.08, 0.0, -0.8),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array(tensor):
    return tensor.detach().cpu().numpy().copy()


def freeze_commands(env, command):
    """Keep manager commands constant through reset AND every native env step."""
    import torch

    for name, value in (("twist", command), ("head_pose", (0.0,) * 4), ("body_pose", (0.0,) * 6)):
        term = env.command_manager.get_term(name)
        target = torch.tensor(value, device=env.device, dtype=term.command.dtype)

        def write(self, env_ids=None, target=target):
            self.command[:] = target
            if hasattr(self, "vel_command_w"):
                self.vel_command_w[:] = target
                self.is_standing_env[:] = bool((target == 0).all())
                self.is_world_env[:] = False
                self.is_heading_env[:] = False

        term._resample_command = MethodType(write, term)
        term._update_command = MethodType(write, term)
        term._update_command()


def raw_from_trace(
    trace,
    *,
    bucket,
    expected_steps,
    dt,
    tracking_metric=DEFAULT_TRACKING_METRIC,
    tracking_tau_s=DEFAULT_TRACKING_TAU_S,
):
    """Score pre-reset terminal evidence with the training-side error metric."""
    obs, action = trace["observation"], trace["action"]
    n = len(action)
    if n < 1 or obs.shape != (n, 61) or action.shape != (n, 14):
        raise ValueError("invalid native trace ABI")
    for key, values in trace.items():
        if np.asarray(values).dtype.kind in "fci" and not np.isfinite(values).all():
            raise ValueError(f"nonfinite native trace: {key}")
    quat = trace["root_quaternion_wxyz"]
    tilt = np.arccos(np.clip(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1.0, 1.0))
    terminated = bool(np.any(trace["terminated"]))
    raw = {
        "survival_fraction": (n - int(terminated)) / expected_steps,
        "tilt_p95_rad": float(np.percentile(tilt, 95)),
        "episode_length_mean": n * dt,
        "fall_rate": float(terminated),
        "action_magnitude_mean": float(np.abs(action).mean()),
    }
    if bucket == "zero":
        raw["zero_drift_m"] = float(np.linalg.norm(trace["root_position"][-1, :2] - trace["initial_root_position"][:2]))
    elif bucket in ("forward", "lateral"):
        axis = 0 if bucket == "forward" else 1
        samplewise, signed_ema = tracking_error_metrics(
            trace["linear_velocity_b"][:, axis],
            trace["command"][:, axis],
            dt=dt,
            tau_s=tracking_tau_s,
        )
        raw["tracking_error_m_s"] = float(
            signed_ema if tracking_metric == TRACKING_METRIC_SIGNED_EMA else samplewise
        )
        if tracking_metric == TRACKING_METRIC_SIGNED_EMA:
            raw["tracking_error_samplewise_m_s"] = float(samplewise)
    else:
        samplewise, signed_ema = tracking_error_metrics(
            trace["angular_velocity_b"][:, 2],
            trace["command"][:, 2],
            dt=dt,
            tau_s=tracking_tau_s,
        )
        raw["angular_tracking_error_rad_s"] = float(
            signed_ema if tracking_metric == TRACKING_METRIC_SIGNED_EMA else samplewise
        )
        if tracking_metric == TRACKING_METRIC_SIGNED_EMA:
            raw["angular_tracking_error_samplewise_rad_s"] = float(samplewise)
    return raw


def run_native(checkpoint: Path, output: Path, *, task: str, seed: int,
               steps: int, device: str, onnx: Path | None = None,
               seed_set_id: str = "adaptive-native-gate-20260920",
               axis_mode: str = "all_static", source_sha: str = "unknown",
               distribution: str = "final",
               zero_mode: str = "nominal") -> dict:
    import torch
    import mjlab_microduck.tasks  # noqa: F401
    from bam.mjlab import BamActuator
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    if (
        steps < 1
        or not seed_set_id.strip()
        or distribution not in ("initial", "final")
        or zero_mode not in ("nominal", "push")
    ):
        raise ValueError("invalid evaluation steps, seed set or distribution")
    checkpoint = checkpoint.resolve()
    output.mkdir(parents=True, exist_ok=True)
    session = None
    if onnx is not None:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    cases, raw_metrics, seed_cases = [], {}, []
    # BAM reads previous-solve force buffers that mjwarp.reset_data does not clear.
    # Fresh environments also isolate stateful reward terms without reset hooks.
    for offset, (bucket, command) in enumerate(BUCKETS.items()):
        # play=True changes push frequency. Keep the TRAINING task and its noise/DR.
        cfg = load_env_cfg(task, play=False)
        cfg.seed = seed
        cfg.scene.num_envs = 1
        cfg.auto_reset = False
        # The zero product bucket measures nominal idle stability.  A push
        # recovery probe is available as a separate diagnostic, but mixing it
        # into the idle gate would score an unobservable post-kick world-frame
        # offset as command-following failure.
        if bucket == "zero" and zero_mode == "nominal":
            cfg.events.pop("push_robot", None)
        if hasattr(cfg, "adaptive_evaluation_interval"):
            cfg.adaptive_evaluation_interval = 0
            # Actor-only load must evaluate exactly the requested distribution,
            # independent of the launch environment or checkpoint rehearsal.
            cfg.adaptive_final_com_fraction = 0.0
        # Evaluation uses an explicit frozen DR distribution. The training
        # curriculum is otherwise evaluated at ``reference_step`` during reset
        # and can silently overwrite the ranges before the first sample.
        for curriculum_name in ("com_range", "head_com_range"):
            cfg.curriculum.pop(curriculum_name, None)
        # Use the same canonical reference distribution for every branch/checkpoint.
        reference_step = 96000 if distribution == "final" else 0
        ranges = (0.015, 0.01) if distribution == "final" else (0.003, 0.003)
        for event, width in zip(("randomize_com", "randomize_head_com"), ranges, strict=True):
            cfg.events[event].params["ranges"] = (-width, width)
        agent_cfg = load_rl_cfg(task)
        raw_env = ManagerBasedRlEnv(cfg=cfg, device=device)
        try:
            raw_env.common_step_counter = reference_step
            env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
            # Inference must NOT restore adaptive trainer RNG, or start another gate.
            runner_cls = load_runner_cls(task)
            if runner_cls is None:
                from rsl_rl.runners import OnPolicyRunner

                runner_cls = OnPolicyRunner
            runner = runner_cls(env, asdict(agent_cfg), device=device)
            runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
            robot = raw_env.scene["robot"]
            actuators = robot.actuators
            if not actuators or not all(isinstance(a, BamActuator) for a in actuators):
                raise ValueError("native battery requires BAM actuators")
            actuator_config = [asdict(a.cfg) for a in actuators]
            bucket_seed = seed + offset
            freeze_commands(raw_env, command)
            raw_env.common_step_counter = reference_step
            # EventManager owns a deepcopy of cfg.events. Re-apply the frozen
            # distribution after runner construction so no manager-side config
            # mutation can silently restore the training curriculum range.
            for event, width in zip(("randomize_com", "randomize_head_com"), ranges, strict=True):
                raw_env.event_manager.get_term_cfg(event).params["ranges"] = (-width, width)
            obs, _ = raw_env.reset(seed=bucket_seed)
            live_ranges = {
                name: raw_env.event_manager.get_term_cfg(name).params["ranges"]
                for name in ("randomize_com", "randomize_head_com")
            }
            for name, width in zip(live_ranges, ranges, strict=True):
                if tuple(live_ranges[name]) != (-width, width):
                    raise ValueError(f"reference distribution drifted: {name}")
            initial = {
                "initial_root_position": _array(robot.data.root_link_pos_w[0]),
                "initial_joint_pos": _array(robot.data.joint_pos[0]),
                "initial_joint_vel": _array(robot.data.joint_vel[0]),
                "reset_body_ipos": _array(raw_env.sim.model.body_ipos),
                "reset_dof_armature": _array(raw_env.sim.model.dof_armature),
                "reset_friction_scale": _array(actuators[0].friction_scale),
            }
            rows = {name: [] for name in (
                "observation", "action", "command", "reward", "terminated", "truncated",
                "root_position", "root_quaternion_wxyz", "linear_velocity_b", "angular_velocity_b",
                "reward_terms", "termination_terms",
            )}
            if session is not None:
                rows["onnx_action"] = []
            reward_names = list(raw_env.reward_manager.active_terms)
            termination_names = list(raw_env.termination_manager.active_terms)
            for _ in range(steps):
                actor_obs = obs["actor"]
                with torch.inference_mode():
                    action = policy({"actor": actor_obs})
                observation = _array(actor_obs[0])
                if observation.shape != (61,) or action.shape != (1, 14):
                    raise ValueError("native actor ABI mismatch")
                expected_command = np.array((*command, *([0.0] * 10)), dtype=np.float32)
                if not np.array_equal(observation[48:], expected_command):
                    raise ValueError("command block drifted from frozen bucket")
                rows["observation"].append(observation)
                rows["action"].append(_array(action[0]))
                rows["command"].append(expected_command)
                if session is not None:
                    rows["onnx_action"].append(session.run(None, {session.get_inputs()[0].name: observation[None]})[0][0])
                obs, reward, done, _ = env.step(action)
                rows["reward"].append(float(reward[0]))
                rows["terminated"].append(bool(raw_env.reset_terminated[0]))
                rows["truncated"].append(bool(raw_env.reset_time_outs[0]))
                for name, tensor in (
                    ("root_position", robot.data.root_link_pos_w),
                    ("root_quaternion_wxyz", robot.data.root_link_quat_w),
                    ("linear_velocity_b", robot.data.root_link_lin_vel_b),
                    ("angular_velocity_b", robot.data.root_link_ang_vel_b),
                ):
                    rows[name].append(_array(tensor[0]))
                rows["reward_terms"].append([float(v[0]) for _, v in raw_env.reward_manager.get_active_iterable_terms(0)])
                rows["termination_terms"].append([bool(raw_env.termination_manager.get_term(n)[0]) for n in termination_names])
                if bool(done[0]):
                    break
            trace = {**initial, **{name: np.asarray(values) for name, values in rows.items()}}
            raw_metrics[bucket] = raw_from_trace(
                trace,
                bucket=bucket,
                expected_steps=steps,
                dt=raw_env.step_dt,
                tracking_metric=TRACKING_METRIC_SIGNED_EMA,
                tracking_tau_s=DEFAULT_TRACKING_TAU_S,
            )
            trace_path = output / f"{bucket}.npz"
            np.savez_compressed(trace_path, **trace)
            parity = None
            if session is not None:
                error = np.abs(trace["action"] - trace["onnx_action"])
                parity = {"max_abs": float(error.max()), "mean_abs": float(error.mean()),
                          "passed": bool(np.allclose(trace["action"], trace["onnx_action"], atol=1e-4, rtol=1e-3))}
            cases.append({"bucket": bucket, "steps": len(trace["action"]), "requested_steps": steps,
                          "finite_61d_14d": True, "trace": str(trace_path.resolve()),
                          "trace_sha256": sha256(trace_path), "parity": parity,
                          "reward_names": reward_names, "termination_names": termination_names,
                          "reward_terms_sum": dict(zip(reward_names, trace["reward_terms"].sum(axis=0).tolist(), strict=True))})
            seed_cases.append({"bucket": bucket, "reset_seed": bucket_seed,
                               "live_com_ranges": live_ranges,
                               "consumed_state_sha256": canonical_sha256({k: v.tolist() for k, v in initial.items()})})
            step_dt = raw_env.step_dt
            event_sources = {mode: list(names) for mode, names in raw_env.event_manager.active_terms.items()}
        finally:
            raw_env.close()
    config = {"name": "native_mjlab_bam_v2", "steps": steps, "commands": BUCKETS,
              "thresholds": DEFAULT_THRESHOLDS, "tracking_metric": TRACKING_METRIC_SIGNED_EMA,
              "tracking_metric_tau_s": DEFAULT_TRACKING_TAU_S,
              "instantaneous_caps": dict(DEFAULT_INSTANTANEOUS_CAPS),
              "zero_mode": zero_mode, "environment_profile": "training",
              "bucket_isolation": "fresh_environment",
              "distribution": distribution, "reference_env_step": reference_step,
              "auto_reset": False, "num_envs": 1, "device": device,
              "step_dt": step_dt,
              "actuator_config": actuator_config,
              "implementation_sha256": sha256(Path(__file__))}
    report = build_capability_report(raw_metrics, axis_mode=axis_mode, evaluator_config=config,
        metadata={"task_id": task, "source_sha": source_sha, "evaluator_config_sha256": canonical_sha256(config),
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "policy_format": "native_mjlab_bam_pt", "seed_set_id": seed_set_id,
                  "evaluation_seed": seed, "generated_at": datetime.now(timezone.utc).isoformat()})
    payload = report.payload
    payload.update({"cases": cases, "seed_manifest": {
        "version": "native-reset-dr-v3", "seed_set_id": seed_set_id,
        "startup_seed": seed, "cases": seed_cases,
        "sources": event_sources,
        "consumed_state_fields": list(initial),
    }, "onnx_sha256": sha256(onnx) if onnx else None})
    payload["report_sha256"] = canonical_sha256(payload)
    (output / "native_capability.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--onnx", type=Path)
    p.add_argument("--seed-set-id", default="adaptive-native-gate-20260920")
    p.add_argument("--axis-mode", default="all_static", choices=("all_static", "com", "head_com", "composed"))
    p.add_argument("--source-sha", default="unknown")
    p.add_argument("--distribution", choices=("initial", "final"), default="final")
    p.add_argument("--zero-mode", choices=("nominal", "push"), default="nominal")
    payload = run_native(**vars(p.parse_args()))
    print(json.dumps(payload["aggregate"], indent=2))
    return 0  # A valid negative capability report is a successful evaluation.


if __name__ == "__main__":
    raise SystemExit(main())
