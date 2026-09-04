"""Trace the strict Velocity-Flat command/action/dynamics path.

This is a bounded diagnostic companion to ``velocity_flat_command_battery``.
It holds the same command cases fixed but records enough intermediate state to
classify a directional response failure before changing task rewards or PPO.
The trace is observational: it does not alter actions, actuator math, or task
configuration beyond freezing command terms for each case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

try:
    from scripts.isaaclab.velocity_flat_battery_spec import CASES
except ModuleNotFoundError:
    from velocity_flat_battery_spec import CASES


TASK = "IsaacLab-Velocity-Flat-MicroDuck"
TRACE_CASES = tuple(case for case in CASES if case.name in {"zero", "forward", "lateral", "yaw"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(term, value: torch.Tensor) -> None:
    term.command[:] = value.reshape(1, -1)
    term.time_left[:] = float("inf")


def _set_command(base_env, value: tuple[float, float, float]) -> None:
    velocity = base_env.command_manager.get_term("base_velocity")
    _freeze(velocity, torch.as_tensor(value, device=velocity.command.device, dtype=velocity.command.dtype))
    velocity.is_standing_env[:] = False
    for name, dim in (("head_pose", 4), ("body_pose", 6)):
        term = base_env.command_manager.get_term(name)
        _freeze(term, torch.zeros(dim, device=term.command.device, dtype=term.command.dtype))


def _refresh_obs(base_env, vec_env):
    base_env.obs_buf = base_env.observation_manager.compute(update_history=False)
    return vec_env.get_observations()


def _first(value):
    if value is None or not torch.is_tensor(value):
        return None
    return value[0].detach().cpu().tolist()


def _actor_obs(obs):
    """Unwrap IsaacLab's policy TensorDict without changing runner input."""

    if torch.is_tensor(obs):
        return obs
    if hasattr(obs, "get"):
        value = obs.get("policy")
        if value is not None:
            return value
    raise TypeError(f"unsupported observation container: {type(obs)!r}")


def _state_snapshot(base_env, robot, actuator, term, action, obs_before) -> dict:
    action_manager = base_env.action_manager
    target = getattr(robot.data, "joint_pos_target", None)
    if target is None:
        target = getattr(term, "processed_actions", None)
    processed = getattr(action_manager, "processed_actions", None)
    raw_manager = getattr(action_manager, "action", None)
    command = base_env.command_manager.get_command("base_velocity")
    root_lin = robot.data.root_lin_vel_b.torch
    root_ang = robot.data.root_ang_vel_b.torch
    fields = {
        "command": _first(command),
        "command_obs_tail": _first(_actor_obs(obs_before)[:, -13:]),
        "raw_policy_action": _first(action),
        "action_manager_action": _first(raw_manager),
        "action_manager_processed": _first(processed),
        "joint_term_raw": _first(getattr(term, "raw_actions", None)),
        "joint_term_processed": _first(getattr(term, "processed_actions", None)),
        "joint_target": _first(target),
        "bam_last_target": _first(getattr(actuator, "_last_target", None)),
        "bam_delayed_target": _first(getattr(actuator, "_delayed_target", None)),
        "bam_previous_motor_effort": _first(getattr(actuator, "_previous_motor_effort", None)),
        "bam_applied_effort": _first(getattr(actuator, "_applied_effort", None)),
        "bam_delay_lag": _first(getattr(getattr(actuator, "_delay", None), "delay", None)),
        "bam_delay_initialized": _first(getattr(actuator, "_delay_initialized", None)),
        "root_lin_vel_b": _first(root_lin),
        "root_ang_vel_b": _first(root_ang),
    }
    finite = True
    for value in fields.values():
        if value is None:
            continue
        finite = finite and all(torch.isfinite(torch.as_tensor(value)).flatten().tolist())
    fields["finite"] = finite
    return fields


def _run_case(vec_env, policy, obs, base_env, robot, actuator, term, name, command, steps):
    _set_command(base_env, command)
    obs = _refresh_obs(base_env, vec_env)
    traces = []
    for step in range(steps):
        with torch.inference_mode():
            action = policy(obs)
        command_obs = obs
        with torch.inference_mode():
            obs, _, dones, _ = vec_env.step(action)
        _set_command(base_env, command)
        obs = _refresh_obs(base_env, vec_env)
        snapshot = _state_snapshot(base_env, robot, actuator, term, action, command_obs)
        snapshot["step"] = step
        snapshot["done_count"] = int(dones.sum().item())
        traces.append(snapshot)
    return {
        "name": name,
        "command": list(command),
        "steps": steps,
        "finite": all(item["finite"] for item in traces),
        "trace": traces,
        "mean_raw_abs_action": float(torch.as_tensor([item["raw_policy_action"] for item in traces]).abs().mean()),
        "mean_bam_abs_effort": float(torch.as_tensor([item["bam_applied_effort"] for item in traces]).abs().mean()),
        "mean_root_lin_vel_b": [
            float(value)
            for value in torch.as_tensor([item["root_lin_vel_b"] for item in traces]).mean(dim=0)
        ],
    }, obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg import MicroduckVelocityFlatPPORunnerCfg
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        env_cfg = make_velocity_flat_env_cfg(play=True, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env = gym.make(TASK, cfg=env_cfg)
        agent_cfg = MicroduckVelocityFlatPPORunnerCfg()
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        handle_deprecated_rsl_rl_cfg(agent_cfg, "5.4.1")
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=vec_env.unwrapped.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        base_env = vec_env.unwrapped
        robot = base_env.scene["robot"]
        actuator = next(iter(robot.actuators.values()))
        term = base_env.action_manager.get_term("joint_pos")
        obs = vec_env.get_observations()
        cases = []
        for index, case in enumerate(TRACE_CASES):
            vec_env.seed(args.seed + index)
            obs, _ = vec_env.reset()
            result, obs = _run_case(
                vec_env, policy, obs, base_env, robot, actuator, term,
                case.name, case.command, args.steps,
            )
            cases.append(result)
        report = {
            "task": TASK,
            "backend": "isaaclab",
            "isaaclab_version": "3.0.0",
            "isaacsim_version": "6.0.1",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps_per_case": args.steps,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "actuator": type(actuator).__name__,
            "clip_actions": agent_cfg.clip_actions,
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
