"""Evaluate a fixed checkpoint on a deterministic Velocity-Flat command battery.

The battery keeps each command fixed for the whole episode slice and records
tracking, reset, posture, and action statistics. It is an evaluation harness,
not a training script and does not change the registered task configuration.
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
    from scripts.isaaclab.velocity_flat_battery_spec import (
        CASES,
        NUM_ENVS,
        SEED,
        STEPS_PER_CASE,
        evaluate_case,
    )
except ModuleNotFoundError:  # direct ``python scripts/isaaclab/...py`` entry
    from velocity_flat_battery_spec import CASES, NUM_ENVS, SEED, STEPS_PER_CASE, evaluate_case


TASK = "IsaacLab-Velocity-Flat-MicroDuck"
# Keep the required scenarios visible at this executable boundary.  The case
# definitions remain centralized in ``velocity_flat_battery_spec`` so the
# MuJoCo and IsaacLab harnesses cannot silently drift apart.
REQUIRED_CASE_NAMES = ("zero", "forward", "lateral", "yaw")
COMMANDS = {case.name: case.command for case in CASES}
if not all(name in COMMANDS for name in REQUIRED_CASE_NAMES):
    raise RuntimeError(f"fixed command battery cases missing required scenarios: {tuple(COMMANDS)!r}")


def _tilt_rad(quat_xyzw: torch.Tensor) -> torch.Tensor:
    scalar = torch.clamp(torch.abs(quat_xyzw[..., 3]), max=1.0)
    return 2.0 * torch.acos(scalar)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_command(term, value: tuple[float, float, float]) -> None:
    term.command[:] = torch.as_tensor(value, device=term.command.device).reshape(1, 3)
    term.is_standing_env[:] = False
    term.time_left[:] = float("inf")


def _run_case(env, policy, obs, name: str, value: tuple[float, float, float], steps: int) -> tuple[dict, object]:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    command_term = base_env.command_manager.get_term("base_velocity")
    _set_command(command_term, value)

    errors_xy: list[torch.Tensor] = []
    errors_yaw: list[torch.Tensor] = []
    actual_xy_values: list[torch.Tensor] = []
    actual_yaw_values: list[torch.Tensor] = []
    heights: list[torch.Tensor] = []
    tilts: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    resets = 0
    finite = True
    for _ in range(steps):
        with torch.inference_mode():
            action = policy(obs)
            obs, _, dones, _ = env.step(action)
        _set_command(command_term, value)
        desired = command_term.command
        actual_xy = robot.data.root_lin_vel_b.torch[:, :2]
        actual_yaw = robot.data.root_ang_vel_b.torch[:, 2]
        root_pos = robot.data.root_link_pos_w.torch
        root_quat = robot.data.root_link_quat_w.torch
        tilt = _tilt_rad(root_quat)
        tensors = (actual_xy, actual_yaw, root_pos, root_quat, action)
        finite = finite and all(bool(torch.isfinite(t).all().item()) for t in tensors)
        errors_xy.append(torch.linalg.norm(desired[:, :2] - actual_xy, dim=-1).detach().cpu())
        errors_yaw.append(torch.abs(desired[:, 2] - actual_yaw).detach().cpu())
        actual_xy_values.append(actual_xy.detach().cpu())
        actual_yaw_values.append(actual_yaw.detach().cpu())
        heights.append(root_pos[:, 2].detach().cpu())
        tilts.append(tilt.detach().cpu())
        actions.append(action.detach().cpu())
        resets += int(dones.sum().item())

    xy = torch.cat(errors_xy)
    yaw = torch.cat(errors_yaw)
    actual_xy_all = torch.cat(actual_xy_values)
    actual_yaw_all = torch.cat(actual_yaw_values)
    height = torch.cat(heights)
    tilt = torch.cat(tilts)
    action_values = torch.cat(actions)
    result = {
        "name": name,
        "command": list(value),
        "steps": steps,
        "envs": base_env.scene.num_envs,
        "finite": finite,
        "mean_error_vel_xy_m_s": float(xy.mean()),
        "p95_error_vel_xy_m_s": float(torch.quantile(xy, 0.95)),
        "mean_error_vel_yaw_rad_s": float(yaw.mean()),
        "p95_error_vel_yaw_rad_s": float(torch.quantile(yaw, 0.95)),
        "mean_actual_vel_xy_m_s": [float(value) for value in actual_xy_all.mean(dim=0)],
        "mean_actual_vel_yaw_rad_s": float(actual_yaw_all.mean()),
        "episode_resets": resets,
        "reset_fraction_per_env_step": resets / (steps * base_env.scene.num_envs),
        "mean_root_height_m": float(height.mean()),
        "min_root_height_m": float(height.min()),
        "max_tilt_rad": float(tilt.max()),
        "mean_abs_action": float(action_values.abs().mean()),
        "p95_abs_action": float(torch.quantile(action_values.abs(), 0.95)),
    }
    case = next(case for case in CASES if case.name == name)
    result["passed"], result["failures"] = evaluate_case(
        case,
        finite=finite,
        mean_actual_xy=result["mean_actual_vel_xy_m_s"],
        mean_actual_yaw=result["mean_actual_vel_yaw_rad_s"],
        max_tilt=result["max_tilt_rad"],
        reset_fraction=result["reset_fraction_per_env_step"],
    )
    return result, obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--steps", type=int, default=STEPS_PER_CASE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        print("ISAACLAB_VELOCITY_BATTERY:app_ready", flush=True)
        import gymnasium as gym

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg import MicroduckVelocityFlatPPORunnerCfg
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        print("ISAACLAB_VELOCITY_BATTERY:tasks_registered", flush=True)
        env_cfg = make_velocity_flat_env_cfg(play=True, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env = gym.make(TASK, cfg=env_cfg)
        print("ISAACLAB_VELOCITY_BATTERY:env_made", flush=True)
        # IsaacLab's wrapper forwards this value to gymnasium.Box(high=...),
        # which requires a numeric bound rather than a boolean.
        vec_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
        print("ISAACLAB_VELOCITY_BATTERY:wrapper_made", flush=True)
        agent_cfg = MicroduckVelocityFlatPPORunnerCfg()
        # The official IsaacLab entrypoint migrates legacy model fields before
        # handing the config to rsl-rl. Keep this standalone harness on the
        # same compatibility path.
        handle_deprecated_rsl_rl_cfg(agent_cfg, "5.4.1")
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=vec_env.unwrapped.device)
        print("ISAACLAB_VELOCITY_BATTERY:runner_made", flush=True)
        runner.load(str(args.checkpoint))
        print("ISAACLAB_VELOCITY_BATTERY:checkpoint_loaded", flush=True)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        cases = []
        obs = vec_env.get_observations()
        for index, (name, command) in enumerate(COMMANDS.items()):
            print(f"ISAACLAB_VELOCITY_BATTERY:case:{name}:start", flush=True)
            vec_env.seed(args.seed + index)
            obs, _ = vec_env.reset()
            result, obs = _run_case(vec_env, policy, obs, name, command, args.steps)
            cases.append(result)
            print(f"ISAACLAB_VELOCITY_BATTERY:case:{name}:done", flush=True)

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
            "actuator": "BamActuator",
            "friction_bridge": "motor_only_external_effort_unavailable",
            "passed": all(case["passed"] for case in cases),
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
