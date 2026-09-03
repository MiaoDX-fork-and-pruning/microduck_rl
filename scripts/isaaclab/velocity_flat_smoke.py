"""Run staged random-action and PPO-shape smoke checks for Velocity-Flat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def main() -> None:
    print("ISAACLAB_VELOCITY_FLAT_SMOKE:main:start", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    print("ISAACLAB_VELOCITY_FLAT_SMOKE:app_ready", flush=True)
    env = None
    try:
        import gymnasium as gym

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.policy_abi import ACTION_SIZE, OBSERVATION_SIZE

        register_tasks()
        print("ISAACLAB_VELOCITY_FLAT_SMOKE:tasks_registered", flush=True)
        env = gym.make(
            "IsaacLab-Velocity-Flat-MicroDuck",
            cfg=__import__(
                "isaaclab_microduck.tasks.velocity_flat",
                fromlist=["make_velocity_flat_env_cfg"],
            ).make_velocity_flat_env_cfg(num_envs=args.num_envs),
        )
        print("ISAACLAB_VELOCITY_FLAT_SMOKE:env_made", flush=True)
        base_env = env.unwrapped
        print("ISAACLAB_VELOCITY_FLAT_SMOKE:ground_event_ready", flush=True)
        obs, _ = env.reset(seed=7)
        print("ISAACLAB_VELOCITY_FLAT_SMOKE:reset_ok", flush=True)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        if policy_obs.ndim == 1:
            policy_obs = policy_obs.unsqueeze(0)
        checks = {
            "num_envs": args.num_envs,
            "observation_shape": list(policy_obs.shape),
            "observation_dim": int(policy_obs.shape[-1]),
            "action_dim": int(base_env.action_manager.total_action_dim),
            "finite_reset": bool(torch.isfinite(policy_obs).all().item()),
            "steps": args.steps,
        }
        if checks["observation_dim"] != OBSERVATION_SIZE:
            raise RuntimeError(f"expected {OBSERVATION_SIZE}D policy observation, got {policy_obs.shape}")
        if checks["action_dim"] != ACTION_SIZE:
            raise RuntimeError(f"expected {ACTION_SIZE}D action, got {checks['action_dim']}")
        finite_steps = True
        terminated_count = 0
        for _ in range(args.steps):
            actions = torch.empty(args.num_envs, ACTION_SIZE, device=base_env.device).uniform_(-1.0, 1.0)
            obs, _, terminated, truncated, _ = env.step(actions)
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            finite_steps = finite_steps and bool(torch.isfinite(policy_obs).all().item())
            terminated_count += int(terminated.sum().item())
            terminated_count += int(truncated.sum().item())
        print("ISAACLAB_VELOCITY_FLAT_SMOKE:steps_ok", flush=True)
        checks.update({"finite_steps": finite_steps, "episode_resets": terminated_count})
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
        print(json.dumps(checks, indent=2, sort_keys=True))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
