"""Check the live Velocity-Flat reward, termination, and critic contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from isaaclab.app import AppLauncher


def _tensor(value):
    return getattr(value, "torch", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym

        from isaaclab_microduck.policy_abi import ACTION_SIZE
        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        cfg = make_velocity_flat_env_cfg(num_envs=args.num_envs)
        cfg.seed = 2026
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        obs, _ = env.reset(seed=2026)
        critic = obs["critic"] if isinstance(obs, dict) else None
        if critic is None:
            raise AssertionError("Velocity-Flat must expose a separate critic observation group")
        critic = _tensor(critic)
        # One complete manager step gives all terms the same state snapshot.
        obs, reward, terminated, truncated, _ = env.step(
            torch.zeros((args.num_envs, ACTION_SIZE), device=base_env.device)
        )
        critic = _tensor(obs["critic"])
        term_items = dict(base_env.reward_manager.get_active_iterable_terms(0))
        done_items = dict(base_env.termination_manager.get_active_iterable_terms(0))
        expected_rewards = {
            "track_lin_vel",
            "track_ang_vel",
            "upright",
            "pose",
            "head_pose",
            "body_pose",
            "head_pose_bias",
            "air_time",
            "body_ang_vel",
            "angular_momentum",
            "action_rate_l2",
            "dof_pos_limits",
            "foot_clearance",
            "foot_swing_height",
            "foot_slip",
            "self_collisions",
        }
        expected_terminations = {"time_out", "fallen", "nan_state", "terrain_out_of_bounds"}
        reward_names = set(term_items)
        termination_names = set(done_items)
        reward_finite = all(bool(torch.isfinite(torch.as_tensor(v)).all().item()) for v in term_items.values())
        termination_finite = all(bool(torch.isfinite(torch.as_tensor(v)).all().item()) for v in done_items.values())
        penalty_names = {
            "body_ang_vel",
            "angular_momentum",
            "action_rate_l2",
            "dof_pos_limits",
            "foot_clearance",
            "foot_swing_height",
            "foot_slip",
            "self_collisions",
        }
        penalty_sign_ok = all(float(term_items[name][0]) <= 1.0e-7 for name in penalty_names)
        # The manager stores weighted per-step values. Compare its raw output
        # with each direct kernel for the stateless terms that do not advance a
        # history buffer when evaluated a second time.
        stateless = {
            "track_lin_vel",
            "track_ang_vel",
            "upright",
            "pose",
            "head_pose",
            "body_pose",
            "body_ang_vel",
            "angular_momentum",
            "dof_pos_limits",
        }
        numeric = {}
        numeric_ok = True
        for name in sorted(stateless):
            cfg_term = base_env.reward_manager.get_term_cfg(name)
            direct = _tensor(cfg_term.func(base_env, **cfg_term.params)).detach().float()
            manager_value = torch.as_tensor(term_items[name], device=direct.device).reshape(-1)
            expected = direct[0] * float(cfg_term.weight)
            error = (manager_value[0] - expected).abs()
            numeric[name] = {"manager": float(manager_value[0]), "direct_weighted": float(expected), "abs_error": float(error)}
            numeric_ok = numeric_ok and bool(torch.isfinite(error).item()) and float(error) <= 1.0e-5
        finite = bool(torch.isfinite(_tensor(reward)).all().item() and torch.isfinite(_tensor(terminated)).all().item() and torch.isfinite(_tensor(truncated)).all().item())
        checks = {
            "reward_names_exact": reward_names == expected_rewards,
            "termination_names_exact": termination_names == expected_terminations,
            "reward_finite": reward_finite,
            "termination_finite": termination_finite,
            "penalty_sign_ok": penalty_sign_ok,
            "stateless_numeric_parity": numeric_ok,
            "critic_shape": list(critic.shape),
            "critic_76d": list(critic.shape) == [args.num_envs, 76],
            "all_outputs_finite": finite and bool(torch.isfinite(critic).all().item()),
        }
        if not all(checks.values()):
            raise AssertionError(f"runtime reward contract failed: {checks}")
        report = {
            "num_envs": args.num_envs,
            "reward_terms": {name: list(value) for name, value in term_items.items()},
            "termination_terms": {name: list(value) for name, value in done_items.items()},
            "numeric_fixture": numeric,
            "checks": checks,
            "critic_shape": list(critic.shape),
            "actor_shape": list(_tensor(obs["policy"]).shape),
            "stateful_terms_finite": sorted(expected_rewards - stateless),
        }
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
