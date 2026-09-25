"""Collect deterministic runtime evidence for Velocity-Flat command terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from isaaclab.app import AppLauncher


def _tensor(value):
    return getattr(value, "torch", value)


def _turn_mask(command: torch.Tensor) -> torch.Tensor:
    return (command[:, :2].abs().sum(dim=1) < 1.0e-8) & (command[:, 2].abs() >= 0.4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        cfg = make_velocity_flat_env_cfg(num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        env.reset(seed=args.seed)
        manager = base_env.command_manager
        dt = float(base_env.step_dt)
        names = ("base_velocity", "head_pose", "body_pose")
        terms = {name: manager.get_term(name) for name in names}
        expected_ranges = {
            name: tuple(float(value) for value in terms[name].cfg.resampling_time_range)
            for name in names
        }
        counters = {
            name: _tensor(terms[name].command_counter).detach().clone()
            for name in names
        }
        elapsed = {
            name: torch.zeros(args.num_envs, device=base_env.device)
            for name in names
        }
        intervals = {name: [] for name in names}
        sampled_time_left = {
            name: _tensor(terms[name].time_left).detach().float().cpu().tolist()
            for name in names
        }
        base_samples = [_tensor(manager.get_command("base_velocity")).detach().clone()]

        for _ in range(args.steps):
            for name in names:
                elapsed[name].add_(dt)
            manager.compute(dt=dt)
            for name in names:
                term = terms[name]
                current = _tensor(term.command_counter)
                changed = current > counters[name]
                if changed.any():
                    ids = changed.nonzero().flatten()
                    intervals[name].extend(elapsed[name][ids].float().cpu().tolist())
                    sampled_time_left[name].extend(
                        _tensor(term.time_left)[ids].detach().float().cpu().tolist()
                    )
                    elapsed[name][ids] = 0.0
                    if name == "base_velocity":
                        base_samples.append(
                            _tensor(manager.get_command(name))[ids].detach().clone()
                        )
                counters[name] = current.detach().clone()

        base_commands = torch.cat(base_samples, dim=0)
        turns = _turn_mask(base_commands)
        turn_commands = base_commands[turns]
        total_samples = int(base_commands.shape[0])
        turn_count = int(turns.sum().item())
        turn_fraction = float(turns.float().mean().item())
        interval_report = {}
        interval_ok = True
        for name in names:
            values = torch.tensor(intervals[name])
            sampled = torch.tensor(sampled_time_left[name])
            lo, hi = expected_ranges[name]
            observed_ok = bool(
                values.numel() > 0
                and (values >= lo - dt - 1.0e-5).all()
                and (values <= hi + dt + 1.0e-5).all()
            )
            sampled_ok = bool((sampled >= lo).all() and (sampled <= hi).all())
            interval_ok = interval_ok and observed_ok and sampled_ok
            interval_report[name] = {
                "expected_seconds": [lo, hi],
                "observed_count": int(values.numel()),
                "observed_min_seconds": float(values.min().item()),
                "observed_max_seconds": float(values.max().item()),
                "sampled_time_left_min_seconds": float(sampled.min().item()),
                "sampled_time_left_max_seconds": float(sampled.max().item()),
                "observed_within_one_control_step": observed_ok,
                "sampled_within_range": sampled_ok,
            }

        turn_semantics_ok = bool(
            turn_count > 0
            and (turn_commands[:, :2] == 0.0).all()
            and (turn_commands[:, 2].abs() >= 0.4).all()
            and (turn_commands[:, 2].abs() <= 1.0).all()
        )
        # This is a seeded statistical proof, not an exact-count contract.
        turn_fraction_ok = 0.10 <= turn_fraction <= 0.20
        finite = bool(torch.isfinite(base_commands).all().item())
        if not interval_ok or not turn_semantics_ok or not turn_fraction_ok or not finite:
            raise AssertionError(
                "command runtime contract failed: "
                f"intervals={interval_ok}, turn_semantics={turn_semantics_ok}, "
                f"turn_fraction={turn_fraction:.6f}, finite={finite}"
            )
        report = {
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "control_dt": dt,
            "intervals": interval_report,
            "base_command_samples": total_samples,
            "turn_bucket_count": turn_count,
            "turn_bucket_fraction": turn_fraction,
            "turn_linear_zero": bool((turn_commands[:, :2] == 0.0).all().item()),
            "turn_yaw_abs_min": float(turn_commands[:, 2].abs().min().item()),
            "turn_yaw_abs_max": float(turn_commands[:, 2].abs().max().item()),
            "finite": finite,
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
