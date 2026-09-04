"""Prove the production Velocity-Flat push event writes only selected envs."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def _tensor(value):
    return getattr(value, "torch", value)


def _delta(before: torch.Tensor, after: torch.Tensor) -> dict[str, object]:
    value = (_tensor(after) - _tensor(before)).detach().float().cpu()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
        "mean_abs": float(value.abs().mean().item()) if value.numel() else 0.0,
    }


def _summary(value: torch.Tensor) -> dict[str, object]:
    value = _tensor(value).detach().float().cpu()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()) if value.numel() else None,
        "max": float(value.max().item()) if value.numel() else None,
        "mean": float(value.mean().item()) if value.numel() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    result: dict[str, object] = {
        "task": "IsaacLab-Velocity-Flat-MicroDuck",
        "seed": args.seed,
        "status": "BLOCKED",
    }
    try:
        import gymnasium as gym

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        if args.num_envs < 4:
            raise ValueError("--num-envs must be at least 4 so selected and unselected envs coexist")
        register_tasks()
        cfg = make_velocity_flat_env_cfg(num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        env.reset(seed=args.seed)
        robot = base_env.scene["robot"]
        data = robot.data
        event_cfg = base_env.event_manager.get_term_cfg("push_robot")
        asset_cfg = event_cfg.params["asset_cfg"]
        velocity_range = {
            "x": (0.25, 0.25),
            "y": (-0.10, -0.10),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        selected = torch.tensor([1, 3], device=base_env.device, dtype=torch.long)
        unselected = torch.tensor(
            [index for index in range(args.num_envs) if index not in {1, 3}],
            device=base_env.device,
            dtype=torch.long,
        )
        default_velocity = _tensor(data.default_root_vel).detach().clone()

        before_first = _tensor(data.root_vel_w).detach().clone()
        event_cfg.func(base_env, selected, velocity_range=velocity_range, asset_cfg=asset_cfg)
        after_first = _tensor(data.root_vel_w).detach().clone()
        first_selected_delta = after_first[selected] - before_first[selected]
        first_unselected_delta = after_first[unselected] - before_first[unselected]

        # A reset must restore compile-time/default velocity before another
        # push, rather than carrying the previous event's impulse forward.
        env.reset(seed=args.seed + 1)
        after_reset = _tensor(data.root_vel_w).detach().clone()
        reset_error = after_reset - default_velocity

        before_second = after_reset.clone()
        event_cfg.func(base_env, selected, velocity_range=velocity_range, asset_cfg=asset_cfg)
        after_second = _tensor(data.root_vel_w).detach().clone()
        second_selected_delta = after_second[selected] - before_second[selected]
        second_unselected_delta = after_second[unselected] - before_second[unselected]

        expected_delta = torch.tensor([0.25, -0.10, 0.0, 0.0, 0.0, 0.0], device=base_env.device)
        checks = {
            "selected_subset": selected.tolist() == [1, 3],
            "before_finite": bool(torch.isfinite(before_first).all().item()),
            "after_finite": bool(torch.isfinite(after_first).all().item()),
            "selected_first_exact": bool(torch.allclose(first_selected_delta, expected_delta, atol=1e-6, rtol=0.0)),
            "unselected_first_unchanged": bool(torch.allclose(first_unselected_delta, torch.zeros_like(first_unselected_delta), atol=1e-7, rtol=0.0)),
            "reset_restores_default": bool(torch.allclose(reset_error, torch.zeros_like(reset_error), atol=1e-6, rtol=0.0)),
            "selected_second_exact": bool(torch.allclose(second_selected_delta, expected_delta, atol=1e-6, rtol=0.0)),
            "unselected_second_unchanged": bool(torch.allclose(second_unselected_delta, torch.zeros_like(second_unselected_delta), atol=1e-7, rtol=0.0)),
            "second_finite": bool(torch.isfinite(after_second).all().item()),
        }
        if not all(checks.values()):
            raise AssertionError(f"push runtime contract failed: {checks}")
        result.update(
            {
                "status": "PROVEN",
                "physics_manager": base_env.sim.physics_manager.__name__,
                "num_envs": args.num_envs,
                "selected_envs": selected.tolist(),
                "unselected_envs": unselected.tolist(),
                "velocity_range": velocity_range,
                "before_first": _summary(before_first),
                "after_first": _summary(after_first),
                "first_selected_delta": first_selected_delta.detach().cpu().tolist(),
                "first_unselected_delta": first_unselected_delta.detach().cpu().tolist(),
                "reset_error": _summary(reset_error),
                "before_second": _summary(before_second),
                "after_second": _summary(after_second),
                "second_selected_delta": second_selected_delta.detach().cpu().tolist(),
                "second_unselected_delta": second_unselected_delta.detach().cpu().tolist(),
                "checks": checks,
                "finite": checks["before_finite"] and checks["after_finite"] and checks["second_finite"],
                "accumulation": "reset_restores_default_before_second_push",
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "BLOCKED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
