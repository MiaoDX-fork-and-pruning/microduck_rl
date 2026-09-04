"""Probe Velocity-Flat asset DR writes on the concrete IsaacLab articulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from isaaclab.app import AppLauncher


def _tensor(value):
    return getattr(value, "torch", value)


def _summary(value: torch.Tensor) -> dict[str, object]:
    value = _tensor(value).detach().float()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
    }


def _delta(before: torch.Tensor, after: torch.Tensor) -> dict[str, object]:
    value = (_tensor(after) - _tensor(before)).detach().float()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "max_abs": float(value.abs().max().item()),
        "mean_abs": float(value.abs().mean().item()),
    }


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

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        env = gym.make(
            "IsaacLab-Velocity-Flat-MicroDuck",
            cfg=make_velocity_flat_env_cfg(num_envs=args.num_envs),
        )
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        env.reset(seed=2026)
        data = robot.data
        first = {
            "armature": _tensor(data.joint_armature).detach().clone(),
            "mass": _tensor(data.body_mass).detach().clone(),
            "inertia": _tensor(data.body_inertia).detach().clone(),
            "com": _tensor(data.body_com_pose_b).detach().clone(),
            "friction": _tensor(data.joint_friction_coeff).detach().clone(),
            "damping": _tensor(data.joint_damping).detach().clone(),
        }
        # A second reset samples reset-time DR from the same compile-time
        # defaults.  This must vary, but must not drift by accumulating the
        # previous sample.
        env.reset(seed=2027)
        second = {
            "armature": _tensor(data.joint_armature).detach().clone(),
            "mass": _tensor(data.body_mass).detach().clone(),
            "inertia": _tensor(data.body_inertia).detach().clone(),
            "com": _tensor(data.body_com_pose_b).detach().clone(),
            "friction": _tensor(data.joint_friction_coeff).detach().clone(),
            "damping": _tensor(data.joint_damping).detach().clone(),
        }
        all_finite = all(
            bool(torch.isfinite(value).all().item())
            for value in (*first.values(), *second.values())
        )
        reset_deltas = {name: _delta(first[name], second[name]) for name in first}
        # Startup mass/inertia is deliberately stable across resets; CoM and
        # armature are reset-time samples and should show a bounded change.
        stable_startup = (
            torch.equal(first["mass"], second["mass"])
            and torch.equal(first["inertia"], second["inertia"])
        )
        reset_dr_effect = (
            reset_deltas["armature"]["max_abs"] > 1.0e-8
            and reset_deltas["com"]["max_abs"] > 1.0e-8
        )
        friction_zero = bool(torch.allclose(second["friction"], torch.zeros_like(second["friction"])))
        damping_zero = bool(torch.allclose(second["damping"], torch.zeros_like(second["damping"])))
        if not all_finite or not stable_startup or not reset_dr_effect or not friction_zero or not damping_zero:
            raise AssertionError(
                "asset dynamics contract failed: "
                f"finite={all_finite}, stable_startup={stable_startup}, "
                f"reset_dr_effect={reset_dr_effect}, friction_zero={friction_zero}, damping_zero={damping_zero}"
            )
        report = {
            "num_envs": args.num_envs,
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "first": {name: _summary(value) for name, value in first.items()},
            "second": {name: _summary(value) for name, value in second.items()},
            "reset_deltas": reset_deltas,
            "startup_mass_inertia_stable": stable_startup,
            "reset_dr_effect_observed": reset_dr_effect,
            "physx_joint_friction_zero": friction_zero,
            "physx_joint_damping_zero": damping_zero,
            "friction_bridge": "motor_only_external_effort_unavailable",
            "finite": all_finite,
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
