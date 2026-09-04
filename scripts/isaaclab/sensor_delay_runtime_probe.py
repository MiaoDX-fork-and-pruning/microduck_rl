"""Observe one-control-step actor gyro delay on the concrete IsaacLab task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import torch

from isaaclab.app import AppLauncher


def _tensor(value: torch.Tensor) -> torch.Tensor:
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

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat_sensors import sensor_corruption
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        env = gym.make(
            "IsaacLab-Velocity-Flat-MicroDuck",
            cfg=make_velocity_flat_env_cfg(num_envs=args.num_envs),
        )
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        env.reset(seed=2026)

        # Raise the articulation briefly so contacts do not erase the injected
        # angular signal before the delayed observation can read it.
        pose = robot.data.default_root_pose.torch.clone()
        pose[:, 2] = 0.60
        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros_like(robot.data.default_root_vel.torch)
        )
        base_env.sim.step()
        base_env.scene.update(base_env.sim.cfg.dt)

        injected = [0.0, 0.8, -0.6, 0.4]
        raw_values: list[list[float]] = []
        delayed_values: list[list[float]] = []
        transitions: list[dict[str, object]] = []
        previous_raw = None
        for value in injected:
            root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
            root_velocity[:, 5] = value
            robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
            base_env.sim.step()
            base_env.scene.update(base_env.sim.cfg.dt)
            raw = _tensor(robot.data.root_ang_vel_b).detach().clone()
            delayed = sensor_corruption(
                base_env, "gyro_runtime", raw, noise=0.0, delay=1
            ).detach().clone()
            raw_values.append(raw[:, 2].float().cpu().tolist())
            delayed_values.append(delayed[:, 2].float().cpu().tolist())
            transition = {
                "raw": raw[:, 2].float().cpu().tolist(),
                "delayed": delayed[:, 2].float().cpu().tolist(),
                "matches_previous_raw": previous_raw is None
                or bool(torch.allclose(delayed[:, 2], previous_raw[:, 2], atol=1e-6, rtol=0.0)),
                "raw_changed": previous_raw is not None
                and bool((raw[:, 2] - previous_raw[:, 2]).abs().max().item() > 1e-5),
            }
            transitions.append(transition)
            previous_raw = raw

        finite = all(
            torch.isfinite(torch.tensor(values)).all().item()
            for values in raw_values + delayed_values
        )
        warm_transitions = transitions[1:]
        changed = any(bool(item["raw_changed"]) for item in warm_transitions)
        lag_matches = all(bool(item["matches_previous_raw"]) for item in warm_transitions)
        if not finite or not changed or not lag_matches:
            raise AssertionError(
                f"runtime gyro delay contract failed: finite={finite}, "
                f"changed={changed}, lag_matches={lag_matches}"
            )
        report = {
            "num_envs": args.num_envs,
            "injected_yaw_rates": injected,
            "raw_values": raw_values,
            "delayed_values": delayed_values,
            "transitions": transitions,
            "delay_steps": 1,
            "finite": finite,
            "dynamic_signal_observed": changed,
            "one_step_lag_matches": lag_matches,
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
