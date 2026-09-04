"""Collect seeded runtime evidence for Velocity-Flat reset/DR/sensor/commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _axis_summary(value: torch.Tensor) -> list[dict[str, object]]:
    value = _tensor(value).detach().float()
    return [_summary(value[:, index]) for index in range(value.shape[1])]


def _assert_runtime_contract(record: dict[str, object], *, num_envs: int) -> None:
    """Fail the probe when sampled runtime state leaves the mjlab bounds."""

    relative = record["root_pos_relative"]
    assert relative["shape"] == [num_envs, 3]
    assert -0.500001 <= relative["min"] <= 0.500001
    assert -0.500001 <= relative["max"] <= 0.500001
    assert 0.1199 <= record["root_pos_relative_axes"][2]["min"] <= 0.1301
    assert 0.1199 <= record["root_pos_relative_axes"][2]["max"] <= 0.1301
    assert record["joint_pos"]["shape"] == [num_envs, 14]
    assert record["actor_obs"]["shape"] == [num_envs, 61]
    assert 3.0 <= record["_delay"]["min"] <= record["_delay"]["max"] <= 6.0
    assert 6.5 <= record["_supply_voltage"]["min"] <= record["_supply_voltage"]["max"] <= 8.2
    assert 0.0 <= record["_vin_drop_gain"]["min"] <= record["_vin_drop_gain"]["max"] <= 0.2
    assert 0.9 <= record["_friction_scale"]["min"] <= record["_friction_scale"]["max"] <= 1.1
    assert record["commands"]["base_velocity"]["shape"] == [num_envs, 3]
    assert record["commands"]["head_pose"]["shape"] == [num_envs, 4]
    assert record["commands"]["body_pose"]["shape"] == [num_envs, 6]
    assert record["actor_obs"]["finite"] and record["joint_pos"]["finite"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--resets", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
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
        cfg.seed = args.seed
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        records = []
        for reset_index in range(args.resets):
            obs, _ = env.reset(seed=args.seed + reset_index)
            commands = {
                name: _summary(base_env.command_manager.get_command(name))
                for name in ("base_velocity", "head_pose", "body_pose")
            }
            data = robot.data
            root_pos = _tensor(data.root_link_pos_w)
            origins = _tensor(base_env.scene.env_origins)
            root_pos_relative = root_pos - origins
            record = {
                "reset_index": reset_index,
                "root_pos": _summary(root_pos),
                "root_pos_relative": _summary(root_pos_relative),
                "root_pos_relative_axes": _axis_summary(root_pos_relative),
                "joint_pos": _summary(data.joint_pos),
                "joint_vel": _summary(data.joint_vel),
                "commands": commands,
                "turn_bucket_count": int(
                    ((torch.abs(_tensor(base_env.command_manager.get_command("base_velocity"))[:, :2]).sum(dim=1) < 1.0e-6)
                    & (torch.abs(_tensor(base_env.command_manager.get_command("base_velocity"))[:, 2]) >= 0.4)
                ).sum().item()
                ),
                "turn_bucket_fraction": float(
                    ((torch.abs(_tensor(base_env.command_manager.get_command("base_velocity"))[:, :2]).sum(dim=1) < 1.0e-6)
                    & (torch.abs(_tensor(base_env.command_manager.get_command("base_velocity"))[:, 2]) >= 0.4)
                ).float().mean().item()
                ),
                "actor_obs": _summary(obs["policy"] if isinstance(obs, dict) else obs),
            }
            for name in ("_encoder_bias", "_imu_mount_quat", "_gyro_lag", "_gravity_lag", "_joint_vel_history"):
                value = getattr(base_env, name, None)
                if value is not None:
                    record[name] = _summary(value)
            actuator = next(iter(robot.actuators.values()))
            for name in ("_supply_voltage", "_vin_drop_gain", "_friction_scale", "_delay"):
                value = getattr(actuator, name, None)
                if value is None:
                    continue
                if name == "_delay":
                    value = value.delay
                record[name] = _summary(value)
            records.append(record)
            _assert_runtime_contract(record, num_envs=args.num_envs)
            for _ in range(args.steps):
                env.step(torch.zeros((args.num_envs, ACTION_SIZE), device=base_env.device))
        report = {
            "seed": args.seed,
            "num_envs": args.num_envs,
            "resets": args.resets,
            "steps_between_resets": args.steps,
            "records": records,
            "finite": all(
                bool(item["root_pos"]["finite"])
                and bool(item["joint_pos"]["finite"])
                and bool(item["root_pos_relative"]["finite"])
                and bool(item["actor_obs"]["finite"])
                for item in records
            ),
        }
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
