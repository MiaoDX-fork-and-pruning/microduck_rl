"""Collect seeded runtime evidence for Velocity-Flat reset/DR/sensor/commands."""

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


def _difference_summary(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, object]:
    delta = (_tensor(observed) - _tensor(reference)).detach().float()
    return {
        "shape": list(delta.shape),
        "finite": bool(torch.isfinite(delta).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
    }


def _axis_summary(value: torch.Tensor) -> list[dict[str, object]]:
    value = _tensor(value).detach().float()
    return [_summary(value[:, index]) for index in range(value.shape[1])]


def _yaw_from_xyzw(quat: torch.Tensor) -> torch.Tensor:
    qx, qy, qz, qw = quat.unbind(dim=-1)
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy.square() + qz.square()),
    )


def _assert_runtime_contract(record: dict[str, object], *, num_envs: int) -> None:
    """Fail the probe when sampled runtime state leaves the mjlab bounds."""

    relative = record["root_pos_relative"]
    assert relative["shape"] == [num_envs, 3]
    assert -0.500001 <= relative["min"] <= 0.500001
    assert -0.500001 <= relative["max"] <= 0.500001
    assert 0.1199 <= record["root_pos_relative_axes"][2]["min"] <= 0.1301
    assert 0.1199 <= record["root_pos_relative_axes"][2]["max"] <= 0.1301
    assert -3.1416 <= record["root_yaw_rad"]["min"] <= record["root_yaw_rad"]["max"] <= 3.1416
    assert record["joint_default_error_max"] <= 1.0e-6
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
    assert all(item["finite"] for item in record["sensor_effects"].values())
    assert record["sensor_effects"]["joint_pos"]["max_abs"] <= 0.016
    assert record["sensor_effects"]["joint_vel"]["max_abs"] <= 0.25
    # Gyro/gravity include the episode-stable <=6 degree mounting rotation;
    # their coordinate-wise delta is therefore larger than the additive noise
    # bound alone.  The loose bound catches explosions while preserving that
    # intended corruption source.
    assert record["sensor_effects"]["gyro"]["max_abs"] <= 0.15
    assert record["sensor_effects"]["gravity"]["max_abs"] <= 0.15


def _command_change(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, object]:
    values = []
    for name in before:
        values.append((_tensor(after[name]) - _tensor(before[name])).abs())
    delta = torch.cat([value.reshape(value.shape[0], -1) for value in values], dim=-1)
    return {
        "max_abs": float(delta.max().item()),
        "changed_envs": int((delta.max(dim=1).values > 1.0e-8).sum().item()),
        "finite": bool(torch.isfinite(delta).all().item()),
    }


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
        from isaaclab_microduck.tasks.velocity_flat import (
            make_velocity_flat_env_cfg,
            policy_gyro,
            policy_joint_pos,
            policy_joint_vel,
            policy_projected_gravity,
        )

        register_tasks()
        cfg = make_velocity_flat_env_cfg(num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        records = []
        command_change_records = []
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
            root_yaw = _yaw_from_xyzw(_tensor(data.root_link_quat_w))
            joint_default_error = torch.abs(
                _tensor(data.joint_pos) - _tensor(data.default_joint_pos)
            ).max(dim=1).values
            record = {
                "reset_index": reset_index,
                "root_pos": _summary(root_pos),
                "root_pos_relative": _summary(root_pos_relative),
                "root_pos_relative_axes": _axis_summary(root_pos_relative),
                "root_yaw_rad": _summary(root_yaw.unsqueeze(1)),
                "joint_default_error_max": float(joint_default_error.max().item()),
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
            print(f"ISAACLAB_PARITY_PROBE:reset_{reset_index}:sensor_effects_start", flush=True)
            sensor_pairs = (
                ("gyro", policy_gyro(base_env, corrupt=False), policy_gyro(base_env, corrupt=True)),
                (
                    "gravity",
                    policy_projected_gravity(base_env, corrupt=False),
                    policy_projected_gravity(base_env, corrupt=True),
                ),
                ("joint_pos", policy_joint_pos(base_env, biased=False), policy_joint_pos(base_env, biased=True)),
                ("joint_vel", policy_joint_vel(base_env, corrupt=False), policy_joint_vel(base_env, corrupt=True)),
            )
            record["sensor_effects"] = {
                name: _difference_summary(clean, corrupt)
                for name, clean, corrupt in sensor_pairs
            }
            print(f"ISAACLAB_PARITY_PROBE:reset_{reset_index}:sensor_effects_done", flush=True)
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
            before_commands = {
                name: _tensor(base_env.command_manager.get_command(name)).detach().clone()
                for name in ("base_velocity", "head_pose", "body_pose")
            }
            for _ in range(args.steps):
                env.step(torch.zeros((args.num_envs, ACTION_SIZE), device=base_env.device))
            after_commands = {
                name: _tensor(base_env.command_manager.get_command(name)).detach().clone()
                for name in ("base_velocity", "head_pose", "body_pose")
            }
            command_change_records.append(_command_change(before_commands, after_commands))
        report = {
            "seed": args.seed,
            "num_envs": args.num_envs,
            "resets": args.resets,
            "steps_between_resets": args.steps,
            "records": records,
            "command_change_records": command_change_records,
            "finite": all(
                bool(item["root_pos"]["finite"])
                and bool(item["joint_pos"]["finite"])
                and bool(item["root_pos_relative"]["finite"])
                and bool(item["actor_obs"]["finite"])
                for item in records
            ),
        }
        if not all(item["finite"] for item in command_change_records):
            raise AssertionError("command resampling produced non-finite values")
        if args.steps * 0.02 >= 2.0 and not any(item["changed_envs"] > 0 for item in command_change_records):
            raise AssertionError("command terms did not resample during the runtime probe")
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
    except BaseException:
        print("ISAACLAB_PARITY_PROBE:failure", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
