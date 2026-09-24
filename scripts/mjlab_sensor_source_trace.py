"""Record raw MJLab sensor and adapter values for the shared source trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from scripts.sensor_source_trace_spec import (
        TRACE_SCHEMA_VERSION,
        mujoco_pose_wxyz,
        scripted_states,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from sensor_source_trace_spec import (  # type: ignore[no-redef]
        TRACE_SCHEMA_VERSION,
        mujoco_pose_wxyz,
        scripted_states,
    )


def _tensor(value: object) -> torch.Tensor:
    return getattr(value, "torch", value)


def _sensor(model, data, suffix: str) -> list[float]:
    matches = [i for i in range(model.nsensor) if str(model.sensor(i).name).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one sensor ending {suffix!r}, got {matches}")
    sensor_id = matches[0]
    start = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return _tensor(data.sensordata)[0, start : start + dim].detach().cpu().tolist()


def _site_values(robot, names: tuple[str, ...]) -> tuple[list[list[float]], list[list[float]]]:
    ids = [robot.site_names.index(name) for name in names]
    pos = _tensor(robot.data.site_pos_w)[0, ids].detach().cpu().tolist()
    vel = _tensor(robot.data.site_lin_vel_w)[0, ids].detach().cpu().tolist()
    return pos, vel


def collect(num_steps: int) -> dict[str, object]:
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

    cfg = make_microduck_velocity_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset(seed=2026)
        robot = env.scene["robot"]
        states = scripted_states(num_steps)
        records: list[dict[str, object]] = []
        for state in states:
            root_pose = torch.tensor([mujoco_pose_wxyz(state["root_pose_xyzw"])], dtype=torch.float32)
            root_velocity = torch.tensor([state["root_velocity_world"]], dtype=torch.float32)
            joint_pos = torch.tensor([state["joint_pos"]], dtype=torch.float32)
            joint_vel = torch.tensor([state["joint_vel"]], dtype=torch.float32)
            robot.data.write_root_pose(root_pose)
            robot.data.write_root_velocity(root_velocity)
            robot.data.write_joint_state(joint_pos, joint_vel)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.sim.sense()
            site_pos, site_vel = _site_values(robot, ("left_foot", "right_foot"))
            records.append(
                {
                    "step": state["step"],
                    "input": state,
                    "raw": {
                        "root_ang_vel_b_adapter": _tensor(robot.data.root_link_ang_vel_b)[0].detach().cpu().tolist(),
                        "projected_gravity_b_adapter": _tensor(robot.data.projected_gravity_b)[0].detach().cpu().tolist(),
                        "joint_pos": _tensor(robot.data.joint_pos)[0].detach().cpu().tolist(),
                        "joint_vel": _tensor(robot.data.joint_vel)[0].detach().cpu().tolist(),
                        "named_gyro": _sensor(env.sim.mj_model, env.sim.data, "angular-velocity"),
                        "named_imu_gyro": _sensor(env.sim.mj_model, env.sim.data, "imu_ang_vel"),
                        "named_root_angmom": _sensor(env.sim.mj_model, env.sim.data, "root_angmom"),
                        "foot_site_pos_w": site_pos,
                        "foot_site_vel_w": site_vel,
                        "contact_found": [
                            _sensor(env.sim.mj_model, env.sim.data, "left_foot_collision_found"),
                            _sensor(env.sim.mj_model, env.sim.data, "right_foot_collision_found"),
                        ],
                        "contact_force": [
                            _sensor(env.sim.mj_model, env.sim.data, "left_foot_collision_force"),
                            _sensor(env.sim.mj_model, env.sim.data, "right_foot_collision_force"),
                        ],
                    },
                }
            )
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "backend": "mjlab",
            "num_steps": num_steps,
            "dt_s": 0.02,
            "source_notes": {
                "gyro": "named MuJoCo angular-velocity and imu_ang_vel sensors plus root-link adapter",
                "gravity": "root-link quaternion projected into body frame; no equivalent named Isaac sensor",
                "feet": "named MJCF sites and MuJoCo contact sensors",
                "subtree_angular_momentum": "named MuJoCo root_angmom sensor",
            },
            "processing": {
                "control_dt_s": 0.02,
                "delay_index": 0,
                "reset_behavior": "explicit seeded reset before scripted writes; no partial reset in trace",
            },
            "records": records,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"backend": report["backend"], "records": len(report["records"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
