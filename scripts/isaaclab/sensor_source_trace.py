"""Record raw IsaacLab source/adapters for the shared MJLab source trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def _tensor(value: object) -> torch.Tensor:
    return getattr(value, "torch", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg
        from isaaclab_microduck.tasks.velocity_flat_contact import _foot_site_state
        from isaaclab_microduck.tasks.parity import subtree_angular_momentum
        from scripts.sensor_source_trace_spec import TRACE_SCHEMA_VERSION, scripted_states

        register_tasks()
        cfg = make_velocity_flat_env_cfg(play=True, num_envs=1)
        cfg.seed = 2026
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        env.reset(seed=2026)
        robot = base_env.scene["robot"]
        joint_ids = {name: index for index, name in enumerate(robot.joint_names)}
        ordered_joint_ids = torch.as_tensor(
            [joint_ids[name] for name in cfg.actions.joint_pos.joint_names],
            device=base_env.device,
            dtype=torch.long,
        )
        states = scripted_states(args.steps)
        records: list[dict[str, object]] = []
        for state in states:
            root_pose = torch.tensor([state["root_pose_xyzw"]], dtype=torch.float32, device=base_env.device)
            root_velocity = torch.tensor([state["root_velocity_world"]], dtype=torch.float32, device=base_env.device)
            q = torch.tensor([state["joint_pos"]], dtype=torch.float32, device=base_env.device)
            qd = torch.tensor([state["joint_vel"]], dtype=torch.float32, device=base_env.device)
            q_sim = torch.zeros((1, len(robot.joint_names)), dtype=q.dtype, device=base_env.device)
            qd_sim = torch.zeros_like(q_sim)
            q_sim[:, ordered_joint_ids] = q
            qd_sim[:, ordered_joint_ids] = qd
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
            robot.write_joint_position_to_sim_index(position=q_sim)
            robot.write_joint_velocity_to_sim_index(velocity=qd_sim)
            base_env.scene.write_data_to_sim()
            base_env.sim.forward()
            base_env.scene.update(base_env.sim.cfg.dt)
            site_pos, site_vel = _foot_site_state(
                base_env,
                type("Cfg", (), {"name": "robot", "body_ids": [robot.body_names.index("ankle_left"), robot.body_names.index("ankle_right")]})(),
            )
            data = robot.data
            body_mass = _tensor(data.body_mass)
            body_pos = _tensor(data.body_com_pos_w)
            body_lin_vel = _tensor(data.body_com_lin_vel_w)
            body_ang_vel = _tensor(data.body_com_ang_vel_w)
            body_inertia = _tensor(data.body_inertia)
            body_quat = _tensor(data.body_com_quat_w)
            angmom = subtree_angular_momentum(body_mass, body_pos, body_lin_vel, body_ang_vel, body_inertia, body_quat)
            contact_sensor = base_env.scene.sensors["feet_ground_contact"]
            contact_force = getattr(contact_sensor.data, "net_forces_w", None)
            contact_force = _tensor(contact_force)[:, :2] if contact_force is not None else torch.zeros(1, 2, 3, device=base_env.device)
            contact_found = (torch.linalg.norm(contact_force, dim=-1) > 1.0).to(torch.float32).unsqueeze(-1)
            records.append(
                {
                    "step": state["step"],
                    "input": state,
                    "raw": {
                        "root_ang_vel_b_adapter": _tensor(data.root_ang_vel_b)[0].detach().cpu().tolist(),
                        "projected_gravity_b_adapter": _tensor(data.projected_gravity_b)[0].detach().cpu().tolist(),
                        "joint_pos": _tensor(data.joint_pos)[0, ordered_joint_ids].detach().cpu().tolist(),
                        "joint_vel": _tensor(data.joint_vel)[0, ordered_joint_ids].detach().cpu().tolist(),
                        "foot_site_pos_w": site_pos[0].detach().cpu().tolist(),
                        "foot_site_vel_w": site_vel[0].detach().cpu().tolist(),
                        "subtree_angmom_adapter": angmom[0].detach().cpu().tolist(),
                        "contact_found": contact_found[0].detach().cpu().tolist(),
                        "contact_force": contact_force[0].detach().cpu().tolist(),
                    },
                }
            )
        report = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "backend": "isaaclab",
            "num_steps": args.steps,
            "dt_s": 0.02,
            "source_notes": {
                "gyro": "Articulation root_ang_vel_b adapter; no named PhysX IMU sensor is consumed",
                "gravity": "Articulation projected_gravity_b adapter; no named PhysX IMU sensor is consumed",
                "feet": "canonical MJCF site offsets reconstructed from PhysX body tensors; ContactSensor net forces",
                "subtree_angular_momentum": "rigid-body COM/mass/inertia/angular-velocity reconstruction",
            },
            "processing": {
                "control_dt_s": 0.02,
                "delay_index": 0,
                "reset_behavior": "explicit seeded reset before scripted writes; no partial reset in trace",
            },
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"backend": report["backend"], "records": len(records), "output": str(args.output)}, indent=2))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
