"""Run a fixed-root, one-joint BAM friction sweep in IsaacLab PhysX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.configclass import configclass
        from isaaclab_microduck.assets.microduck import MICRODUCK_CFG, policy_target_to_sim
        from isaaclab_microduck.policy_abi import HOME_POSITION

        robot_cfg = MICRODUCK_CFG.copy()
        sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot = scene.articulations["robot"]
        actuator = next(iter(robot.actuators.values()))
        dt = float(sim.cfg.dt)
        root_pose = robot.data.default_root_pose.torch.clone()
        root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
        policy_home = torch.as_tensor(HOME_POSITION, device=robot.device).reshape(1, -1)
        home = policy_target_to_sim(policy_home, robot.joint_names)
        cases = []
        for scale in (0.5, 1.0, 1.5):
            zeros = home.clone()
            robot.write_joint_position_to_sim_index(position=zeros)
            robot.write_joint_velocity_to_sim_index(velocity=zeros)
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
            robot.reset()
            actuator.set_friction_scale(scale)
            policy_target = policy_home.clone()
            policy_target[:, 0] += 0.2
            target = policy_target_to_sim(policy_target, robot.joint_names)
            samples = []
            for step in range(args.steps):
                # Imported MJCF roots cannot use PhysX's fix_root_link joint in
                # this runtime. Re-apply the root state before each solve to
                # isolate the first joint while preserving the real actuator.
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                motor_effort = actuator.applied_effort
                velocity = robot.data.joint_vel.torch
                external_effort = torch.zeros_like(motor_effort)
                static, dynamic, viscous = actuator.physx_friction_coefficients(
                    motor_effort, external_effort, velocity
                )
                robot.write_joint_friction_coefficient_to_sim_index(
                    joint_friction_coeff=static,
                    joint_dynamic_friction_coeff=dynamic,
                    joint_viscous_friction_coeff=viscous,
                )
                sim.step()
                scene.update(dt)
                samples.append(
                    {
                        "time_s": (step + 1) * dt,
                        "position_rad": float(robot.data.joint_pos.torch[0, 0]),
                        "velocity_rad_s": float(robot.data.joint_vel.torch[0, 0]),
                        "effort_nm": float(actuator.applied_effort[0, 0]),
                        "static_friction_nm": float(static[0, 0]),
                    }
                )
            cases.append(
                {
                    "friction_scale": scale,
                    "final_position_rad": samples[-1]["position_rad"],
                    "peak_abs_velocity_rad_s": max(abs(s["velocity_rad_s"]) for s in samples),
                    "peak_static_friction_nm": max(s["static_friction_nm"] for s in samples),
                    "finite": all(
                        torch.isfinite(torch.tensor(list(sample.values()))).all().item()
                        for sample in samples
                    ),
                    "samples": samples,
                }
            )
        report = {
            "joint": "left_hip_yaw",
            "steps": args.steps,
            "dt": dt,
            "fixed_root": "kinematic_state_reset",
            "friction_bridge": "motor_only_external_effort_unavailable",
            "cases": cases,
        }
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
    finally:
        app.close()


if __name__ == "__main__":
    main()
