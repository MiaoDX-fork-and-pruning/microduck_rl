"""Run the deterministic IsaacLab physics bring-up battery.

The battery deliberately keeps the scenarios small and observable.  It is a
runtime sanity check for the converted Microduck articulation, not a claim of
MuJoCo/PhysX parity: contact-force, slip and BAM friction comparisons remain
explicitly listed as pending until their bridges are implemented.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch

from isaaclab.app import AppLauncher


def _finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _apply_friction_bridge(robot, actuator) -> str:
    """Apply the available BAM budget through IsaacLab's PhysX joint API.

    The explicit actuator callback does not expose solved external joint load,
    so this first bridge uses the motor-only budget and records that limitation.
    """

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
    return "motor_only_external_effort_unavailable"


def _tilt_rad(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Return angle from upright for IsaacLab's xyzw quaternion layout."""

    scalar = torch.clamp(torch.abs(quat_xyzw[..., 3]), max=1.0)
    return 2.0 * torch.acos(scalar)


def _restore_home(robot, home_target: torch.Tensor) -> None:
    print("ISAACLAB_PHYSICS_BATTERY:restore_home:start", flush=True)
    root_pose = robot.data.default_root_pose.torch.clone()
    root_vel = robot.data.default_root_vel.torch.clone()
    print("ISAACLAB_PHYSICS_BATTERY:restore_home:defaults_read", flush=True)
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    print("ISAACLAB_PHYSICS_BATTERY:restore_home:pose_written", flush=True)
    robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
    print("ISAACLAB_PHYSICS_BATTERY:restore_home:velocity_written", flush=True)
    robot.write_joint_position_to_sim_index(position=home_target.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()


def _run_case(scene, sim, robot, name: str, prepare: Callable[[], None], target: torch.Tensor, steps: int) -> dict:
    print(f"ISAACLAB_PHYSICS_BATTERY:case:{name}:prepare", flush=True)
    prepare()
    print(f"ISAACLAB_PHYSICS_BATTERY:case:{name}:prepared", flush=True)
    dt = float(sim.cfg.dt)
    heights: list[float] = []
    tilts: list[float] = []
    speeds: list[float] = []
    efforts: list[float] = []
    finite = True
    for _ in range(steps):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        friction_bridge = _apply_friction_bridge(robot, next(iter(robot.actuators.values())))
        sim.step()
        scene.update(dt)
        # Isaac Sim 6.0.1 has a broken shorthand root_pos_w/root_quat_w view for this imported
        # asset; the explicit link accessors remain valid.
        root_pos = robot.data.root_link_pos_w.torch
        root_quat = robot.data.root_link_quat_w.torch
        joint_vel = robot.data.joint_vel.torch
        applied = robot.data.applied_torque.torch
        finite = finite and _finite(root_pos, root_quat, joint_vel, applied)
        heights.append(float(root_pos[0, 2]))
        tilts.append(float(_tilt_rad(root_quat[0])))
        speeds.append(float(torch.abs(joint_vel[0]).max()))
        efforts.append(float(torch.abs(applied[0]).max()))
    return {
        "name": name,
        "steps": steps,
        "duration_s": steps * dt,
        "finite": finite,
        "initial_height_m": heights[0],
        "final_height_m": heights[-1],
        "min_height_m": min(heights),
        "max_tilt_rad": max(tilts),
        "peak_joint_speed_rad_s": max(speeds),
        "peak_abs_effort_nm": max(efforts),
        "friction_bridge": friction_bridge,
    }


def main() -> None:
    print("ISAACLAB_PHYSICS_BATTERY:main:start", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    print("ISAACLAB_PHYSICS_BATTERY:app_ready", flush=True)
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.configclass import configclass
        from isaaclab_microduck.assets.microduck import MICRODUCK_CFG

        sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))
        print("ISAACLAB_PHYSICS_BATTERY:sim_ready", flush=True)

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
        print("ISAACLAB_PHYSICS_BATTERY:scene_ready", flush=True)
        # Spawn the plane after InteractiveScene cloning.  Isaac Sim 6.0.1
        # exits during scene construction when the plane is an entity config.
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        print("ISAACLAB_PHYSICS_BATTERY:ground_ready", flush=True)
        sim.reset()
        print("ISAACLAB_PHYSICS_BATTERY:sim_reset", flush=True)
        robot = scene.articulations["robot"]
        print("ISAACLAB_PHYSICS_BATTERY:robot_ready", flush=True)
        from isaaclab_microduck.policy_abi import HOME_POSITION

        home_target = torch.as_tensor(HOME_POSITION, device=robot.device).reshape(1, -1)
        print(f"ISAACLAB_PHYSICS_BATTERY:defaults joint={home_target[0].tolist()}", flush=True)

        def home() -> None:
            _restore_home(robot, home_target)

        def free_fall() -> None:
            _restore_home(robot, home_target)
            pose = robot.data.default_root_pose.torch.clone()
            pose[:, 2] = 0.45
            robot.write_root_pose_to_sim_index(root_pose=pose)
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros_like(robot.data.default_root_vel.torch)
            )
            robot.reset()

        def step() -> None:
            _restore_home(robot, home_target)

        step_target = home_target.clone()
        step_target[:, 0] += 0.2
        cases = [
            _run_case(scene, sim, robot, "home_settle", home, home_target, args.steps),
            _run_case(scene, sim, robot, "free_fall", free_fall, home_target, args.steps),
            _run_case(scene, sim, robot, "target_step", step, step_target, args.steps),
            _run_case(scene, sim, robot, "nan_soak", home, home_target, args.steps),
        ]
        report = {
            "dt": float(sim.cfg.dt),
            "robot": "microduck_walk.usd",
            "actuator": type(next(iter(robot.actuators.values()))).__name__,
            "contact_force_parity": "pending",
            "slip_parity": "pending",
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
