"""Benchmark a causal one-step-lag PhysX external-load friction bridge.

This is a diagnostic only.  It does not modify the production BAM actuator or
Velocity-Flat task.  PhysX force getters are sampled after ``sim.step`` and
``scene.update``; the resulting projected-minus-actuation effort is applied to
the friction fields on the following pre-step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

from isaaclab_microduck.actuators.physx_friction_bridge import LaggedExternalEffort


def _torch(value):
    import warp as wp

    return wp.to_torch(value)


def _restore_home(robot, home_target: torch.Tensor) -> None:
    root_pose = robot.data.default_root_pose.torch.clone()
    root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
    robot.write_joint_position_to_sim_index(position=home_target.clone())
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.default_joint_vel.torch))
    robot.reset()


def _run_case(scene, sim, robot, actuator, home_target, *, mode: str, scale: float, steps: int) -> dict[str, object]:
    actuator.set_friction_scale(scale)
    _restore_home(robot, home_target)
    bridge = LaggedExternalEffort(robot.num_instances, robot.num_joints, device=robot.device)
    target = home_target.clone()
    target[:, 0] += 0.2
    view = robot.root_view
    rows: list[dict[str, float | bool]] = []
    reset_external_norm = float(bridge.external_effort().abs().max().item())
    dt = float(sim.cfg.dt)
    for step in range(steps):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        motor_effort = actuator.applied_effort
        joint_velocity = robot.data.joint_vel.torch
        external_effort = bridge.external_effort() if mode == "one_step_lag" else torch.zeros_like(motor_effort)
        static, dynamic, viscous = actuator.physx_friction_coefficients(
            motor_effort, external_effort, joint_velocity
        )
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=static,
            joint_dynamic_friction_coeff=dynamic,
            joint_viscous_friction_coeff=viscous,
        )
        sim.step()
        scene.update(dt)
        projected = _torch(view.get_dof_projected_joint_forces())
        actuation = _torch(view.get_dof_actuation_forces())
        observed = bridge.observe(projected, actuation)
        rows.append(
            {
                "step": step,
                "used_external_effort_max_abs": float(external_effort.abs().max().item()),
                "observed_external_effort_max_abs": float(observed.abs().max().item()),
                "static_friction_max": float(static.max().item()),
                "joint_velocity_max_abs": float(joint_velocity.abs().max().item()),
                "finite": bool(torch.isfinite(external_effort).all().item()),
            }
        )
    return {
        "mode": mode,
        "friction_scale": scale,
        "steps": steps,
        "dt": dt,
        "reset_external_effort_max_abs": reset_external_norm,
        "first_step_uses_zero_external": rows[0]["used_external_effort_max_abs"] == 0.0,
        "finite": all(bool(row["finite"]) for row in rows),
        "peak_observed_external_effort": max(float(row["observed_external_effort_max_abs"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=80)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.configclass import configclass
        from isaaclab_microduck.assets.microduck import MICRODUCK_CFG, policy_target_to_sim
        from isaaclab_microduck.policy_abi import HOME_POSITION

        sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot = scene.articulations["robot"]
        actuator = next(iter(robot.actuators.values()))
        policy_home = torch.as_tensor(HOME_POSITION, device=robot.device).reshape(1, -1)
        home_target = policy_target_to_sim(policy_home, robot.joint_names)
        cases = [
            _run_case(scene, sim, robot, actuator, home_target, mode=mode, scale=scale, steps=args.steps)
            for mode in ("motor_only", "one_step_lag")
            for scale in (0.5, 1.0, 1.5)
        ]
        report = {
            "joint_names": list(robot.joint_names),
            "force_source": "projected_joint_forces_minus_actuation_forces",
            "sample_phase": "after_scene_update",
            "apply_phase": "next_pre_step",
            "lag_steps": 1,
            "production_wiring": "diagnostic_only_not_applied",
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
