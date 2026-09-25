"""Run a small real-USD IsaacLab BAM actuator response bench.

This is intentionally a bring-up bench, not a training harness.  It records
the first servo joint under target steps and sinusoidal targets at two supply
voltages.  The explicit actuator's friction budget is reported separately;
it is not silently applied as a PhysX joint friction property.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def _run_case(scene, sim, robot, actuator, name: str, voltage: float, target_fn, steps: int, dt: float) -> dict:
    actuator.set_supply_voltage(voltage)
    samples: list[dict[str, float]] = []
    for step in range(steps):
        target = target_fn(step * dt)
        command = torch.zeros((robot.num_instances, robot.num_joints), device=robot.device)
        command[:, 0] = target
        robot.set_joint_position_target(command)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        samples.append(
            {
                "time": (step + 1) * dt,
                "target": float(target),
                "position": float(robot.data.joint_pos.torch[0, 0]),
                "velocity": float(robot.data.joint_vel.torch[0, 0]),
                "effort": float(actuator.applied_effort[0, 0]),
            }
        )
    positions = [sample["position"] for sample in samples]
    efforts = [sample["effort"] for sample in samples]
    return {
        "name": name,
        "voltage": voltage,
        "steps": steps,
        "final_position": positions[-1],
        "peak_abs_effort": max(abs(value) for value in efforts),
        "peak_abs_velocity": max(abs(sample["velocity"]) for sample in samples),
        "samples": samples,
    }


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
        from isaaclab_microduck.assets.microduck import MICRODUCK_CFG

        sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot = scene.articulations["robot"]
        actuator = next(iter(robot.actuators.values()))
        dt = sim.cfg.dt
        cases = [
            _run_case(scene, sim, robot, actuator, "step_7.5v", 7.5, lambda t: 0.2, args.steps, dt),
            _run_case(scene, sim, robot, actuator, "step_6.5v", 6.5, lambda t: 0.2, args.steps, dt),
            _run_case(
                scene,
                sim,
                robot,
                actuator,
                "sine_7.5v",
                7.5,
                lambda t: 0.15 * torch.sin(torch.tensor(2.0 * 3.141592653589793 * 1.0 * t, device=robot.device)),
                args.steps,
                dt,
            ),
        ]
        report = {
            "dt": dt,
            "joint": robot.joint_names[0],
            "actuator": type(actuator).__name__,
            "friction_bridge": "not_applied_to_physx",
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
