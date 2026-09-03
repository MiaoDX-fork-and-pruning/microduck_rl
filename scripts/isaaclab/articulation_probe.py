"""Instantiate the converted Microduck articulation in Isaac Sim 6.0.1.

The probe emits stage markers so a slow Kit/PhysX startup cannot be confused
with an actuator or asset failure.
"""

from __future__ import annotations

from isaacsim import SimulationApp


def main() -> None:
    print("ISAACLAB_ARTICULATION_PROBE:app_imported", flush=True)
    app = SimulationApp({"headless": True})
    print("ISAACLAB_ARTICULATION_PROBE:app_ready", flush=True)
    try:
        import torch
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.configclass import configclass
        from isaaclab_microduck.assets.microduck import MICRODUCK_CFG

        print("ISAACLAB_ARTICULATION_PROBE:imports_ready", flush=True)
        sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
        print("ISAACLAB_ARTICULATION_PROBE:sim_ready", flush=True)

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
        print("ISAACLAB_ARTICULATION_PROBE:scene_ready", flush=True)
        sim.reset()
        print("ISAACLAB_ARTICULATION_PROBE:sim_reset", flush=True)
        robot = scene.articulations["robot"]
        actuator = next(iter(robot.actuators.values()))
        limits = robot.data.joint_effort_limits.torch
        print(
            "ISAACLAB_ARTICULATION_PROBE:articulation_ready "
            f"joints={robot.num_joints} actuator={type(actuator).__name__} "
            f"effort_limit={limits[0, :3].tolist()}",
            flush=True,
        )
        robot.set_joint_position_target(torch.ones(1, 14, device="cuda:0") * 0.1)
        for _ in range(3):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.cfg.dt)
        print(
            "ISAACLAB_ARTICULATION_PROBE:step_ok "
            f"joint_pos={robot.data.joint_pos.torch[0, :3].tolist()} "
            f"applied={robot.data.applied_torque.torch[0, :3].tolist()}",
            flush=True,
        )
    finally:
        app.close()
        print("ISAACLAB_ARTICULATION_PROBE:closed", flush=True)


if __name__ == "__main__":
    main()
