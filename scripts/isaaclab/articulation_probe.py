"""Instantiate the converted Microduck articulation in Isaac Sim 6.0.1.

The probe emits stage markers so a slow Kit/PhysX startup cannot be confused
with an actuator or asset failure.
"""

from __future__ import annotations

import argparse
import signal

from isaacsim import SimulationApp


def _timeout_handler(signum, frame):
    del signum, frame
    raise TimeoutError("simulation reset exceeded probe timeout")


def main() -> None:
    print("ISAACLAB_ARTICULATION_PROBE:app_imported", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-collisions", action="store_true")
    parser.add_argument("--reset-timeout", type=int, default=45)
    args = parser.parse_args()
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
        if args.no_collisions:
            from pxr import UsdPhysics

            removed = 0
            for prim in scene.stage.Traverse():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    removed += 1
            print(f"ISAACLAB_ARTICULATION_PROBE:collisions_removed={removed}", flush=True)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.reset_timeout)
        try:
            sim.reset()
        except TimeoutError as exc:
            print(f"ISAACLAB_ARTICULATION_PROBE:reset_timeout:{exc}", flush=True)
            return
        finally:
            signal.alarm(0)
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
