"""Probe PhysX joint-force tensors and their update timing on Microduck."""

from __future__ import annotations

import argparse
import json

import torch

from isaaclab.app import AppLauncher


def _torch(value):
    import warp as wp

    return wp.to_torch(value)


def _summary(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().float()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "max_abs": float(value.abs().max().item()),
        "first": value.reshape(-1)[:8].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils.configclass import configclass
        from isaaclab_microduck.assets.microduck import MICRODUCK_CFG

        sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim.reset()
        robot = scene.articulations["robot"]
        view = robot.root_view
        result: dict[str, object] = {
            "view_type": type(view).__name__,
            "joint_names": robot.joint_names,
            "phases": [],
        }

        def capture(phase: str) -> None:
            projected = _torch(view.get_dof_projected_joint_forces())
            incoming = _torch(view.get_link_incoming_joint_force())
            actuation = _torch(view.get_dof_actuation_forces())
            result["phases"].append(
                {
                    "phase": phase,
                    "projected_joint_forces": _summary(projected),
                    "incoming_joint_force": _summary(incoming),
                    "actuation_forces": _summary(actuation),
                }
            )

        capture("after_reset")
        target = robot.data.default_joint_pos.torch.clone()
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        capture("after_write_before_step")
        sim.step()
        capture("after_step_before_scene_update")
        scene.update(sim.cfg.dt)
        capture("after_scene_update")
        scene.write_data_to_sim()
        capture("second_after_write_before_step")
        sim.step()
        capture("second_after_step_before_scene_update")
        scene.update(sim.cfg.dt)
        capture("second_after_scene_update")

        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            from pathlib import Path

            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded)
        print(encoded, end="")
    finally:
        app.close()


if __name__ == "__main__":
    main()
