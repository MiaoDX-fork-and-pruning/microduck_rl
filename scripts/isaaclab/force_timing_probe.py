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
            "target_step": {"joint_index": 0, "target_rad": 0.35},
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
        # A nonzero target is required here.  HOME produces near-zero forces,
        # which can make a stale pre-step getter look equivalent to a fresh
        # post-step getter even though the timing contract is different.
        target = robot.data.default_joint_pos.torch.clone()
        target[:, 0] += 0.35
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

        phases = result["phases"]
        assert isinstance(phases, list)

        def max_delta(left: int, right: int, key: str) -> float:
            first = phases[left][key]
            second = phases[right][key]
            assert isinstance(first, dict) and isinstance(second, dict)
            first_values = torch.as_tensor(first["first"], dtype=torch.float64)
            second_values = torch.as_tensor(second["first"], dtype=torch.float64)
            return float((first_values - second_values).abs().max().item())

        result["timing_summary"] = {
            "projected_force_delta_write_to_post_step": max_delta(
                1, 2, "projected_joint_forces"
            ),
            "incoming_force_delta_write_to_post_step": max_delta(
                1, 2, "incoming_joint_force"
            ),
            "actuation_delta_write_to_post_step": max_delta(
                1, 2, "actuation_forces"
            ),
            "projected_force_delta_post_step_to_scene_update": max_delta(
                2, 3, "projected_joint_forces"
            ),
            "incoming_force_delta_post_step_to_scene_update": max_delta(
                2, 3, "incoming_joint_force"
            ),
            "actuation_delta_post_step_to_scene_update": max_delta(
                2, 3, "actuation_forces"
            ),
            "interpretation": (
                "force getters are sampled from the previous solved state before "
                "sim.step; the actuator callback cannot use same-step external load"
            ),
        }

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
