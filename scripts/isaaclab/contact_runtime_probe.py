"""Inspect Velocity-Flat articulation and contact paths in a live PhysX scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def _move_articulation_root_to_asset(env, env_ids) -> None:
    """Experimental root layout used to isolate PhysX nested-root warnings."""

    del env_ids
    from pxr import PhysxSchema, UsdPhysics

    stage = env.sim.stage
    robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    nested = stage.GetPrimAtPath("/World/envs/env_0/Robot/Geometry/trunk_base")
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    PhysxSchema.PhysxArticulationAPI.Apply(robot)
    nested.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    nested.RemoveAPI(PhysxSchema.PhysxArticulationAPI)


def _schema_paths(stage, schema_name: str) -> list[str]:
    return [str(prim.GetPath()) for prim in stage.Traverse() if schema_name in prim.GetAppliedSchemas()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-mode", choices=("nested", "asset"), default="nested")
    parser.add_argument("--candidate-ankle-view", action="store_true")
    parser.add_argument("--filtered-self-collision", action="store_true")
    parser.add_argument("--raw-self-contact", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--replicate-physics", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym

        from isaaclab.sim.utils.queries import resolve_matching_prims_from_source
        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.policy_abi import ACTION_SIZE
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        cfg = make_velocity_flat_env_cfg(num_envs=args.num_envs)
        if args.raw_self_contact:
            # The manager ContactSensor filtered view is intentionally removed
            # for this probe so the underlying PhysX view can be tested against
            # the exact nested rigid-body paths without its body-path rewrite.
            cfg.scene.self_collision_trunk = None
            cfg.scene.self_collision_legs = None
        if args.replicate_physics:
            cfg.scene.replicate_physics = True
        if args.root_mode == "asset":
            cfg.scene.robot.articulation_root_prim_path = ""
            cfg.events.activate_contact_reporters.func = _move_articulation_root_to_asset
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        stage = base_env.sim.stage
        robot = base_env.scene.articulations["robot"]
        sensor = base_env.scene.sensors["feet_ground_contact"]
        root_expr = cfg.scene.robot.prim_path + (cfg.scene.robot.articulation_root_prim_path or "")
        resolved = resolve_matching_prims_from_source(
            cfg.scene.robot.prim_path + "/Geometry/trunk_base",
            lambda prim: "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas(),
        )
        report = {
            "root_mode": args.root_mode,
            "articulation_root_expr": root_expr,
            "articulation_schema_paths": _schema_paths(stage, "PhysicsArticulationRootAPI"),
            "contact_reporter_paths": _schema_paths(stage, "PhysxContactReportAPI"),
            "resolved_body_paths": [
                {"source": str(prim.GetPath()), "destination": destination}
                for prim, destination in resolved
            ],
            "articulation_body_names": list(robot.body_names),
            "contact_body_names": list(sensor.body_names),
            "contact_view_paths": list(sensor.body_physx_view.prim_paths),
            "contact_sensor_count": int(sensor.num_sensors),
        }
        if args.filtered_self_collision:
            import warp as wp

            observations, _ = env.reset(seed=7)
            reward = None
            terminated = None
            truncated = None
            for _ in range(args.steps):
                observations, reward, terminated, truncated, _ = env.step(
                    torch.zeros((args.num_envs, ACTION_SIZE), device=base_env.device)
                )
            filtered = {}
            for name in ("self_collision_trunk", "self_collision_legs"):
                self_sensor = base_env.scene.sensors[name]
                matrix = self_sensor.data.force_matrix_w.torch
                filtered[name] = {
                    "body_names": list(self_sensor.body_names),
                    "matrix_shape": list(matrix.shape),
                    "finite": bool(torch.isfinite(matrix).all().item()),
                    "max_force": float(torch.linalg.norm(matrix, dim=-1).max().item()),
                }
                counts = wp.to_torch(self_sensor._contact_counts)
                filtered[name]["contact_counts_shape"] = list(counts.shape)
                filtered[name]["contact_counts"] = counts.cpu().tolist()
            foot_force = sensor.data.net_forces_w.torch
            report["filtered_self_collision"] = filtered
            report["feet_ground_max_force"] = float(torch.linalg.norm(foot_force, dim=-1).max().item())
            reward_terms = dict(base_env.reward_manager.get_active_iterable_terms(0))
            termination_terms = dict(base_env.termination_manager.get_active_iterable_terms(0))
            contact_reward_names = (
                "air_time",
                "foot_clearance",
                "foot_swing_height",
                "foot_slip",
                "self_collisions",
            )
            report["manager_contract"] = {
                "observation_shapes": {
                    name: list(value.shape) for name, value in observations.items()
                },
                "action_dim": int(base_env.action_manager.total_action_dim),
                "reward_finite": bool(torch.isfinite(reward).all().item()),
                "terminated_finite": bool(torch.isfinite(terminated).all().item()),
                "truncated_finite": bool(torch.isfinite(truncated).all().item()),
                "reward_terms": reward_terms,
                "termination_terms": termination_terms,
                "contact_rewards_finite": all(
                    name in reward_terms
                    and bool(torch.isfinite(torch.as_tensor(reward_terms[name])).all().item())
                    for name in contact_reward_names
                ),
            }
        if args.raw_self_contact:
            # ContactSensor stores the singleton PhysX tensor view after
            # initialization; SimulationContext does not expose it publicly.
            sim_view = base_env.scene.sensors["feet_ground_contact"]._physics_sim_view
            known_paths = list(base_env.scene.sensors["feet_ground_contact"].body_physx_view.prim_paths)
            raw_cases = {}
            for name, body_path, filters in (
                (
                    "trunk",
                    "/World/envs/env_*/Robot/Geometry/trunk_base",
                    [
                        "/World/envs/env_*/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg",
                        "/World/envs/env_*/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2",
                    ],
                ),
                (
                    "left_leg",
                    "/World/envs/env_*/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg",
                    [
                        "/World/envs/env_*/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2",
                    ],
                ),
            ):
                try:
                    # Use the tensor view's concrete path spelling when
                    # available; this avoids guessing whether a nested USD
                    # body pattern is parent-relative or leaf-relative.
                    body_candidates = [
                        p for p in known_paths
                        if p.endswith("/trunk_base") and name == "trunk"
                        or p.endswith("/leg") and name == "left_leg"
                    ]
                    # Passing one concrete body and one concrete partner per
                    # environment avoids the PhysX tensor plugin's ambiguity
                    # with nested USD paths under an env glob.
                    if body_candidates:
                        body_patterns = body_candidates
                        filter_patterns = [
                            [
                                f.replace("/World/envs/env_*/", f"/World/envs/{p.split('/')[3]}/")
                                for f in filters
                            ]
                            for p in body_patterns
                        ]
                    else:
                        body_patterns = [body_path]
                        filter_patterns = [filters]
                    view = sim_view.create_rigid_contact_view(
                        body_patterns,
                        filter_patterns=filter_patterns,
                        max_contact_data_count=64 * args.num_envs,
                    )
                    net = view.get_net_contact_forces(dt=float(base_env.sim.cfg.dt))
                    _, points, _, _, counts, starts = view.get_contact_data(dt=float(base_env.sim.cfg.dt))
                    raw_cases[name] = {
                        "sensor_count": int(view.sensor_count),
                        "filter_count": int(view.filter_count),
                        "net_shape": list(net.shape),
                        "point_shape": list(points.shape),
                        "count_shape": list(counts.shape),
                        "start_shape": list(starts.shape),
                        "counts": torch.as_tensor(counts).tolist(),
                        "finite": bool(torch.isfinite(torch.as_tensor(net)).all().item()),
                    }
                except Exception as exc:  # retain all view diagnostics in the report
                    raw_cases[name] = {"error": f"{type(exc).__name__}: {exc}"}
            report["raw_self_contact"] = raw_cases
        if args.candidate_ankle_view:
            ankle_glob = "/World/envs/env_*/Robot/Geometry/trunk_base/*/*/*/*/*"
            print(f"CONTACT_PROBE_CANDIDATE_START:{ankle_glob}", flush=True)
            rigid_view = sensor._physics_sim_view.create_rigid_body_view(ankle_glob)
            contact_view = sensor._physics_sim_view.create_rigid_contact_view(ankle_glob)
            report["candidate_ankle_view"] = {
                "glob": ankle_glob,
                "rigid_count": int(rigid_view.count),
                "rigid_paths": list(rigid_view.prim_paths),
                "contact_sensor_count": int(contact_view.sensor_count),
            }
            exact_views = []
            for side, path in (
                (
                    "left",
                    "/World/envs/env_*/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_left",
                ),
                (
                    "right",
                    "/World/envs/env_*/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_right",
                ),
            ):
                print(f"CONTACT_PROBE_EXACT_{side.upper()}_START:{path}", flush=True)
                rigid = sensor._physics_sim_view.create_rigid_body_view(path)
                contact = sensor._physics_sim_view.create_rigid_contact_view(path)
                exact_views.append(
                    {
                        "side": side,
                        "glob": path,
                        "rigid_count": int(rigid.count),
                        "rigid_paths": list(rigid.prim_paths),
                        "contact_sensor_count": int(contact.sensor_count),
                    }
                )
                print(f"CONTACT_PROBE_EXACT_{side.upper()}_END", flush=True)
            report["candidate_exact_views"] = exact_views
            print("CONTACT_PROBE_CANDIDATE_END", flush=True)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
