"""Inspect Velocity-Flat articulation and contact paths in a live PhysX scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

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
    parser.add_argument(
        "--forced-self-contact",
        action="store_true",
        help="compare concrete raw contact counts before and after a deterministic MJCF-valid pose",
    )
    parser.add_argument(
        "--forced-self-contact-direct",
        action="store_true",
        help="write the deterministic MJCF-valid pose directly before stepping PhysX",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--replicate-physics", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.forced_self_contact_direct:
        args.raw_self_contact = True
        args.forced_self_contact = True
    launcher = AppLauncher(args)
    app = launcher.app
    print("CONTACT_PROBE:app_ready", flush=True)
    env = None
    try:
        print("CONTACT_PROBE:imports_start", flush=True)
        import gymnasium as gym
        print("CONTACT_PROBE:import_gym_done", flush=True)

        from isaaclab.sim.utils.queries import resolve_matching_prims_from_source
        print("CONTACT_PROBE:import_queries_done", flush=True)
        from isaaclab_microduck.tasks import register_tasks
        print("CONTACT_PROBE:import_tasks_done", flush=True)
        from isaaclab_microduck.policy_abi import (
            ACTION_SIZE,
            HOME_POSITION,
            POLICY_JOINT_ORDER,
            reorder_policy_joints,
        )
        print("CONTACT_PROBE:import_policy_done", flush=True)
        from isaaclab_microduck.tasks.velocity_flat_contact import self_collision_cost
        print("CONTACT_PROBE:import_contact_done", flush=True)
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg
        print("CONTACT_PROBE:import_cfg_done", flush=True)

        print("CONTACT_PROBE:imports_done", flush=True)
        register_tasks()
        print("CONTACT_PROBE:tasks_registered", flush=True)
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
        print("CONTACT_PROBE:env_made", flush=True)
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
            raw_views = {}
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
                    raw_views[name] = view
                except Exception as exc:  # retain all view diagnostics in the report
                    raw_cases[name] = {"error": f"{type(exc).__name__}: {exc}"}
            report["raw_self_contact"] = raw_cases
            if args.forced_self_contact or args.forced_self_contact_direct:
                if set(raw_views) != {"trunk", "left_leg"}:
                    raise RuntimeError(
                        "forced self-contact fixture requires both concrete trunk and left-leg views"
                    )

                def _counts(view):
                    values = view.get_contact_data(dt=float(base_env.sim.cfg.dt))[4]
                    return torch.as_tensor(values, device=base_env.device).clone()

                # This pose was found by evaluating robot_walk.xml with
                # MuJoCo's collision detector.  It produces a trunk/left-leg
                # overlap while keeping all 14 joints inside their authored
                # limits, making the PhysX check a geometry-source comparison.
                forced_action = torch.tensor(
                    [
                        -0.29586331, -0.34041397, -0.02215451, -0.46743155, -0.33582527,
                        0.04877433, 0.03852781, 0.14460955, -0.26784449,
                        0.32895650, 0.51486951, -0.38104408, 0.50578374, -0.34331586,
                    ],
                    dtype=robot.data.default_joint_pos.torch.dtype,
                    device=base_env.device,
                ).expand(args.num_envs, -1).clone()
                forced_policy_q = forced_action + torch.as_tensor(
                    HOME_POSITION, dtype=forced_action.dtype, device=base_env.device
                ).reshape(1, -1)
                print("CONTACT_PROBE_FORCED:pose_ready", flush=True)
                home_q = robot.data.default_joint_pos.torch.clone()
                zero_dq = torch.zeros_like(home_q)
                root_pose = robot.data.default_root_pose.torch.clone()
                root_vel = robot.data.default_root_vel.torch.clone()

                # Initialize the manager and PhysX state through the same
                # reset path used by the production task before issuing raw
                # articulation writes.  Calling SimulationContext.step on an
                # unreset ManagerBasedEnv can terminate Kit without a Python
                # exception in Isaac Sim 6.0.1.
                print("CONTACT_PROBE_FORCED:reset", flush=True)
                env.reset(seed=7)
                print("CONTACT_PROBE_FORCED:reset_done", flush=True)
                # Reuse the same concrete views as the production reward term.
                # A second live set can destabilize the PhysX tensor plugin.
                self_collision_cost(base_env, ("missing",))
                production_views = getattr(base_env, "_velocity_flat_raw_self_contact_views", None)
                if production_views is not None:
                    raw_views = {"trunk": production_views[0], "left_leg": production_views[1]}
                # ``env.reset`` already writes the clean HOME state through
                # the manager and gives us a synchronized zero-contact
                # baseline.  Avoid a second raw write here: Isaac Sim 6.0.1
                # can terminate Kit when a ManagerBasedEnv has pending
                # actuator data and ``scene.write_data_to_sim`` is called
                # immediately after an external joint-state write.
                baseline_total = float(
                    getattr(base_env, "_velocity_flat_last_raw_self_contact_counts", torch.zeros(args.num_envs, device=base_env.device)).sum().item()
                )

                counts_history: list[float] = []
                if args.forced_self_contact_direct:
                    # This isolates collision/reporting semantics from BAM
                    # convergence.  The state write uses the same simulator
                    # joint ordering as the articulation, then one complete
                    # PhysX step synchronizes raw contact views.
                    print("CONTACT_PROBE_FORCED:direct_write", flush=True)
                    sim_q = forced_policy_q[..., torch.as_tensor(
                        reorder_policy_joints(torch.arange(ACTION_SIZE), robot.joint_names),
                        device=base_env.device,
                        dtype=torch.long,
                    )]
                    robot.write_joint_position_to_sim_index(position=sim_q)
                    robot.write_joint_velocity_to_sim_index(velocity=zero_dq)
                    written_q = robot.data.joint_pos.torch.detach().clone()
                    written_target_error = float(torch.abs(written_q - sim_q).max().item())
                    # The articulation write reaches PhysX directly.  Do not
                    # flush manager actuator buffers here: Isaac Sim 6.0.1
                    # can terminate when that flush follows an external state
                    # write in the same frame.
                    base_env.sim.step()
                    base_env.scene.update(base_env.sim.cfg.dt)
                    counts_now = _counts(raw_views["trunk"]).sum() + _counts(raw_views["left_leg"]).sum()
                    counts_history.append(float(counts_now.item()))
                    forced_total = float(counts_now.item())
                else:
                    # Drive toward the valid collision pose through the complete
                    # manager/actuator path.  The action was found in MuJoCo as
                    # a HOME-relative pose and is inside the policy clip range.
                    print("CONTACT_PROBE_FORCED:target_drive", flush=True)
                    for _ in range(300):
                        env.step(forced_action)
                        counts_history.append(
                            float(
                                getattr(
                                    base_env,
                                    "_velocity_flat_last_raw_self_contact_counts",
                                    torch.zeros(args.num_envs, device=base_env.device),
                                ).sum().item()
                            )
                        )
                    print("CONTACT_PROBE_FORCED:target_drive_done", flush=True)
                    forced_total = float(
                        getattr(base_env, "_velocity_flat_last_raw_self_contact_counts", torch.zeros(args.num_envs, device=base_env.device)).sum().item()
                    )
                actual_q = robot.data.joint_pos.torch.detach().clone()
                target_sim_q = forced_policy_q[..., torch.as_tensor(
                    reorder_policy_joints(torch.arange(ACTION_SIZE), robot.joint_names),
                    device=base_env.device,
                    dtype=torch.long,
                )]
                report["forced_self_contact"] = {
                    "pose_source": "robot_walk.xml_mujoco_collision_search_seed_2026",
                    "policy_joint_order": list(POLICY_JOINT_ORDER),
                    "policy_joint_position": forced_policy_q[0].cpu().tolist(),
                    "sim_joint_order": list(robot.joint_names),
                    "actual_sim_joint_position": actual_q[0].cpu().tolist(),
                    "written_target_error_max": written_target_error if args.forced_self_contact_direct else None,
                    "target_error_max": float(torch.abs(actual_q - target_sim_q).max().item()),
                    "baseline_total": int(baseline_total),
                    "forced_total": int(forced_total),
                    "peak_transition_total": int(max(counts_history, default=0.0)),
                    "counts_history_total": counts_history,
                    "transition_nonzero": bool(baseline_total == 0 and forced_total > 0),
                    "finite": bool(torch.isfinite(actual_q).all().item()),
                }
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
    except BaseException:
        print("CONTACT_PROBE:failure", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
