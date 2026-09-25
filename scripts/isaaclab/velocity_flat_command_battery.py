"""Evaluate a fixed checkpoint on a deterministic Velocity-Flat command battery.

The battery keeps each command fixed for the whole episode slice and records
tracking, reset, posture, and action statistics. It is an evaluation harness,
not a training script and does not change the registered task configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

from isaaclab_microduck.actuators.physx_friction_bridge import LaggedExternalEffort

try:
    from scripts.isaaclab.velocity_flat_battery_spec import (
        CASES,
        NUM_ENVS,
        SEED,
        STEPS_PER_CASE,
        evaluate_case,
    )
except ModuleNotFoundError:  # direct ``python scripts/isaaclab/...py`` entry
    from velocity_flat_battery_spec import CASES, NUM_ENVS, SEED, STEPS_PER_CASE, evaluate_case


DEFAULT_TASK = "IsaacLab-Velocity-Flat-MicroDuck"
# Backward-compatible module constant for callers that imported ``TASK``.
TASK = DEFAULT_TASK
# Keep the required scenarios visible at this executable boundary.  The case
# definitions remain centralized in ``velocity_flat_battery_spec`` so the
# MuJoCo and IsaacLab harnesses cannot silently drift apart.
REQUIRED_CASE_NAMES = ("zero", "forward", "lateral", "yaw")
COMMANDS = {case.name: case.command for case in CASES}
if not all(name in COMMANDS for name in REQUIRED_CASE_NAMES):
    raise RuntimeError(f"fixed command battery cases missing required scenarios: {tuple(COMMANDS)!r}")


def _tilt_rad(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Return the body-up tilt, excluding heading/yaw rotation.

    The previous ``2*acos(abs(qw))`` metric measured total orientation, so a
    pure yaw turn was incorrectly reported as a near-\u03c0 tilt.  Normalize the
    quaternion and rotate the body z-axis implicitly via its world z component;
    this is the roll/pitch angle used by the fallen termination.
    """

    quat = quat_xyzw / torch.linalg.vector_norm(quat_xyzw, dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(dim=-1)
    body_up_world_z = 1.0 - 2.0 * (x.square() + y.square())
    return torch.acos(torch.clamp(body_up_world_z, min=-1.0, max=1.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_command_term(term, value: torch.Tensor) -> None:
    """Hold a command term fixed for the whole deterministic battery case."""

    term.command[:] = value.reshape(1, -1)
    # CommandManager normally resamples on reset and whenever this timer
    # expires.  Infinity keeps the fixed battery value stable between the
    # explicit case writes below, including the pose slots.
    term.time_left[:] = float("inf")


def _set_command(base_env, value: tuple[float, float, float]) -> None:
    """Set the complete 13D command block used by the policy observation."""

    velocity = base_env.command_manager.get_term("base_velocity")
    _freeze_command_term(
        velocity,
        torch.as_tensor(value, device=velocity.command.device, dtype=velocity.command.dtype),
    )
    velocity.is_standing_env[:] = False
    # Velocity-Flat's common battery commands only the twist.  Pose commands
    # are nevertheless part of the actor ABI, so make their neutral values
    # explicit instead of inheriting random values from reset/resampling.
    for name, dim in (("head_pose", 4), ("body_pose", 6)):
        term = base_env.command_manager.get_term(name)
        _freeze_command_term(term, torch.zeros(dim, device=term.command.device, dtype=term.command.dtype))


def _refresh_fixed_command_observation(base_env):
    """Recompute obs after a reset so command slots match the forced values."""

    base_env.obs_buf = base_env.observation_manager.compute(update_history=False)


def _warp_to_torch(value):
    import warp as wp

    return wp.to_torch(value)


def _apply_lagged_friction(robot, actuator, bridge: LaggedExternalEffort) -> float:
    """Write the last post-step external-load sample for the next solve."""

    motor_effort = actuator.applied_effort
    external_effort = bridge.external_effort()
    velocity = robot.data.joint_vel.torch
    static, dynamic, viscous = actuator.physx_friction_coefficients(
        motor_effort, external_effort, velocity
    )
    robot.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=static,
        joint_dynamic_friction_coeff=dynamic,
        joint_viscous_friction_coeff=viscous,
    )
    return float(external_effort.abs().max().item())


def _observe_lagged_friction(robot, bridge: LaggedExternalEffort, dones: torch.Tensor) -> float:
    """Capture solved external load, excluding environments reset this step."""

    projected = _warp_to_torch(robot.root_view.get_dof_projected_joint_forces())
    actuation = _warp_to_torch(robot.root_view.get_dof_actuation_forces())
    done_ids = torch.nonzero(dones, as_tuple=False).flatten()
    if done_ids.numel():
        bridge.reset(done_ids)
    active_ids = torch.nonzero(~dones, as_tuple=False).flatten()
    if active_ids.numel():
        observed = bridge.observe(projected[active_ids], actuation[active_ids], active_ids)
        return float(observed.abs().max().item())
    return 0.0


def _run_case(
    env,
    policy,
    obs,
    name: str,
    value: tuple[float, float, float],
    steps: int,
    friction_bridge: LaggedExternalEffort | None = None,
) -> tuple[dict, object]:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    if friction_bridge is not None:
        friction_bridge.reset()
    _set_command(base_env, value)
    _refresh_fixed_command_observation(base_env)
    obs = env.get_observations()
    command_term = base_env.command_manager.get_term("base_velocity")

    errors_xy: list[torch.Tensor] = []
    errors_yaw: list[torch.Tensor] = []
    actual_xy_values: list[torch.Tensor] = []
    actual_yaw_values: list[torch.Tensor] = []
    heights: list[torch.Tensor] = []
    tilts: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    observed_external_effort_peak = 0.0
    applied_external_effort_peak = 0.0
    resets = 0
    finite = True
    for _ in range(steps):
        if friction_bridge is not None:
            applied_external_effort_peak = max(
                applied_external_effort_peak,
                _apply_lagged_friction(robot, next(iter(robot.actuators.values())), friction_bridge),
            )
        with torch.inference_mode():
            action = policy(obs)
            obs, _, dones, _ = env.step(action)
        if friction_bridge is not None:
            observed_external_effort_peak = max(
                observed_external_effort_peak,
                _observe_lagged_friction(robot, friction_bridge, dones),
            )
        _set_command(base_env, value)
        _refresh_fixed_command_observation(base_env)
        obs = env.get_observations()
        desired = command_term.command
        actual_xy = robot.data.root_lin_vel_b.torch[:, :2]
        actual_yaw = robot.data.root_ang_vel_b.torch[:, 2]
        root_pos = robot.data.root_link_pos_w.torch
        root_quat = robot.data.root_link_quat_w.torch
        tilt = _tilt_rad(root_quat)
        tensors = (actual_xy, actual_yaw, root_pos, root_quat, action)
        finite = finite and all(bool(torch.isfinite(t).all().item()) for t in tensors)
        errors_xy.append(torch.linalg.norm(desired[:, :2] - actual_xy, dim=-1).detach().cpu())
        errors_yaw.append(torch.abs(desired[:, 2] - actual_yaw).detach().cpu())
        actual_xy_values.append(actual_xy.detach().cpu())
        actual_yaw_values.append(actual_yaw.detach().cpu())
        heights.append(root_pos[:, 2].detach().cpu())
        tilts.append(tilt.detach().cpu())
        actions.append(action.detach().cpu())
        resets += int(dones.sum().item())

    xy = torch.cat(errors_xy)
    yaw = torch.cat(errors_yaw)
    actual_xy_all = torch.cat(actual_xy_values)
    actual_yaw_all = torch.cat(actual_yaw_values)
    height = torch.cat(heights)
    tilt = torch.cat(tilts)
    action_values = torch.cat(actions)
    result = {
        "name": name,
        "command": list(value),
        "steps": steps,
        "envs": base_env.scene.num_envs,
        "finite": finite,
        "mean_error_vel_xy_m_s": float(xy.mean()),
        "p95_error_vel_xy_m_s": float(torch.quantile(xy, 0.95)),
        "mean_error_vel_yaw_rad_s": float(yaw.mean()),
        "p95_error_vel_yaw_rad_s": float(torch.quantile(yaw, 0.95)),
        "mean_actual_vel_xy_m_s": [float(value) for value in actual_xy_all.mean(dim=0)],
        "mean_actual_vel_yaw_rad_s": float(actual_yaw_all.mean()),
        "episode_resets": resets,
        "reset_fraction_per_env_step": resets / (steps * base_env.scene.num_envs),
        "mean_root_height_m": float(height.mean()),
        "min_root_height_m": float(height.min()),
        "max_tilt_rad": float(tilt.max()),
        "mean_abs_action": float(action_values.abs().mean()),
        "p95_abs_action": float(torch.quantile(action_values.abs(), 0.95)),
        "friction_bridge": "one_step_lag_external_effort" if friction_bridge is not None else "production_motor_only",
        "observed_external_effort_peak_nm": observed_external_effort_peak,
        "applied_external_effort_peak_nm": applied_external_effort_peak,
    }
    case = next(case for case in CASES if case.name == name)
    result["passed"], result["failures"] = evaluate_case(
        case,
        finite=finite,
        mean_actual_xy=result["mean_actual_vel_xy_m_s"],
        mean_actual_yaw=result["mean_actual_vel_yaw_rad_s"],
        max_tilt=result["max_tilt_rad"],
        reset_fraction=result["reset_fraction_per_env_step"],
    )
    return result, obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--steps", type=int, default=STEPS_PER_CASE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--friction-mode",
        choices=("production", "one_step_lag"),
        default="production",
        help="Diagnostic only: apply the reset-safe one-step-lag PhysX friction bridge.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        print("ISAACLAB_VELOCITY_BATTERY:app_ready", flush=True)
        import gymnasium as gym

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg import (
            MicroduckVelocityFlatAdaptedPPORunnerCfg,
            MicroduckVelocityFlatCommandBucketsPPORunnerCfg,
            MicroduckVelocityFlatCommandBucketsSymmetryPPORunnerCfg,
            MicroduckVelocityFlatStrictificationPPORunnerCfg,
            MicroduckVelocityFlatPPORunnerCfg,
        )
        from isaaclab_microduck.tasks.velocity_flat import (
            make_velocity_flat_adapted_env_cfg,
            make_velocity_flat_command_buckets_env_cfg,
            make_velocity_flat_command_buckets_symmetry_env_cfg,
            make_velocity_flat_strictification_env_cfg,
            make_velocity_flat_env_cfg,
        )

        register_tasks()
        print("ISAACLAB_VELOCITY_BATTERY:tasks_registered", flush=True)
        profiles = {
            DEFAULT_TASK: (make_velocity_flat_env_cfg, MicroduckVelocityFlatPPORunnerCfg),
            "IsaacLab-Velocity-Flat-MicroDuck-Adapted": (
                make_velocity_flat_adapted_env_cfg,
                MicroduckVelocityFlatAdaptedPPORunnerCfg,
            ),
            "IsaacLab-Velocity-Flat-MicroDuck-CommandBuckets": (
                make_velocity_flat_command_buckets_env_cfg,
                MicroduckVelocityFlatCommandBucketsPPORunnerCfg,
            ),
            "IsaacLab-Velocity-Flat-MicroDuck-CommandBucketsSymmetry": (
                make_velocity_flat_command_buckets_symmetry_env_cfg,
                MicroduckVelocityFlatCommandBucketsSymmetryPPORunnerCfg,
            ),
            "IsaacLab-Velocity-Flat-MicroDuck-Strictification": (
                make_velocity_flat_strictification_env_cfg,
                MicroduckVelocityFlatStrictificationPPORunnerCfg,
            ),
        }
        try:
            make_env_cfg, runner_cfg_type = profiles[args.task]
        except KeyError as exc:
            raise ValueError(f"unknown IsaacLab battery task: {args.task!r}") from exc
        env_cfg = make_env_cfg(play=True, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env = gym.make(args.task, cfg=env_cfg)
        print("ISAACLAB_VELOCITY_BATTERY:env_made", flush=True)
        if args.task == "IsaacLab-Velocity-Flat-MicroDuck-Strictification":
            # A fresh evaluation env starts at curriculum step zero.  Replay
            # T22 checkpoints under the final strict live-manager profile so
            # the acceptance battery cannot accidentally use the bootstrap
            # root-height or reward configuration.
            from isaaclab_microduck.tasks.velocity_flat import _STRICTIFICATION_STAGES
            from isaaclab_microduck.tasks.velocity_flat_dr import curriculum_strictification

            base_env = env.unwrapped
            base_env.common_step_counter = 1000 * 24
            curriculum_strictification(base_env, None, _STRICTIFICATION_STAGES)
            print("ISAACLAB_VELOCITY_BATTERY:strict_profile_applied", flush=True)
        agent_cfg = runner_cfg_type()
        # Use the same action boundary as training and the production mjlab
        # recipe. In particular, do not silently make evaluation safer by
        # clipping raw policy output here.
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print("ISAACLAB_VELOCITY_BATTERY:wrapper_made", flush=True)
        # The official IsaacLab entrypoint migrates legacy model fields before
        # handing the config to rsl-rl. Keep this standalone harness on the
        # same compatibility path.
        handle_deprecated_rsl_rl_cfg(agent_cfg, "5.4.1")
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=vec_env.unwrapped.device)
        print("ISAACLAB_VELOCITY_BATTERY:runner_made", flush=True)
        runner.load(str(args.checkpoint))
        print("ISAACLAB_VELOCITY_BATTERY:checkpoint_loaded", flush=True)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        friction_bridge = None
        if args.friction_mode == "one_step_lag":
            robot = vec_env.unwrapped.scene["robot"]
            friction_bridge = LaggedExternalEffort(
                vec_env.unwrapped.scene.num_envs,
                robot.num_joints,
                device=vec_env.unwrapped.device,
            )
        cases = []
        obs = vec_env.get_observations()
        for index, (name, command) in enumerate(COMMANDS.items()):
            print(f"ISAACLAB_VELOCITY_BATTERY:case:{name}:start", flush=True)
            vec_env.seed(args.seed + index)
            obs, _ = vec_env.reset()
            result, obs = _run_case(
                vec_env,
                policy,
                obs,
                name,
                command,
                args.steps,
                friction_bridge,
            )
            cases.append(result)
            print(f"ISAACLAB_VELOCITY_BATTERY:case:{name}:done", flush=True)

        report = {
            "task": args.task,
            "backend": "isaaclab",
            "isaaclab_version": "3.0.0",
            "isaacsim_version": "6.0.1",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps_per_case": args.steps,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "actuator": "BamActuator",
            "clip_actions": agent_cfg.clip_actions,
            "friction_mode": args.friction_mode,
            "friction_bridge": (
                "one_step_lag_external_effort"
                if friction_bridge is not None
                else "production_motor_only_external_effort_unavailable"
            ),
            "passed": all(case["passed"] for case in cases),
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
