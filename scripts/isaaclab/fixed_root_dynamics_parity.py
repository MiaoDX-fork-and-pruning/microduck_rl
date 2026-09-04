"""Compare fixed-root BAM dynamics in MuJoCo and IsaacLab.

This is a bounded proof, not a training/evaluation harness.  Both backends
start from the same walk-MJCF HOME state, use the same fixed root pose,
``dt=0.005``, target sequences, voltage sag, and three-control-step target
delay.  MuJoCo is run twice: once with the BAM friction budget in its native
solver fields and once motor-only.  IsaacLab cannot observe the solved
external joint torque in its pre-step actuator callback, so its trajectory is
explicitly classified against both references rather than being presented as
exact solver parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from isaaclab_microduck.actuators.bam_math import (
    XL330_M6,
    friction_budget,
    voltage_torque,
)
from isaaclab_microduck.policy_abi import HOME_POSITION, POLICY_JOINT_ORDER


DT = 0.005
DELAY = 3
NOMINAL_VOLTAGE = 7.5
DROP_GAIN = 0.1
ROOT_Z = 0.5


def _joint_names_and_ids(model) -> tuple[list[str], list[int], list[int]]:
    import mujoco

    names: list[str] = []
    qpos_ids: list[int] = []
    dof_ids: list[int] = []
    for name in POLICY_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"MuJoCo model is missing policy joint {name!r}")
        names.append(name)
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))
    return names, qpos_ids, dof_ids


def _compile_mujoco():
    import mujoco

    root = Path(__file__).parents[2]
    xml = root / "src/mjlab_microduck/robot/microduck/robot_walk.xml"
    spec = mujoco.MjSpec.from_file(str(xml))
    force_limit = NOMINAL_VOLTAGE * XL330_M6.kt / XL330_M6.resistance
    policy_set = set(POLICY_JOINT_ORDER)
    for actuator in spec.actuators:
        target = actuator.target
        target_name = target.name if hasattr(target, "name") else str(target)
        if target_name not in policy_set:
            continue
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.forcerange = (-force_limit, force_limit)
        actuator.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for joint in spec.joints:
        if joint.name in policy_set:
            joint.armature = XL330_M6.kt * 0.0 + 0.0018077432831600838
            joint.damping = np.zeros((3, 1), dtype=np.float64)
            joint.frictionloss = 0.0
    model = spec.compile()
    model.opt.timestep = DT
    _, qpos_ids, dof_ids = _joint_names_and_ids(model)
    actuator_ids: list[int] = []
    for name in POLICY_JOINT_ORDER:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"MuJoCo model is missing actuator {name!r}")
        actuator_ids.append(int(aid))
    return model, qpos_ids, dof_ids, actuator_ids


def _targets(case: str, step: int) -> np.ndarray:
    target = np.asarray(HOME_POSITION, dtype=np.float64).copy()
    if case == "step":
        target[0] += 0.2
    elif case == "sine":
        target[0] += 0.15 * np.sin(2.0 * np.pi * 1.0 * step * DT)
    else:
        raise ValueError(f"unknown case {case!r}")
    return target


class _FixedDelay:
    def __init__(self, first: np.ndarray) -> None:
        self.history: deque[np.ndarray] = deque(
            (first.copy() for _ in range(DELAY + 1)), maxlen=DELAY + 1
        )

    def push(self, value: np.ndarray) -> np.ndarray:
        self.history.append(value.copy())
        return self.history[0].copy()


def _mujoco_case(
    model,
    qpos_ids: list[int],
    dof_ids: list[int],
    actuator_ids: list[int],
    case: str,
    *,
    with_friction: bool,
    steps: int,
) -> dict:
    import mujoco

    data = mujoco.MjData(model)
    root_qpos = np.asarray([0.0, 0.0, ROOT_Z, 1.0, 0.0, 0.0, 0.0])
    data.qpos[:] = 0.0
    data.qpos[:7] = root_qpos
    data.qpos[qpos_ids] = np.asarray(HOME_POSITION, dtype=np.float64)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    delay = _FixedDelay(_targets(case, 0))
    previous_motor = np.zeros(14, dtype=np.float64)
    rows: list[dict[str, float | list[float]]] = []
    for step in range(steps):
        # Re-apply the same root before every solve, matching the IsaacLab
        # fixed-root state protocol used by the companion path.
        data.qpos[:7] = root_qpos
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        raw_target = _targets(case, step)
        delayed_target = delay.push(raw_target)
        position = data.qpos[qpos_ids].copy()
        velocity = data.qvel[dof_ids].copy()
        supply = max(NOMINAL_VOLTAGE - DROP_GAIN * np.abs(previous_motor).sum(), 6.0)
        target_t = torch.as_tensor(delayed_target, dtype=torch.float64).reshape(1, -1)
        position_t = torch.as_tensor(position, dtype=torch.float64).reshape(1, -1)
        velocity_t = torch.as_tensor(velocity, dtype=torch.float64).reshape(1, -1)
        supply_t = torch.tensor([[supply]], dtype=torch.float64)
        _, motor_t = voltage_torque(
            target_t, position_t, velocity_t, params=XL330_M6, vin=supply_t
        )
        motor = motor_t[0].detach().cpu().numpy()
        external = (-data.qfrc_bias[dof_ids] + data.qfrc_constraint[dof_ids]).copy()
        if with_friction:
            budget = friction_budget(
                torch.as_tensor(motor).reshape(1, -1),
                torch.as_tensor(external).reshape(1, -1),
                torch.as_tensor(velocity).reshape(1, -1),
                params=XL330_M6,
            )[0].detach().cpu().numpy()
            model.dof_frictionloss[dof_ids] = budget
            model.dof_damping[dof_ids] = XL330_M6.friction_viscous
        else:
            model.dof_frictionloss[dof_ids] = 0.0
            model.dof_damping[dof_ids] = 0.0
        data.ctrl[:] = 0.0
        data.ctrl[actuator_ids] = motor
        mujoco.mj_step(model, data)
        rows.append(
            {
                "step": step,
                "raw_target_left_hip_yaw": float(raw_target[0]),
                "delayed_target_left_hip_yaw": float(delayed_target[0]),
                "supply_voltage": float(supply),
                "motor_effort_left_hip_yaw": float(motor[0]),
                "position_left_hip_yaw": float(data.qpos[qpos_ids[0]]),
                "velocity_left_hip_yaw": float(data.qvel[dof_ids[0]]),
                "position_max_abs": float(np.abs(data.qpos[qpos_ids]).max()),
                "velocity_max_abs": float(np.abs(data.qvel[dof_ids]).max()),
            }
        )
        previous_motor = motor
    return {
        "backend": "mjlab_mujoco_reference",
        "friction_mode": "bam_solver_fields" if with_friction else "motor_only",
        "rows": rows,
    }


def _summary_error(left: list[dict], right: list[dict], field: str) -> float:
    return max(abs(float(a[field]) - float(b[field])) for a, b in zip(left, right))


def _assemble_report(cases: dict[str, dict], isaac: dict[str, dict], steps: int) -> dict:
    report_cases: dict[str, dict] = {}
    for case, refs in cases.items():
        ib = isaac[case]
        bam_rows = refs["mujoco_bam"]["rows"]
        motor_rows = refs["mujoco_motor_only"]["rows"]
        report_cases[case] = {
            "isaaclab": ib,
            "mujoco_bam": refs["mujoco_bam"],
            "mujoco_motor_only": refs["mujoco_motor_only"],
            "target_max_abs_error": _summary_error(ib["rows"], bam_rows, "delayed_target_left_hip_yaw"),
            "effort_max_abs_error": _summary_error(ib["rows"], bam_rows, "motor_effort_left_hip_yaw"),
            "q_max_abs_error_vs_mujoco_bam": _summary_error(ib["rows"], bam_rows, "position_left_hip_yaw"),
            "qdot_max_abs_error_vs_mujoco_bam": _summary_error(ib["rows"], bam_rows, "velocity_left_hip_yaw"),
            "q_max_abs_error_vs_motor_only": _summary_error(ib["rows"], motor_rows, "position_left_hip_yaw"),
            "qdot_max_abs_error_vs_motor_only": _summary_error(ib["rows"], motor_rows, "velocity_left_hip_yaw"),
        }
    finite_values = [
        float(row[field])
        for case in report_cases.values()
        for row in case["isaaclab"]["rows"]
        for field in (
            "delayed_target_left_hip_yaw",
            "motor_effort_left_hip_yaw",
            "position_left_hip_yaw",
            "velocity_left_hip_yaw",
        )
    ]
    return {
        "proof": "fixed_root_same_state_bam_dynamics",
        "dt": DT,
        "root_z_m": ROOT_Z,
        "delay_steps": DELAY,
        "nominal_voltage": NOMINAL_VOLTAGE,
        "drop_gain": DROP_GAIN,
        "steps": steps,
        "cases": report_cases,
        "finite": bool(np.isfinite(finite_values).all()),
        "interpretation": {
            "motor_effort_and_delayed_target": "backend-neutral actuator/controller comparison",
            "mujoco_bam_vs_isaaclab": "solver plus PhysX same-step external-load friction timing delta",
            "mujoco_motor_only_vs_isaaclab": "isolates remaining solver/asset trajectory delta after removing BAM friction fields",
            "external_load": "IsaacLab pre-step callback cannot observe same-step solved external joint torque",
        },
    }


def _run_isaaclab(
    args, cases: dict[str, dict], steps: int, output_path: Path
) -> dict[str, dict]:
    # IsaacLab imports are intentionally delayed until after the MuJoCo traces,
    # keeping the reference path runnable in the repository's normal uv env.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import gymnasium as gym
        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg
        from isaaclab_microduck.assets.microduck import policy_target_to_sim
        from isaaclab_microduck.tasks.parity import ControlStepDelay

        register_tasks()
        cfg = make_velocity_flat_env_cfg(num_envs=1)
        # Remove task DR from this same-state comparison.  The production
        # task keeps these events stochastic; this proof isolates controller,
        # authored asset, and solver behavior with neutral one-point ranges.
        cfg.events.randomize_com.params["ranges"] = (0.0, 0.0)
        cfg.events.randomize_head_com.params["ranges"] = (0.0, 0.0)
        cfg.events.randomize_mass_inertia.params["alpha_range"] = (0.0, 0.0)
        cfg.events.randomize_armature.params["ranges"] = (1.0, 1.0)
        cfg.events.randomize_bam_friction.params["scale_range"] = (1.0, 1.0)
        cfg.events.randomize_foot_material.params["static_friction_range"] = (1.0, 1.0)
        cfg.events.randomize_foot_material.params["dynamic_friction_range"] = (1.0, 1.0)
        env = gym.make(
            "IsaacLab-Velocity-Flat-MicroDuck",
            cfg=cfg,
        )
        base_env = env.unwrapped
        scene = base_env.scene
        sim = base_env.sim
        env.reset(seed=2026)
        robot = scene["robot"]
        actuator = next(iter(robot.actuators.values()))
        # Freeze all stochastic actuator quantities and use the same lag as the
        # MuJoCo reference.  This is a proof configuration, not the training cfg.
        actuator._delay = ControlStepDelay(
            1,
            robot.num_joints,
            min_lag=DELAY,
            max_lag=DELAY,
            device=robot.device,
            sample_lag_each_push=True,
        )
        actuator._delay_initialized.zero_()
        actuator._supply_voltage.fill_(NOMINAL_VOLTAGE)
        actuator._vin_drop_gain.fill_(DROP_GAIN)
        actuator._friction_scale.fill_(1.0)
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, 2] = ROOT_Z
        root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
        policy_home = torch.as_tensor(HOME_POSITION, device=robot.device).reshape(1, -1)
        home = policy_target_to_sim(policy_home, robot.joint_names)
        output: dict[str, dict] = {}
        for case in cases:
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
            robot.write_joint_position_to_sim_index(position=home)
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(robot.data.default_joint_vel.torch)
            )
            actuator.reset()
            actuator._delay_initialized.zero_()
            actuator._delay.reset(
                torch.zeros(1, dtype=torch.long, device=robot.device), home
            )
            rows: list[dict[str, float | list[float]]] = []
            for step in range(steps):
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                robot.write_root_velocity_to_sim_index(root_velocity=root_velocity)
                policy_target = torch.as_tensor(
                    _targets(case, step), dtype=torch.float32, device=robot.device
                ).reshape(1, -1)
                sim_target = policy_target_to_sim(policy_target, robot.joint_names)
                robot.set_joint_position_target(sim_target)
                scene.write_data_to_sim()
                sim.step()
                scene.update(DT)
                position = robot.data.joint_pos.torch[0]
                velocity = robot.data.joint_vel.torch[0]
                left_id = robot.joint_names.index("left_hip_yaw")
                rows.append(
                    {
                        "step": step,
                        "raw_target_left_hip_yaw": float(_targets(case, step)[0]),
                        "delayed_target_left_hip_yaw": float(actuator._delayed_target[0, left_id]),
                        "supply_voltage": float(
                            (NOMINAL_VOLTAGE - DROP_GAIN * actuator._previous_motor_effort.abs().sum()).clamp_min(6.0)
                        ),
                        "motor_effort_left_hip_yaw": float(actuator._previous_motor_effort[0, left_id]),
                        "position_left_hip_yaw": float(position[left_id]),
                        "velocity_left_hip_yaw": float(velocity[left_id]),
                        "position_max_abs": float(position.abs().max()),
                        "velocity_max_abs": float(velocity.abs().max()),
                    }
                )
            output[case] = {"backend": "isaaclab_physx", "friction_mode": "motor_only", "rows": rows}
        # Persist the raw IsaacLab side before teardown.  Isaac Sim's Kit
        # shutdown can terminate the interpreter from ``env.close()`` on
        # some container builds, so the bounded proof must not depend on the
        # caller reaching its post-return serializer.
        output_path.with_name(output_path.stem + ".isaac.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n"
        )
        output_path.write_text(
            json.dumps(_assemble_report(cases, output, steps), indent=2, sort_keys=True)
            + "\n"
        )
        return output
    finally:
        if "env" in locals():
            env.close()
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="write the MuJoCo reference without importing IsaacLab",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="existing MuJoCo reference JSON for the IsaacLab half",
    )
    # Isaac Sim's AppLauncher owns the remaining runtime flags.  Keep the
    # reference-only path usable in the normal mjlab uv environment, where
    # IsaacLab is intentionally absent.
    if "--reference-only" not in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < DELAY + 2:
        raise ValueError("--steps must exceed the fixed delay")
    if args.reference_only:
        model, qpos_ids, dof_ids, actuator_ids = _compile_mujoco()
        cases: dict[str, dict] = {}
        for case in ("step", "sine"):
            mujoco_bam = _mujoco_case(
                model, qpos_ids, dof_ids, actuator_ids, case, with_friction=True, steps=args.steps
            )
            mujoco_motor = _mujoco_case(
                model, qpos_ids, dof_ids, actuator_ids, case, with_friction=False, steps=args.steps
            )
            cases[case] = {"mujoco_bam": mujoco_bam, "mujoco_motor_only": mujoco_motor}
    else:
        if args.reference is None:
            raise ValueError("IsaacLab mode requires --reference from --reference-only")
        cases = json.loads(args.reference.read_text())["cases"]
    if args.reference_only:
        report = {
            "proof": "fixed_root_same_state_bam_dynamics_reference",
            "dt": DT,
            "root_z_m": ROOT_Z,
            "delay_steps": DELAY,
            "nominal_voltage": NOMINAL_VOLTAGE,
            "drop_gain": DROP_GAIN,
            "steps": args.steps,
            "cases": cases,
            "reference_only": True,
            "finite": True,
            "interpretation": {
                "mujoco_bam": "BAM friction budget written into native solver fields",
                "mujoco_motor_only": "same state/targets with friction fields disabled",
            },
        }
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(encoded, end="")
        return
    isaac = _run_isaaclab(args, cases, args.steps, args.output)
    report = _assemble_report(cases, isaac, args.steps)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
