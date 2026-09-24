"""Deterministic state sequence shared by the MJLab/IsaacLab source probes.

The sequence is kinematic rather than policy-driven: both simulators receive
the same root pose/velocity and joint pose/velocity at each sample. This keeps
the comparison about measurement sources and frame conventions instead of
contact solver trajectories or observation normalization.
"""

from __future__ import annotations

import math

import numpy as np

from isaaclab_microduck.policy_abi import HOME_POSITION, POLICY_JOINT_ORDER


TRACE_SCHEMA_VERSION = 1
CONTROL_DT_S = 0.02


def _quat_xyzw(roll: float, pitch: float, yaw: float) -> list[float]:
    """Return an XYZ intrinsic Euler quaternion in IsaacLab ``xyzw`` order."""

    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def scripted_states(num_steps: int = 12) -> list[dict[str, object]]:
    """Build a seeded, bounded trajectory of raw state inputs.

    Positions are kept well above the plane so contact impulses cannot alter
    the source comparison. Velocities are written explicitly at every sample.
    """

    if num_steps < 2:
        raise ValueError("sensor trace needs at least two samples")
    home = np.asarray(HOME_POSITION, dtype=np.float64)
    states: list[dict[str, object]] = []
    amplitudes = np.asarray(
        [0.035, 0.018, 0.045, 0.055, 0.030, 0.025, 0.040, 0.030, 0.020, 0.035, 0.018, 0.045, 0.055, 0.030],
        dtype=np.float64,
    )
    frequencies = np.asarray(
        [1.0, 1.3, 0.8, 1.1, 1.5, 0.7, 1.2, 1.4, 0.9, 1.0, 1.3, 0.8, 1.1, 1.5],
        dtype=np.float64,
    )
    for step in range(num_steps):
        t = step * CONTROL_DT_S
        roll = 0.08 * math.sin(0.7 * t)
        pitch = 0.06 * math.cos(0.55 * t)
        yaw = 0.20 * math.sin(0.45 * t)
        joint_phase = frequencies * t + 0.17
        joint_pos = home + amplitudes * np.sin(joint_phase)
        joint_vel = amplitudes * frequencies * np.cos(joint_phase)
        states.append(
            {
                "step": step,
                "time_s": t,
                "root_pose_xyzw": [
                    0.04 * math.sin(0.5 * t),
                    0.03 * math.cos(0.4 * t),
                    0.50 + 0.01 * math.sin(0.3 * t),
                    *_quat_xyzw(roll, pitch, yaw),
                ],
                "root_velocity_world": [
                    0.02 * math.cos(0.5 * t),
                    -0.012 * math.sin(0.4 * t),
                    0.003 * math.cos(0.3 * t),
                    0.14 * math.cos(0.7 * t),
                    -0.11 * math.sin(0.55 * t),
                    0.23 * math.cos(0.45 * t),
                ],
                "joint_pos": joint_pos.tolist(),
                "joint_vel": joint_vel.tolist(),
            }
        )
    return states


def mujoco_pose_wxyz(root_pose_xyzw: list[float]) -> list[float]:
    """Convert an IsaacLab ``xyzw`` pose to MuJoCo's ``wxyz`` qpos order."""

    x, y, z, qx, qy, qz, qw = root_pose_xyzw
    return [x, y, z, qw, qx, qy, qz]


__all__ = [
    "CONTROL_DT_S",
    "POLICY_JOINT_ORDER",
    "TRACE_SCHEMA_VERSION",
    "mujoco_pose_wxyz",
    "scripted_states",
]
