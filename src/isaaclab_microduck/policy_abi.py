"""Simulator-neutral Microduck policy ABI specification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

OBSERVATION_SIZE = 61
ACTION_SIZE = 14
COMMAND_SIZE = 13
ACTION_SCALE = 1.0

POLICY_JOINT_ORDER = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

# Servo order: left leg, neck/head, right leg.
HOME_POSITION = np.asarray(
    [0.0, -0.0873, -0.4579, -0.0049, 0.4530,
     0.3491, 0.3491, 0.0, 0.0,
     0.0, 0.0873, 0.4579, 0.0049, -0.4530],
    dtype=np.float32,
)


def build_observation(
    gyro: Sequence[float],
    projected_gravity: Sequence[float],
    joint_position_relative_home: Sequence[float],
    joint_velocity: Sequence[float],
    previous_raw_action: Sequence[float],
    command: Sequence[float],
) -> np.ndarray:
    """Concatenate the hardware-facing actor terms in ABI order."""

    fields = tuple(np.asarray(field, dtype=np.float32).reshape(-1) for field in (
        gyro,
        projected_gravity,
        joint_position_relative_home,
        joint_velocity,
        previous_raw_action,
        command,
    ))
    expected = (3, 3, 14, 14, 14, COMMAND_SIZE)
    if tuple(field.size for field in fields) != expected:
        raise ValueError(f"invalid ABI field sizes: {[field.size for field in fields]}")
    return np.concatenate(fields)


def action_to_target(raw_action: Sequence[float], scale: float = ACTION_SCALE) -> np.ndarray:
    """Map raw policy output to servo target positions around HOME."""

    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.size != ACTION_SIZE:
        raise ValueError(f"expected {ACTION_SIZE} action values, got {action.size}")
    return HOME_POSITION + np.float32(scale) * action


def reorder_policy_joints(values: Sequence[float] | np.ndarray, target_order: Sequence[str]) -> np.ndarray:
    """Reorder canonical policy joints into a simulator's named traversal order."""

    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] != ACTION_SIZE:
        raise ValueError(f"expected {ACTION_SIZE} policy joints, got {array.shape[-1]}")
    if set(target_order) != set(POLICY_JOINT_ORDER):
        raise ValueError(f"unexpected simulator joint names: {list(target_order)}")
    index = {name: i for i, name in enumerate(POLICY_JOINT_ORDER)}
    return array[..., [index[name] for name in target_order]]
