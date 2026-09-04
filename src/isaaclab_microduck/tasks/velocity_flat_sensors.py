"""Stateful actor sensor corruption for IsaacLab Velocity-Flat."""

from __future__ import annotations

import torch


def _ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = angle * 0.5
    q = torch.zeros((*angle.shape, 4), device=angle.device, dtype=angle.dtype)
    # IsaacLab math utilities use xyzw quaternion order.
    q[..., :3] = axis * torch.sin(half)[..., None]
    q[..., 3] = torch.cos(half)
    return q


def reset_imu_mounting(env, env_ids, max_angle_deg: float = 6.0) -> None:
    """Sample one constant random-axis mounting rotation per environment."""

    ids = _ids(env, env_ids)
    axis = torch.randn(ids.numel(), 3, device=env.device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = torch.empty(ids.numel(), device=env.device).uniform_(
        -torch.deg2rad(torch.tensor(max_angle_deg, device=env.device)),
        torch.deg2rad(torch.tensor(max_angle_deg, device=env.device)),
    )
    q = _axis_angle_quat(axis, angle)
    state = getattr(env, "_imu_mount_quat", None)
    if state is None or state.shape != (env.num_envs, 4):
        state = torch.zeros(env.num_envs, 4, device=env.device)
        state[:, 3] = 1.0
        env._imu_mount_quat = state
    state[ids] = q


def imu_mount_quat(env) -> torch.Tensor:
    q = getattr(env, "_imu_mount_quat", None)
    if q is None:
        q = torch.zeros(env.num_envs, 4, device=env.device)
        q[:, 3] = 1.0
        env._imu_mount_quat = q
    return q


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply xyzw IsaacLab quaternion(s) to vectors."""

    qv, qw = q[..., :3], q[..., 3:4]
    return v + 2.0 * torch.cross(qv, torch.cross(qv, v, dim=-1) + qw * v, dim=-1)


def misaligned_imu(value: torch.Tensor, env) -> torch.Tensor:
    """Rotate sensor vectors by the episode-stable mounting quaternion."""

    return quat_apply(imu_mount_quat(env), value)


def reset_actor_sensor_state(env, env_ids) -> None:
    """Reset history, action, and episode-stable encoder/IMU corruption."""

    ids = _ids(env, env_ids)
    for name in ("_previous_action", "_gyro_history", "_gravity_history", "_joint_vel_history"):
        value = getattr(env, name, None)
        if value is None:
            continue
        if value.ndim >= 2 and name.endswith("history"):
            value[:, ids] = 0.0
        else:
            value[ids] = 0.0
    bias = getattr(env, "_encoder_bias", None)
    if bias is not None:
        bias[ids] = torch.empty_like(bias[ids]).uniform_(-0.015, 0.015)
    reset_imu_mounting(env, ids)


__all__ = ["misaligned_imu", "quat_apply", "reset_actor_sensor_state", "reset_imu_mounting"]
