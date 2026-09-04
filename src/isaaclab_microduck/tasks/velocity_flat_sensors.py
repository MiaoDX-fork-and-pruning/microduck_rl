"""Stateful actor sensor corruption for IsaacLab Velocity-Flat."""

from __future__ import annotations

import torch

from isaaclab_microduck.tasks.parity import observation_noise


def _writable_clone(value: torch.Tensor) -> torch.Tensor:
    """Copy a tensor out of inference mode into mutable episode state."""

    with torch.inference_mode(False):
        out = torch.empty_like(value)
        out.copy_(value.detach())
    return out


def _writable_zeros(shape: tuple[int, ...], *, device, dtype=torch.float32) -> torch.Tensor:
    with torch.inference_mode(False):
        return torch.zeros(shape, device=device, dtype=dtype)


def _ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = angle * 0.5
    q = _writable_zeros((*angle.shape, 4), device=angle.device, dtype=angle.dtype)
    # IsaacLab math utilities use xyzw quaternion order.
    q[..., :3] = axis * torch.sin(half)[..., None]
    q[..., 3] = torch.cos(half)
    return q


def reset_imu_mounting(env, env_ids, max_angle_deg: float = 6.0) -> None:
    """Initialize one constant random-axis mounting rotation per environment.

    The mounting error models a calibration/assembly property of a robot.  It
    therefore survives episode resets; only environments that have not been
    initialized yet are sampled.  This also makes subset resets independent of
    the other environments in the batch.
    """

    ids = _ids(env, env_ids)
    initialized = getattr(env, "_imu_mount_initialized", None)
    if initialized is None or initialized.shape != (env.num_envs,):
        initialized = _writable_zeros((env.num_envs,), device=env.device, dtype=torch.bool)
        env._imu_mount_initialized = initialized
    elif initialized.is_inference():
        initialized = _writable_clone(initialized)
        env._imu_mount_initialized = initialized
    ids = ids[~initialized[ids]]
    if ids.numel() == 0:
        return

    axis = torch.randn(ids.numel(), 3, device=env.device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = torch.empty(ids.numel(), device=env.device).uniform_(
        0.0, torch.deg2rad(torch.tensor(max_angle_deg, device=env.device))
    )
    q = _axis_angle_quat(axis, angle)
    state = getattr(env, "_imu_mount_quat", None)
    if state is None or state.shape != (env.num_envs, 4):
        state = _writable_zeros((env.num_envs, 4), device=env.device)
        state[:, 3] = 1.0
        env._imu_mount_quat = state
    elif state.is_inference():
        state = _writable_clone(state)
        env._imu_mount_quat = state
    state[ids] = q
    initialized[ids] = True


def imu_mount_quat(env) -> torch.Tensor:
    q = getattr(env, "_imu_mount_quat", None)
    if q is None:
        # Lazy initialization covers callers that evaluate observations before
        # the reset event.  Once sampled, the mounting error is never changed
        # by episode resets.
        reset_imu_mounting(env, None)
        q = env._imu_mount_quat
    elif q.is_inference():
        q = _writable_clone(q)
        env._imu_mount_quat = q
    return q


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply xyzw IsaacLab quaternion(s) to vectors."""

    qv, qw = q[..., :3], q[..., 3:4]
    return v + 2.0 * torch.cross(qv, torch.cross(qv, v, dim=-1) + qw * v, dim=-1)


def misaligned_imu(value: torch.Tensor, env) -> torch.Tensor:
    """Rotate sensor vectors by the episode-stable mounting quaternion."""

    return quat_apply(imu_mount_quat(env), value)


def sensor_corruption(
    env,
    name: str,
    value: torch.Tensor,
    *,
    noise: float,
    delay: int,
    delay_update_period: int = 0,
) -> torch.Tensor:
    """Apply bounded noise and control-step delayed sensor samples."""

    if delay < 0 or delay_update_period < 0:
        raise ValueError("delay and delay_update_period must be non-negative")
    if delay == 0:
        return observation_noise(value, noise)

    key = f"_{name}_history"
    history = getattr(env, key, None)
    if history is None or history.shape != (delay + 1, *value.shape):
        with torch.inference_mode(False):
            history = torch.empty((delay + 1, *value.shape), device=value.device, dtype=value.dtype)
            history.copy_(value.unsqueeze(0))
        setattr(env, key, history)
    elif history.is_inference():
        history = _writable_clone(history)
        setattr(env, key, history)
    with torch.inference_mode(False):
        history[:-1].copy_(history[1:])
        history[-1].copy_(value)

    if delay_update_period == 0:
        lag = torch.full((value.shape[0],), delay, device=value.device, dtype=torch.long)
    else:
        lag_key = f"_{name}_lag"
        step_key = f"_{name}_delay_step"
        phase_key = f"_{name}_delay_phase"
        shape = (value.shape[0],)
        lag = getattr(env, lag_key, None)
        step = getattr(env, step_key, None)
        phase = getattr(env, phase_key, None)
        if lag is None or lag.shape != shape:
            with torch.inference_mode(False):
                lag = torch.zeros(shape, device=value.device, dtype=torch.long)
                step = torch.zeros(shape, device=value.device, dtype=torch.long)
                phase = torch.randint(0, delay_update_period, shape, device=value.device)
            setattr(env, lag_key, lag)
            setattr(env, step_key, step)
            setattr(env, phase_key, phase)
        elif lag.is_inference() or step.is_inference() or phase.is_inference():
            lag = _writable_clone(lag)
            step = _writable_clone(step)
            phase = _writable_clone(phase)
            setattr(env, lag_key, lag)
            setattr(env, step_key, step)
            setattr(env, phase_key, phase)
        due = ((step + phase) % delay_update_period) == 0
        if due.any():
            sampled = torch.randint(0, delay + 1, shape, device=value.device)
            lag[due] = sampled[due]
        step.add_(1)

    out = history[delay - lag, torch.arange(value.shape[0], device=value.device)]
    return observation_noise(out, noise)


def reset_actor_sensor_state(env, env_ids) -> None:
    """Reset history, action, and episode-stable encoder/IMU corruption."""

    ids = _ids(env, env_ids)
    for name in ("_previous_action", "_gyro_history", "_gravity_history", "_joint_vel_history"):
        value = getattr(env, name, None)
        if value is None:
            continue
        if value.is_inference():
            value = _writable_clone(value)
            setattr(env, name, value)
        if value.ndim >= 2 and name.endswith("history"):
            value[:, ids] = 0.0
        else:
            value[ids] = 0.0
    # DelayBuffer semantics reset the sample clock for only the reset subset.
    # The next observation starts with lag zero until its periodic sampler is
    # due again; non-reset environments retain their phase and lag.
    for name in ("gyro", "gravity"):
        for suffix in ("_delay_step", "_lag"):
            value = getattr(env, f"_{name}{suffix}", None)
            if value is not None:
                if value.is_inference():
                    value = _writable_clone(value)
                    setattr(env, f"_{name}{suffix}", value)
                value[ids] = 0
    for name in ("gyro", "gravity"):
        value = getattr(env, f"_{name}_delay_phase", None)
        if value is not None:
            if value.is_inference():
                value = _writable_clone(value)
                setattr(env, f"_{name}_delay_phase", value)
            period = 64
            value[ids] = torch.randint(0, period, (ids.numel(),), device=env.device)
    bias = getattr(env, "_encoder_bias", None)
    if bias is not None:
        if bias.is_inference():
            bias = _writable_clone(bias)
            env._encoder_bias = bias
        bias[ids] = torch.empty_like(bias[ids]).uniform_(-0.015, 0.015)
    # Mounting misalignment is startup/per-environment state, not per-episode
    # noise.  This call only initializes previously unseen subset entries.
    reset_imu_mounting(env, ids)


__all__ = ["misaligned_imu", "quat_apply", "reset_actor_sensor_state", "reset_imu_mounting", "sensor_corruption"]
