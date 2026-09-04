"""Backend-neutral Velocity-Flat semantic primitives.

These functions deliberately have no IsaacLab dependency.  They are the
executable reference for the pieces that must agree with mjlab before a
PhysX training run is allowed.  Framework adapters call them while CPU tests
exercise them directly with seeded tensors.
"""

from __future__ import annotations

from collections import deque

import torch


def clip_policy_action(action: torch.Tensor, limit: float = 1.0) -> torch.Tensor:
    """Apply the policy-space clip used by the mjlab RSL-RL wrapper."""

    if limit <= 0:
        raise ValueError("action clip limit must be positive")
    return torch.clamp(action, -float(limit), float(limit))


def policy_action_to_target(
    action: torch.Tensor,
    home: torch.Tensor,
    *,
    scale: float = 1.0,
    clip: float = 1.0,
) -> torch.Tensor:
    """Map clipped raw actions to absolute servo targets around HOME."""

    if action.shape[-1] != home.shape[-1]:
        raise ValueError(f"action/home dimension mismatch: {action.shape[-1]} != {home.shape[-1]}")
    return home + float(scale) * clip_policy_action(action, clip)


def effective_supply_voltage(
    nominal_voltage: torch.Tensor,
    previous_effort: torch.Tensor,
    drop_gain: torch.Tensor | float,
    *,
    minimum: float = 6.0,
) -> torch.Tensor:
    """Compute BAM's load-dependent battery sag with a hard voltage floor."""

    gain = torch.as_tensor(drop_gain, dtype=nominal_voltage.dtype, device=nominal_voltage.device)
    load = previous_effort.abs().sum(dim=-1, keepdim=True)
    return torch.clamp(nominal_voltage - gain * load, min=float(minimum))


class ControlStepDelay:
    """Per-environment FIFO for BAM's 3..6 control-step target delay.

    ``push`` returns the target that is visible to the actuator this control
    step.  The queue is intentionally resettable by env id, preventing stale
    actions from leaking across episode boundaries.
    """

    def __init__(self, num_envs: int, width: int, *, min_lag: int = 3, max_lag: int = 6, device=None):
        if min_lag < 0 or max_lag < min_lag:
            raise ValueError("invalid delay range")
        self.min_lag = int(min_lag)
        self.max_lag = int(max_lag)
        self.delay = torch.full((num_envs,), self.max_lag, dtype=torch.long, device=device)
        self._history = deque(maxlen=self.max_lag + 1)
        for _ in range(self.max_lag + 1):
            self._history.append(torch.zeros(num_envs, width, device=device))

    @staticmethod
    def _copy_to_writable_storage(value: torch.Tensor) -> torch.Tensor:
        """Detach an input into storage that remains mutable across resets.

        IsaacLab playback runs policy inference under ``torch.inference_mode``.
        Tensors created there cannot later be modified in ordinary mode, so a
        plain ``clone`` is not sufficient for a stateful actuator queue.
        Explicitly disabling inference mode while allocating and copying keeps
        the queue's ownership separate from the policy's temporary tensors.
        """

        with torch.inference_mode(False):
            stored = torch.empty(
                value.shape,
                dtype=value.dtype,
                device=value.device,
            )
            stored.copy_(value.detach())
        return stored

    def set_delays(self, delays: torch.Tensor) -> None:
        if delays.shape != self.delay.shape:
            raise ValueError("delay tensor must have one value per environment")
        self.delay.copy_(delays.to(device=self.delay.device, dtype=torch.long).clamp(self.min_lag, self.max_lag))

    def push(self, target: torch.Tensor) -> torch.Tensor:
        self._history.append(self._copy_to_writable_storage(target))
        stack = torch.stack(tuple(self._history), dim=0)
        # History is oldest -> newest.  A lag of zero reads the newest target.
        offsets = self.max_lag - self.delay
        env = torch.arange(target.shape[0], device=target.device)
        return stack[offsets, env]

    def reset(self, env_ids: torch.Tensor, value: torch.Tensor | None = None) -> None:
        ids = env_ids.to(device=self.delay.device, dtype=torch.long)
        fill = 0.0 if value is None else value
        # Adapters commonly pass a compact ``value[ids]`` tensor, while a
        # direct caller may pass a full ``(num_envs, width)`` tensor.  Accept
        # both forms; indexing a compact tensor by global env ids is invalid
        # whenever a non-zero subset resets.
        local_value = torch.is_tensor(fill) and fill.shape[0] == ids.numel()
        for item in self._history:
            if not torch.is_tensor(fill):
                item[ids] = fill
            elif local_value:
                item[ids] = fill
            else:
                item[ids] = fill[ids]


def sample_uniform_with_zero(
    shape: tuple[int, ...], low: torch.Tensor | float, high: torch.Tensor | float, *,
    zero_probability: float = 0.0, generator: torch.Generator | None = None, device=None,
) -> torch.Tensor:
    """Sample a uniform command while making exact zero a deliberate bucket."""

    out = torch.empty(shape, device=device)
    out.uniform_(0.0, 1.0, generator=generator)
    lo = torch.as_tensor(low, device=device, dtype=out.dtype)
    hi = torch.as_tensor(high, device=device, dtype=out.dtype)
    out = lo + out * (hi - lo)
    if zero_probability:
        if not 0.0 <= zero_probability <= 1.0:
            raise ValueError("zero_probability must be in [0, 1]")
        mask = torch.rand(shape[:-1] if len(shape) > 1 else shape, device=device, generator=generator) < zero_probability
        out[mask] = 0.0
    return out


def force_turn_in_place(command: torch.Tensor, fraction: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Force a fraction of ``(vx, vy, yaw)`` commands into the turn bucket."""

    if command.shape[-1] != 3:
        raise ValueError("velocity command must be 3D")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("turn fraction must be in [0, 1]")
    result = command.clone()
    n = command.shape[0]
    mask = torch.rand(n, device=command.device, generator=generator) < fraction
    if mask.any():
        result[mask, :2] = 0.0
        signs = torch.where(torch.rand(int(mask.sum()), device=command.device, generator=generator) < 0.5, -1.0, 1.0)
        magnitude = 0.4 + 0.6 * torch.rand(int(mask.sum()), device=command.device, generator=generator)
        result[mask, 2] = signs * magnitude
    return result


def gaussian_tracking(error: torch.Tensor, std: float) -> torch.Tensor:
    """mjlab Gaussian tracking kernel, reduced over the final dimension."""

    if std <= 0:
        raise ValueError("tracking std must be positive")
    return torch.exp(-torch.sum(error.square(), dim=-1) / (std * std))


def l1_penalty(error: torch.Tensor) -> torch.Tensor:
    """Self-negating penalty convention used by microduck MDP helpers."""

    return -error.abs().sum(dim=-1)


def observation_noise(value: torch.Tensor, amplitude: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Zero-mean uniform sensor noise with deterministic generator support."""

    if amplitude < 0:
        raise ValueError("noise amplitude must be non-negative")
    return value + torch.empty_like(value).uniform_(-amplitude, amplitude, generator=generator)


def _quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert scalar-last quaternions to rotation matrices."""

    x, y, z, w = quat.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def subtree_angular_momentum(
    mass: torch.Tensor,
    com_pos_w: torch.Tensor,
    com_lin_vel_w: torch.Tensor,
    com_ang_vel_w: torch.Tensor,
    inertia_principal: torch.Tensor,
    inertia_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Compute rigid-body subtree angular momentum about its aggregate COM.

    All body tensors use ``(env, body, ...)`` layout. ``inertia_principal``
    accepts either diagonal values, flattened matrices, or 3x3 matrices.
    The returned vector is expressed in world coordinates.
    """

    if mass.ndim == 3 and mass.shape[-1] == 1:
        mass = mass.squeeze(-1)
    if inertia_principal.shape[-1] == 9:
        inertia_principal = inertia_principal.reshape(*inertia_principal.shape[:-1], 3, 3)
    elif inertia_principal.shape[-2:] == (3, 3):
        pass
    elif inertia_principal.shape[-1] == 3:
        inertia_principal = torch.diag_embed(inertia_principal)
    else:
        raise ValueError(f"unexpected body inertia shape: {tuple(inertia_principal.shape)}")

    total_mass = mass.sum(dim=1, keepdim=True).clamp_min(torch.finfo(mass.dtype).eps)
    subtree_com = (mass.unsqueeze(-1) * com_pos_w).sum(dim=1, keepdim=True) / total_mass.unsqueeze(-1)
    relative = com_pos_w - subtree_com
    orbital = torch.cross(relative, mass.unsqueeze(-1) * com_lin_vel_w, dim=-1)
    rotation = _quat_xyzw_to_matrix(inertia_quat_w)
    world_inertia = rotation @ inertia_principal @ rotation.transpose(-1, -2)
    spin = torch.matmul(world_inertia, com_ang_vel_w.unsqueeze(-1)).squeeze(-1)
    return (orbital + spin).sum(dim=1)


__all__ = [
    "ControlStepDelay", "clip_policy_action", "effective_supply_voltage",
    "force_turn_in_place", "gaussian_tracking", "l1_penalty",
    "observation_noise", "policy_action_to_target", "sample_uniform_with_zero",
    "subtree_angular_momentum",
]
