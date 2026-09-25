"""Left-right symmetry augmentation for the IsaacLab 61D Microduck policy.

The transform mirrors the production MJLab 61D actor layout.  It is exposed
only through diagnostic runner configs; the canonical strict runner keeps
symmetry disabled so parity experiments remain auditable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Left leg (0..4) <-> right leg (9..13); neck/head joints stay in place.
_JOINT_PERM = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
_JOINT_SIGN = (-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

_OBS_PERM = (
    (0, 1, 2)
    + (3, 4, 5)
    + tuple(6 + index for index in _JOINT_PERM)
    + tuple(20 + index for index in _JOINT_PERM)
    + tuple(34 + index for index in _JOINT_PERM)
    + (48, 49, 50)
    + (51, 52, 53, 54)
    + (55, 56, 57, 58, 59, 60)
)
_OBS_SIGN = (
    (-1.0, 1.0, -1.0)
    + (1.0, -1.0, 1.0)
    + _JOINT_SIGN
    + _JOINT_SIGN
    + _JOINT_SIGN
    + (1.0, -1.0, -1.0)
    + (1.0, 1.0, -1.0, -1.0)
    + (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
)

_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if device not in _CACHE:
        _CACHE[device] = (
            torch.tensor(_OBS_PERM, dtype=torch.long, device=device),
            torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device),
            torch.tensor(_JOINT_PERM, dtype=torch.long, device=device),
            torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device),
        )
    return _CACHE[device]


@torch.no_grad()
def microduck_velocity_symmetry(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return original plus left-right mirrored observations/actions."""

    if obs is not None:
        batch = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        actor = obs["policy"]
        obs_perm, obs_sign, _, _ = _tensors(actor.device)
        obs_aug["policy"][:batch] = actor
        obs_aug["policy"][batch:] = actor[:, obs_perm] * obs_sign
        # Privileged critic inputs are not actor ABI and are repeated unchanged.
    else:
        obs_aug = None

    if actions is not None:
        batch = actions.shape[0]
        _, _, action_perm, action_sign = _tensors(actions.device)
        mirrored = actions[:, action_perm] * action_sign
        actions_aug = torch.cat((actions, mirrored), dim=0)
    else:
        actions_aug = None
    return obs_aug, actions_aug


__all__ = ["microduck_velocity_symmetry"]
