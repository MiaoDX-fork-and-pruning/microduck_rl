"""Causal helpers for probing a delayed PhysX BAM friction bridge.

PhysX force getters expose the previous solved state when read from the
actuator's pre-step callback.  This module keeps that limitation explicit: a
sample captured after ``sim.step`` is only eligible for the following control
step.  It is intentionally simulator-independent so reset and tensor
contracts can be tested without Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class LaggedExternalEffort:
    """Store one-step-old external joint effort with explicit reset semantics."""

    def __init__(self, num_envs: int, num_joints: int, *, device: torch.device | str, dtype=torch.float32):
        if num_envs <= 0 or num_joints <= 0:
            raise ValueError("num_envs and num_joints must be positive")
        self.num_envs = int(num_envs)
        self.num_joints = int(num_joints)
        self._external = torch.zeros((num_envs, num_joints), device=device, dtype=dtype)
        self._valid = torch.zeros((num_envs,), device=device, dtype=torch.bool)

    @property
    def valid(self) -> torch.Tensor:
        """Whether each environment has a post-step force sample available."""

        return self._valid

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        ids = self._resolve_ids(env_ids)
        self._external[ids] = 0.0
        self._valid[ids] = False

    def observe(
        self,
        projected_joint_forces: torch.Tensor,
        actuation_forces: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Capture external effort as projected force minus actuation force.

        The returned value is detached and is safe to use at the next
        pre-step.  Non-finite or shape-mismatched getter data is rejected so a
        diagnostic cannot silently turn a backend failure into zero friction.
        """

        ids = self._resolve_ids(env_ids)
        projected = torch.as_tensor(projected_joint_forces, device=self._external.device, dtype=self._external.dtype)
        actuation = torch.as_tensor(actuation_forces, device=self._external.device, dtype=self._external.dtype)
        expected = (self.num_envs, self.num_joints) if env_ids is None else (ids.numel(), self.num_joints)
        if projected.shape != expected or actuation.shape != expected:
            raise ValueError(
                f"PhysX force tensors must have shape {expected}; got "
                f"projected={tuple(projected.shape)}, actuation={tuple(actuation.shape)}"
            )
        if not torch.isfinite(projected).all() or not torch.isfinite(actuation).all():
            raise ValueError("PhysX force getters returned non-finite values")
        external = projected - actuation
        if env_ids is None:
            self._external.copy_(external.detach())
        else:
            self._external[ids] = external.detach()
        self._valid[ids] = True
        return external.detach()

    def external_effort(self) -> torch.Tensor:
        """Return the last post-step sample, or zero during each reset warm-up."""

        return torch.where(self._valid[:, None], self._external, torch.zeros_like(self._external))

    def _resolve_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self._external.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self._external.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=self._external.device, dtype=torch.long).reshape(-1)
