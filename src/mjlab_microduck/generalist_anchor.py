"""Hybrid G0 teacher-action anchoring primitives.

This module is deliberately independent of rsl_rl.  A hybrid PPO adapter can
call these functions while direct PPO remains completely unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def action_anchor_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    hold_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean squared action error and number of selected samples.

    ``hold_mask`` excludes transition windows; an empty selection returns a
    differentiable zero tensor and count zero, which keeps PPO updates safe.
    """
    if student_actions.shape != teacher_actions.shape or student_actions.ndim != 2:
        raise ValueError("student and teacher actions must have matching shape [N,A]")
    if hold_mask.ndim != 1 or hold_mask.shape[0] != student_actions.shape[0]:
        raise ValueError("hold_mask must have shape [N]")
    if not torch.isfinite(student_actions).all() or not torch.isfinite(teacher_actions).all():
        raise ValueError("teacher and student actions must be finite")
    selected = hold_mask.to(dtype=torch.bool)
    count = selected.sum()
    if count.item() == 0:
        return student_actions.sum() * 0.0, count
    error = (student_actions[selected] - teacher_actions[selected]).pow(2).mean()
    return error, count


def per_behavior_anchor_error(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    behavior_ids: torch.Tensor,
    hold_mask: torch.Tensor,
    behavior_count: int,
) -> torch.Tensor:
    """Compute detached MSE for each behavior, with NaN for no samples."""
    if behavior_ids.ndim != 1 or behavior_ids.shape[0] != student_actions.shape[0]:
        raise ValueError("behavior_ids must have shape [N]")
    values = student_actions.new_full((behavior_count,), float("nan"))
    for behavior in range(behavior_count):
        mask = hold_mask & (behavior_ids == behavior)
        loss, count = action_anchor_loss(student_actions, teacher_actions, mask)
        if count.item():
            values[behavior] = loss.detach()
    return values


@dataclass
class AnchorSchedule:
    """Per-behavior anchor weights with threshold-gated multiplicative decay."""

    weights: list[float]
    teacher_targets: list[float]
    threshold: float = 0.9
    decay: float = 0.5
    min_weight: float = 0.0
    _unlocked: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.teacher_targets):
            raise ValueError("weights and teacher_targets must have equal length")
        if not 0.0 < self.threshold <= 1.0 or not 0.0 < self.decay < 1.0:
            raise ValueError("threshold must be in (0,1], decay must be in (0,1)")

    def update(self, measured_returns: list[float | None]) -> list[float]:
        """Decay eligible behavior weights and return a copy of current weights."""
        if len(measured_returns) != len(self.weights):
            raise ValueError("measured_returns length mismatch")
        for index, value in enumerate(measured_returns):
            if value is None or index in self._unlocked:
                continue
            if value >= self.threshold * self.teacher_targets[index]:
                self.weights[index] = max(self.min_weight, self.weights[index] * self.decay)
                self._unlocked.add(index)
        return list(self.weights)
