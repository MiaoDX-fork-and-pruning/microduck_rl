"""Hybrid-only PPO adapter for the G0 frozen-teacher action anchor."""
from __future__ import annotations
import torch
from torch import nn
from rsl_rl.algorithms import PPO
from .generalist_anchor import action_anchor_loss, per_behavior_anchor_error
from .generalist_anchor_storage import GeneralistAnchorStorage

class GeneralistHybridPPO(PPO):
    """PPO with an optional masked behavior-teacher action loss."""
    def __init__(self, *args, anchor_coef: float = 0.0, behavior_count: int = 3, teacher_provider=None, metadata_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_coef, self.behavior_count = float(anchor_coef), int(behavior_count)
        self.teacher_provider, self.metadata_provider = teacher_provider, metadata_provider

    def act(self, obs):
        actions = super().act(obs)
        if self.teacher_provider is not None:
            self.transition.teacher_actions = self.teacher_provider(obs).detach()
            if self.metadata_provider is None: raise RuntimeError("metadata_provider is required with teacher_provider")
            self.transition.hold_mask, self.transition.behavior_ids = self.metadata_provider(obs)
        return actions

    def update(self):
        if not isinstance(self.storage, GeneralistAnchorStorage) or self.anchor_coef == 0.0: return super().update()
        obs = self.storage.observations.flatten(0, 1); teacher = self.storage.teacher_actions.flatten(0, 1)
        hold = self.storage.hold_mask.flatten(0, 1).squeeze(-1); behaviors = self.storage.behavior_ids.flatten(0, 1).squeeze(-1)
        mean_actions = self.actor(obs); anchor, count = action_anchor_loss(mean_actions, teacher, hold)
        self.optimizer.zero_grad(); (self.anchor_coef * anchor).backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.optimizer.step()
        errors = per_behavior_anchor_error(mean_actions.detach(), teacher, behaviors, hold, self.behavior_count)
        result = super().update(); result.update(anchor=float(anchor.detach()), anchor_samples=float(count))
        result.update({f"anchor_behavior_{i}": float(v) for i, v in enumerate(errors) if torch.isfinite(v)})
        return result
