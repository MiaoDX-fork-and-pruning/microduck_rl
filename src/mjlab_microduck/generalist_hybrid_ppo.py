"""Opt-in PPO adapter for the G0 teacher-action anchor.

The anchor is part of the same objective and optimizer step as PPO.  This is
deliberately a small feed-forward adapter; direct PPO continues to use rsl_rl.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage

from .generalist_anchor import action_anchor_loss
from .generalist_anchor_storage import GeneralistAnchorStorage


class GeneralistHybridPPO(PPO):
    """PPO whose minibatch objective includes masked teacher-action MSE."""

    def __init__(self, *args, anchor_weights: float | list[float] = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_weight = float(anchor_weights if isinstance(anchor_weights, (int, float)) else max(anchor_weights, default=0.0))
        if not isinstance(self.storage, GeneralistAnchorStorage):
            raise TypeError("GeneralistHybridPPO requires GeneralistAnchorStorage")

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("hybrid anchor currently supports feed-forward policies only")
        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "anchor": 0.0, "anchor_count": 0.0}
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            logp = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            entropy = self.actor.output_entropy
            ratio = torch.exp(logp - torch.squeeze(batch.old_actions_log_prob))
            adv = torch.squeeze(batch.advantages)
            surrogate = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)).mean()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values-batch.returns).pow(2), (clipped-batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns-values).pow(2).mean()
            mean_actions = self.actor(batch.observations.detach().clone())
            anchor, count = action_anchor_loss(mean_actions, batch.teacher_actions, batch.hold_mask.squeeze(-1))
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean() + self.anchor_weight * anchor
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["value"] += value_loss.item(); totals["surrogate"] += surrogate.item(); totals["entropy"] += entropy.mean().item()
            totals["anchor"] += anchor.item(); totals["anchor_count"] += count.item()
        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {"value": totals["value"]/n, "surrogate": totals["surrogate"]/n, "entropy": totals["entropy"]/n,
                "anchor": totals["anchor"]/n, "anchor_count": totals["anchor_count"]/n}


def construct_algorithm(actor: Any, critic: Any, *, num_envs: int, num_transitions_per_env: int,
                        obs: Any, actions_shape: tuple[int, ...], device: str = "cpu", **kwargs) -> GeneralistHybridPPO:
    """Construct hybrid PPO and its metadata-aware rollout storage."""
    storage = GeneralistAnchorStorage("rl", num_envs, num_transitions_per_env, obs, actions_shape, device)
    return GeneralistHybridPPO(actor, critic, storage, device=device, **kwargs)
