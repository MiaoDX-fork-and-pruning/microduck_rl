"""RL rollout storage extension for the hybrid G0 teacher anchor.

The adapter deliberately remains opt-in: direct PPO continues to use
``rsl_rl.storage.RolloutStorage`` unchanged.  Teacher metadata is carried
alongside each RL transition and is shuffled with the other flattened fields.
"""

from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage


class GeneralistAnchorStorage(RolloutStorage):
    """Rollout storage carrying frozen-teacher actions and anchor metadata."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.training_type != "rl":
            raise ValueError("GeneralistAnchorStorage requires training_type='rl'")
        self.teacher_actions = torch.zeros_like(self.actions)
        self.hold_mask = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, dtype=torch.bool, device=self.device
        )
        self.behavior_ids = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, dtype=torch.long, device=self.device
        )

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Store a normal transition plus teacher action, mask, and behavior id."""
        for name in ("teacher_actions", "hold_mask", "behavior_ids"):
            if not hasattr(transition, name) or getattr(transition, name) is None:
                raise ValueError(f"anchor transition is missing {name}")
        super().add_transition(transition)
        index = self.step - 1
        self.teacher_actions[index].copy_(transition.teacher_actions)
        self.hold_mask[index].copy_(transition.hold_mask.view(-1, 1).bool())
        self.behavior_ids[index].copy_(transition.behavior_ids.view(-1, 1).long())

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Yield parent RL batches with metadata shuffled by identical indices."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore
        teacher = self.teacher_actions.flatten(0, 1)
        hold = self.hold_mask.flatten(0, 1)
        behaviors = self.behavior_ids.flatten(0, 1)
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx], actions=actions[batch_idx],
                    values=values[batch_idx], advantages=advantages[batch_idx], returns=returns[batch_idx],
                    old_actions_log_prob=old_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_params),
                )
                batch.teacher_actions = teacher[batch_idx]
                batch.hold_mask = hold[batch_idx]
                batch.behavior_ids = behaviors[batch_idx]
                yield batch
