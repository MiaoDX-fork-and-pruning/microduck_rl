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

from .generalist_anchor import AnchorSchedule, per_behavior_anchor_error, weighted_action_anchor_loss
from .generalist_anchor_storage import GeneralistAnchorStorage
from .generalist_teachers import FrozenG0Teachers


class GeneralistHybridPPO(PPO):
    """PPO whose minibatch objective includes masked teacher-action MSE."""

    def __init__(
        self,
        *args,
        anchor_weights: float | list[float] = 0.0,
        anchor_behavior_count: int = 3,
        anchor_schedule: AnchorSchedule | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if anchor_behavior_count <= 0:
            raise ValueError("anchor_behavior_count must be positive")
        if anchor_schedule is not None:
            if not isinstance(anchor_schedule, AnchorSchedule):
                raise TypeError("anchor_schedule must be an AnchorSchedule")
            if len(anchor_schedule.weights) != anchor_behavior_count:
                raise ValueError("anchor_schedule length must match anchor_behavior_count")
            weights = list(anchor_schedule.weights)
        else:
            weights = (
            [float(anchor_weights)] * anchor_behavior_count
            if isinstance(anchor_weights, (int, float))
            else list(anchor_weights)
            )
        if len(weights) != anchor_behavior_count:
            raise ValueError("anchor_weights length must match anchor_behavior_count")
        if not weights or any(weight < 0.0 for weight in weights):
            raise ValueError("anchor_weights must contain non-negative values")
        self.anchor_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.anchor_schedule = anchor_schedule
        if not isinstance(self.storage, GeneralistAnchorStorage):
            raise TypeError("GeneralistHybridPPO requires GeneralistAnchorStorage")
        self.teacher_provider = None
        self.metadata_provider = None

    def update_anchor_schedule(self, measured_returns: list[float | None]) -> list[float]:
        """Update scheduled per-behavior weights and synchronize the PPO loss.

        This hook is intentionally explicit: callers provide the latest
        behavior returns at an evaluation/curriculum boundary.  With no
        schedule configured it is a no-op that returns the current weights.
        """
        if self.anchor_schedule is None:
            if len(measured_returns) != self.anchor_weights.numel():
                raise ValueError("measured_returns length mismatch")
            return self.anchor_weights.detach().cpu().tolist()
        weights = self.anchor_schedule.update(measured_returns)
        self.anchor_weights.copy_(torch.as_tensor(weights, device=self.device))
        return list(weights)

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        """Construct the hybrid algorithm with metadata-aware RL storage."""
        from rsl_rl.algorithms.ppo import (
            resolve_callable, resolve_obs_groups, resolve_rnd_config, resolve_symmetry_config
        )
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        cfg["algorithm"].pop("share_cnn_encoders", None)
        storage = GeneralistAnchorStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        algorithm = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        base_env = getattr(env, "unwrapped", env)
        algorithm.teacher_provider = FrozenG0Teachers(device=device)
        algorithm.metadata_provider = lambda _obs: (
            getattr(base_env, "g0_transition_destination") < 0,
            getattr(base_env, "g0_behavior_id"),
        )
        return algorithm

    def act(self, obs):
        actions = super().act(obs)
        if self.teacher_provider is None or self.metadata_provider is None:
            raise RuntimeError("hybrid PPO requires teacher and metadata providers")
        teacher_obs = obs["actor"] if hasattr(obs, "keys") and "actor" in obs.keys() else obs
        self.transition.teacher_actions = self.teacher_provider(teacher_obs).detach()
        hold, behaviors = self.metadata_provider(obs)
        self.transition.hold_mask = hold.detach()
        self.transition.behavior_ids = behaviors.detach()
        return actions

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("hybrid anchor currently supports feed-forward policies only")
        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "anchor": 0.0, "anchor_count": 0.0}
        behavior_error_sums = torch.zeros_like(self.anchor_weights)
        behavior_error_counts = torch.zeros_like(self.anchor_weights)
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
            behavior_ids = batch.behavior_ids.squeeze(-1)
            hold_mask = batch.hold_mask.squeeze(-1)
            anchor, count = weighted_action_anchor_loss(
                mean_actions, batch.teacher_actions, behavior_ids, hold_mask, self.anchor_weights
            )
            behavior_errors = per_behavior_anchor_error(
                mean_actions, batch.teacher_actions, behavior_ids, hold_mask, self.anchor_weights.numel()
            )
            present = torch.isfinite(behavior_errors)
            for behavior in present.nonzero(as_tuple=False).flatten():
                sample_count = (hold_mask & (behavior_ids == behavior)).sum()
                behavior_error_sums[behavior] += behavior_errors[behavior] * sample_count
                behavior_error_counts[behavior] += sample_count
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean() + anchor
            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["value"] += value_loss.item(); totals["surrogate"] += surrogate.item(); totals["entropy"] += entropy.mean().item()
            totals["anchor"] += anchor.item(); totals["anchor_count"] += count.item()
        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        metrics = {"value": totals["value"]/n, "surrogate": totals["surrogate"]/n, "entropy": totals["entropy"]/n,
                   "anchor": totals["anchor"]/n, "anchor_count": totals["anchor_count"]/n}
        for behavior in range(behavior_error_sums.numel()):
            count = behavior_error_counts[behavior].item()
            metrics[f"anchor_error_behavior_{behavior}"] = (
                behavior_error_sums[behavior].item() / count if count else float("nan")
            )
            metrics[f"anchor_weight_behavior_{behavior}"] = self.anchor_weights[behavior].item()
        return metrics


def construct_algorithm(actor: Any, critic: Any, *, num_envs: int, num_transitions_per_env: int,
                        obs: Any, actions_shape: tuple[int, ...], device: str = "cpu", **kwargs) -> GeneralistHybridPPO:
    """Construct hybrid PPO and its metadata-aware rollout storage."""
    storage = GeneralistAnchorStorage("rl", num_envs, num_transitions_per_env, obs, actions_shape, device)
    return GeneralistHybridPPO(actor, critic, storage, device=device, **kwargs)
