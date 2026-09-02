"""Auditable PyTorch actor definitions for generalist-v0 experiments."""
from __future__ import annotations

import torch
from torch import nn


class G0MultiHeadActor(nn.Module):
    """One conditioned actor with a shared trunk and stand/locomotion heads."""

    def __init__(self, bounded: bool = True):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(71, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
        self.heads = nn.ModuleList((nn.Linear(256, 14), nn.Linear(256, 14)))
        self.bounded = bounded

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(observation)
        outputs = torch.stack([head(hidden) for head in self.heads], dim=1)
        weights = observation[:, 48:50].unsqueeze(-1)
        action = (outputs * weights).sum(dim=1)
        return torch.tanh(action) if self.bounded else action


def build_actor(metadata: dict) -> nn.Module:
    if metadata.get("model_kind") == "g0_multihead":
        return G0MultiHeadActor(bounded=metadata.get("bounded_actions", True))
    architecture = metadata.get("architecture", [71, 512, 256, 128, 14])
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(architecture, architecture[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(architecture) - 2:
            layers.append(nn.Tanh())
    if metadata.get("bounded_actions"):
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)
