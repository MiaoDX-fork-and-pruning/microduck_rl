"""Auditable PyTorch actor definitions for generalist-v0 experiments."""
from __future__ import annotations

import torch
from torch import nn


class VelstandProbeActor(nn.Module):
    """Frozen Phase-B architecture matching the accepted specialist MLP."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(71, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 14),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation).clamp(-1.0, 1.0)


def initialize_velstand_probe_from_teacher(
    model: VelstandProbeActor, checkpoint: str,
) -> dict[str, object]:
    """Exactly fold the 61D teacher normalizer into the 71D probe actor."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["actor_state_dict"]
    mean = state["obs_normalizer._mean"].reshape(-1)
    std = state["obs_normalizer._std"].reshape(-1) + 0.01
    legacy_to_g0 = tuple(range(48)) + tuple(range(54, 67))
    with torch.no_grad():
        first = model.mlp[0]
        first.weight.zero_()
        normalized_weight = state["mlp.0.weight"] / std.unsqueeze(0)
        first.weight[:, legacy_to_g0] = normalized_weight
        first.bias.copy_(state["mlp.0.bias"] - normalized_weight @ mean)
        for index in (2, 4, 6):
            model.mlp[index].weight.copy_(state[f"mlp.{index}.weight"])
            model.mlp[index].bias.copy_(state[f"mlp.{index}.bias"])
    return {
        "mapping": {"specialist[0:48]": "student[0:48]", "specialist[48:61]": "student[54:67]"},
        "unused_student_inputs_zeroed": list(range(48, 54)) + list(range(67, 71)),
        "normalizer": "folded_into_first_linear_std_plus_0.01",
    }


class G0MultiHeadActor(nn.Module):
    """One conditioned actor with a shared trunk and three G0 behavior heads."""

    def __init__(self, bounded: bool = True):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(71, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
        self.heads = nn.ModuleList(nn.Linear(256, 14) for _ in range(3))
        self.bounded = bounded

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(observation)
        outputs = torch.stack([head(hidden) for head in self.heads], dim=1)
        weights = observation[:, 48:51].unsqueeze(-1)
        action = (outputs * weights).sum(dim=1)
        return torch.tanh(action) if self.bounded else action


class RoutedG0TeacherActor(nn.Module):
    """Single 71D/14D graph routing frozen G0 teachers by behavior one-hot."""
    def __init__(self, stand: nn.Module, locomotion: nn.Module):
        super().__init__(); self.stand = stand; self.locomotion = locomotion
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy = torch.cat((observation[:, :48], observation[:, 54:67]), dim=1)
        route = observation[:, 49:50]
        return torch.where(route > 0.5, self.locomotion(legacy), self.stand(legacy))


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
