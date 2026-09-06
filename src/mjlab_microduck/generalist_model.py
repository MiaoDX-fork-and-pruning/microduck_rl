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


class GatedAdapterG0Actor(nn.Module):
    """Condition-gated shared actor with lightweight residual behavior adapters.

    The trunk and action head are shared across all G0 behaviors. Each adapter
    is a low-rank residual in hidden space, so the model can separate behavior
    corrections without embedding a complete specialist policy branch.
    """

    def __init__(self, bounded: bool = True, hidden_dim: int = 256,
                 adapter_dim: int = 32, behavior_count: int = 3):
        super().__init__()
        if hidden_dim < 1 or adapter_dim < 1 or behavior_count < 1:
            raise ValueError("gated adapter dimensions must be positive")
        self.behavior_count = behavior_count
        self.bounded = bounded
        self.trunk = nn.Sequential(
            nn.Linear(71, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.gate = nn.Linear(6, behavior_count)
        self.adapter_down = nn.ModuleList(
            nn.Linear(hidden_dim, adapter_dim) for _ in range(behavior_count)
        )
        self.adapter_up = nn.ModuleList(
            nn.Linear(adapter_dim, hidden_dim) for _ in range(behavior_count)
        )
        self.action_head = nn.Linear(hidden_dim, 14)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != 71:
            raise ValueError("G0 gated-adapter input must have shape [N,71]")
        hidden = self.trunk(observation)
        condition = observation[:, 48:54]
        weights = torch.softmax(self.gate(condition), dim=-1)
        residuals = torch.stack([
            self.adapter_up[index](torch.tanh(self.adapter_down[index](hidden)))
            for index in range(self.behavior_count)
        ], dim=1)
        hidden = hidden + (residuals * weights.unsqueeze(-1)).sum(dim=1)
        action = self.action_head(hidden)
        return torch.tanh(action) if self.bounded else action


class FiLMG0Actor(nn.Module):
    """One shared trunk/action head with condition-dependent feature modulation."""

    def __init__(self, bounded: bool = True, hidden_dim: int = 512,
                 output_hidden_dim: int = 256, behavior_count: int = 3):
        super().__init__()
        if hidden_dim < 1 or output_hidden_dim < 1 or behavior_count < 1:
            raise ValueError("FiLM dimensions must be positive")
        self.bounded = bounded
        self.behavior_count = behavior_count
        self.output_hidden_dim = output_hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(71, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_hidden_dim), nn.Tanh(),
        )
        self.condition_film = nn.Linear(6, 2 * output_hidden_dim)
        self.action_head = nn.Linear(output_hidden_dim, 14)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != 71:
            raise ValueError("G0 FiLM input must have shape [N,71]")
        hidden = self.trunk(observation)
        modulation = self.condition_film(observation[:, 48:54])
        scale = modulation[:, :self.output_hidden_dim]
        shift = modulation[:, self.output_hidden_dim:]
        hidden = hidden * (1.0 + scale) + shift
        action = self.action_head(hidden)
        return torch.tanh(action) if self.bounded else action


class RoutedG0TeacherActor(nn.Module):
    """Single 71D/14D graph routing frozen foot-mode teachers by one-hot."""
    def __init__(self, stand: nn.Module, locomotion: nn.Module, *additional: nn.Module):
        super().__init__(); self.teachers = nn.ModuleList((stand, locomotion, *additional))
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        legacy = torch.cat((observation[:, :48], observation[:, 54:67]), dim=1)
        route = observation[:, 48:54].argmax(dim=1)
        outputs = torch.stack([teacher(legacy) for teacher in self.teachers], dim=1)
        return outputs[torch.arange(observation.shape[0], device=observation.device), route]


def build_actor(metadata: dict) -> nn.Module:
    if metadata.get("model_kind") == "g0_multihead":
        return G0MultiHeadActor(bounded=metadata.get("bounded_actions", True))
    if metadata.get("model_kind") == "gated_adapter":
        return GatedAdapterG0Actor(
            bounded=metadata.get("bounded_actions", True),
            hidden_dim=int(metadata.get("hidden_dim", 256)),
            adapter_dim=int(metadata.get("adapter_dim", 32)),
            behavior_count=int(metadata.get("behavior_count", 3)),
        )
    if metadata.get("model_kind") == "film":
        return FiLMG0Actor(
            bounded=metadata.get("bounded_actions", True),
            hidden_dim=int(metadata.get("hidden_dim", 512)),
            output_hidden_dim=int(metadata.get("output_hidden_dim", 256)),
            behavior_count=int(metadata.get("behavior_count", 3)),
        )
    architecture = metadata.get("architecture", [71, 512, 256, 128, 14])
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(architecture, architecture[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(architecture) - 2:
            layers.append(nn.Tanh())
    if metadata.get("bounded_actions"):
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)
