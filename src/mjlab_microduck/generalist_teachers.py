"""Batched frozen specialist ensemble for hybrid G0 training."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .generalist_schema import BEHAVIOR_OFFSET, TWIST_OFFSET

_BEHAVIOR_TO_TEACHER = {0: "velstand_flat", 1: "velocity_flat", 2: "sitstand_flat"}


def compare_action_batches(native: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float | bool]:
    """Return deterministic Phase-A parity metrics for two action batches."""
    if native.ndim != 2 or reconstructed.shape != native.shape or native.shape[1] != 14:
        raise ValueError("native and reconstructed actions must both have shape [N,14]")
    if not torch.isfinite(native).all() or not torch.isfinite(reconstructed).all():
        raise ValueError("native and reconstructed actions must be finite")
    delta = (reconstructed - native).abs()
    return {
        "finite": True,
        "samples": float(native.shape[0]),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "passed": bool((delta.max() <= 1e-4) and (delta.mean() <= 1e-5)),
    }


class FrozenG0Teachers(nn.Module):
    """Run the selected frozen 61D specialist on a batch of 71D observations."""

    def __init__(self, artifact_root: str | Path = "artifacts/specialists", device: str = "cpu"):
        super().__init__()
        root = Path(artifact_root)
        self.models = nn.ModuleDict()
        self.means: dict[str, torch.Tensor] = {}
        self.stds: dict[str, torch.Tensor] = {}
        for name in _BEHAVIOR_TO_TEACHER.values():
            payload = torch.load(root / name / "checkpoint.pt", map_location=device, weights_only=False)
            actor = nn.Sequential(nn.Linear(61, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                  nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 14)).to(device)
            state = payload["actor_state_dict"]
            actor.load_state_dict({k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")})
            actor.eval()
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            self.models[name] = actor
            self.means[name] = state["obs_normalizer._mean"].reshape(-1).to(device)
            # EmpiricalNormalization exports ``_std + epsilon``. The raw
            # checkpoint std alone diverges from the deployable ONNX graph.
            self.stds[name] = (state["obs_normalizer._std"].reshape(-1) + 0.01).to(device)

    @torch.no_grad()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != 71:
            raise ValueError("G0 teacher input must have shape [N,71]")
        behavior = observations[:, BEHAVIOR_OFFSET:BEHAVIOR_OFFSET + 3].argmax(dim=1)
        legacy = torch.cat((observations[:, :48], observations[:, TWIST_OFFSET:TWIST_OFFSET + 13]), dim=1)
        result = torch.zeros((len(observations), 14), device=observations.device, dtype=observations.dtype)
        for behavior_id, name in _BEHAVIOR_TO_TEACHER.items():
            selected = behavior == behavior_id
            if selected.any():
                x = (legacy[selected] - self.means[name].to(legacy)) / self.stds[name].to(legacy).clamp_min(1e-6)
                result[selected] = self.models[name](x).clamp(-1.0, 1.0)
        return result
