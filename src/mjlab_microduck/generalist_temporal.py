"""H4 temporal-window contract for the generalist G0 candidate."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

PROPRIO_DIM = 48
CONDITION_DIM = 23
WINDOW = 4
OBS_DIM = WINDOW * PROPRIO_DIM + CONDITION_DIM
ACTION_DIM = 14
SCHEMA = "generalist-g0-h4"
SCHEMA_VERSION = 1


def make_window(frames: np.ndarray, *, end: int, segment_ids: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    segment_ids = np.asarray(segment_ids)
    if frames.ndim != 2 or frames.shape[1] != 71:
        raise ValueError("frames must have shape [N,71]")
    if segment_ids.shape != (len(frames),):
        raise ValueError("segment_ids must align with frames")
    if not 0 <= end < len(frames):
        raise IndexError(end)
    candidates = np.flatnonzero(segment_ids[:end + 1] != segment_ids[end])
    segment_start = int(candidates[-1] + 1) if len(candidates) else 0
    start = max(segment_start, end - WINDOW + 1)
    proprio = frames[start:end + 1, :PROPRIO_DIM]
    if len(proprio) < WINDOW:
        proprio = np.concatenate([np.repeat(proprio[:1], WINDOW - len(proprio), axis=0), proprio])
    return np.concatenate([proprio.reshape(-1), frames[end, PROPRIO_DIM:]], axis=0)


def build_windows(frames: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    return np.stack([make_window(frames, end=i, segment_ids=segment_ids) for i in range(len(frames))]).astype(np.float32)


class History:
    """Online H4 history with explicit segment resets."""
    def __init__(self):
        self._proprio: list[np.ndarray] = []

    def reset(self) -> None:
        self._proprio.clear()

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (71,) or not np.isfinite(frame).all():
            raise ValueError("frame must be finite [71]")
        self._proprio.append(frame[:PROPRIO_DIM].copy())
        self._proprio = self._proprio[-WINDOW:]
        padded = [self._proprio[0]] * (WINDOW - len(self._proprio)) + self._proprio
        return np.concatenate([*padded, frame[PROPRIO_DIM:]]).astype(np.float32)


def validate_batch(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[1] != OBS_DIM or y.ndim != 2 or y.shape[1] != ACTION_DIM:
        raise ValueError(f"expected [{OBS_DIM}, {ACTION_DIM}] batches, got {x.shape}, {y.shape}")
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid temporal batch")


class H4Actor(nn.Module):
    """One shared temporal trunk and one shared 14D action head."""
    def __init__(self, hidden_dim: int = 256, bounded: bool = True):
        super().__init__()
        self.bounded = bounded
        self.trunk = nn.Sequential(nn.Linear(OBS_DIM, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.action_head = nn.Linear(hidden_dim, ACTION_DIM)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != OBS_DIM:
            raise ValueError(f"H4 input must have shape [N,{OBS_DIM}]")
        action = self.action_head(self.trunk(observation))
        return torch.tanh(action) if self.bounded else action
