"""Versioned 71D conditioned-policy input contract for generalist-v0."""

from __future__ import annotations

import numpy as np

SCHEMA = "generalist-v0"
SCHEMA_VERSION = 2
PROPRIO_DIM = 48
BEHAVIOR_DIM = 6
COND_DIM = 17
OBS_DIM = PROPRIO_DIM + BEHAVIOR_DIM + COND_DIM
ACTION_DIM = 14
BEHAVIORS = ("stand", "locomotion", "sit_stand", "ground_pick", "kick", "roulade")
BEHAVIOR_OFFSET = 48
TWIST_OFFSET = 54
HEAD_OFFSET = 57
BODY_OFFSET = 61
PHASE_OFFSET = 67
POSTURE_OFFSET = 69
SIDE_OFFSET = 70


def make_conditioned_observation(
    specialist_observation: np.ndarray,
    requested_command: np.ndarray,
    behavior: str,
) -> np.ndarray:
    """Adapt legacy 61D observations into the frozen 71D student input."""
    obs = np.asarray(specialist_observation, dtype=np.float32)
    cmd = np.asarray(requested_command, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != 61:
        raise ValueError(f"expected observation [N,61], got {obs.shape}")
    if cmd.ndim != 2 or cmd.shape[0] != obs.shape[0] or cmd.shape[1] < 13:
        raise ValueError(f"expected command [N,>=13], got {cmd.shape}")
    if behavior not in BEHAVIORS:
        raise ValueError(f"unknown behavior {behavior!r}")
    if not np.isfinite(obs).all() or not np.isfinite(cmd).all():
        raise ValueError("non-finite legacy observation or command")
    one_hot = np.zeros((obs.shape[0], BEHAVIOR_DIM), dtype=np.float32)
    one_hot[:, BEHAVIORS.index(behavior)] = 1.0
    # Legacy command block is [twist(3), head_pose(4), body_pose(6)].
    phase, posture, side = legacy_command_fields(cmd, behavior)
    condition = np.concatenate((one_hot, cmd[:, :13], phase, posture, side), axis=1)
    result = np.concatenate((obs[:, :PROPRIO_DIM], condition), axis=1)
    if result.shape[1] != OBS_DIM:
        raise AssertionError(result.shape)
    return result


def legacy_command_fields(command: np.ndarray, behavior: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cmd = np.asarray(command, dtype=np.float32)
    if cmd.ndim != 2 or cmd.shape[1] < 13 or not np.isfinite(cmd).all():
        raise ValueError("legacy command must be finite [N,>=13]")
    if behavior not in BEHAVIORS:
        raise ValueError(f"unknown behavior {behavior!r}")
    phase = np.zeros((len(cmd), 2), dtype=np.float32)
    posture = np.zeros((len(cmd), 1), dtype=np.float32)
    side = np.zeros((len(cmd), 1), dtype=np.float32)
    if behavior == "ground_pick":
        phase[:] = cmd[:, :2]
    elif behavior == "sit_stand":
        posture[:, 0] = cmd[:, 0]
    return phase, posture, side


def validate_batch(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[1] != OBS_DIM:
        raise ValueError(f"student input must be [N,{OBS_DIM}], got {x.shape}")
    if y.ndim != 2 or y.shape[1] != ACTION_DIM:
        raise ValueError(f"action must be [N,{ACTION_DIM}], got {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite teacher sample")
