from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from isaaclab_microduck.policy_abi import (
    ACTION_SIZE,
    HOME_POSITION,
    OBSERVATION_SIZE,
    action_to_target,
    build_observation,
)


def test_observation_layout_is_61d() -> None:
    observation = build_observation([0] * 3, [0] * 3, [0] * 14, [0] * 14, [0] * 14, [0] * 13)

    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32


def test_action_transform_is_home_relative() -> None:
    raw = np.ones(ACTION_SIZE, dtype=np.float32)

    np.testing.assert_allclose(action_to_target(raw, scale=0.25), HOME_POSITION + 0.25)


def test_abi_rejects_wrong_field_or_action_sizes() -> None:
    with pytest.raises(ValueError):
        build_observation([0] * 2, [0] * 3, [0] * 14, [0] * 14, [0] * 14, [0] * 13)
    with pytest.raises(ValueError):
        action_to_target([0] * (ACTION_SIZE - 1))


def test_golden_fixture_is_stable() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/microduck_policy_abi_v1.json").read_text()
    )
    observation = build_observation(*(fixture[name] for name in (
        "gyro", "projected_gravity", "joint_position_relative_home",
        "joint_velocity", "previous_raw_action", "command",
    )))

    assert observation.shape == (61,)
    expected_target = np.asarray(
        [0.0, 0.0127, -0.5579, 0.1951, 0.2530, 0.6491, 0.0491,
         0.4, -0.4, 0.9579, -0.4127, 1.0579, -0.5951, 0.2470],
        dtype=np.float32,
    )
    np.testing.assert_allclose(action_to_target(fixture["raw_action"]), expected_target)
