from __future__ import annotations

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
