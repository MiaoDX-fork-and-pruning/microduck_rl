from __future__ import annotations

import pytest
import torch

from isaaclab_microduck.actuators.physx_friction_bridge import LaggedExternalEffort


def test_lagged_bridge_warms_with_zero_then_exposes_previous_sample() -> None:
    bridge = LaggedExternalEffort(2, 3, device="cpu")
    assert torch.equal(bridge.external_effort(), torch.zeros(2, 3))
    projected = torch.tensor([[3.0, 2.0, 1.0], [0.0, -1.0, 4.0]])
    actuation = torch.tensor([[1.0, 2.0, 0.5], [0.0, 1.0, 2.0]])
    observed = bridge.observe(projected, actuation)
    expected = projected - actuation
    assert torch.equal(observed, expected)
    assert torch.equal(bridge.external_effort(), expected)


def test_lagged_bridge_subset_reset_zeroes_only_selected_environment() -> None:
    bridge = LaggedExternalEffort(2, 2, device="cpu")
    bridge.observe(torch.ones(2, 2), torch.zeros(2, 2))
    bridge.reset([1])
    assert torch.equal(bridge.external_effort()[0], torch.ones(2))
    assert torch.equal(bridge.external_effort()[1], torch.zeros(2))
    assert bridge.valid.tolist() == [True, False]


def test_lagged_bridge_rejects_bad_shape_and_nonfinite_force() -> None:
    bridge = LaggedExternalEffort(1, 2, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        bridge.observe(torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="non-finite"):
        bridge.observe(torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2))
