import torch

from mjlab_microduck.generalist_teachers import FrozenG0Teachers


def test_frozen_teachers_are_batched_and_finite():
    teacher = FrozenG0Teachers(device="cpu")
    observations = torch.zeros(3, 71)
    observations[0, 48] = 1.0
    observations[1, 49] = 1.0
    observations[2, 50] = 1.0
    actions = teacher(observations)
    assert actions.shape == (3, 14)
    assert torch.isfinite(actions).all()
    assert actions.abs().max() <= 1.0
