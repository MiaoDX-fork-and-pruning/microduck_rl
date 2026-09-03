import torch
import pytest
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from mjlab_microduck.generalist_anchor_storage import GeneralistAnchorStorage


def _storage():
    obs = TensorDict({"actor": torch.zeros(3)}, batch_size=[2])
    storage = GeneralistAnchorStorage("rl", 2, 2, obs, (2,))
    for t in range(2):
        tr = RolloutStorage.Transition()
        tr.observations = TensorDict({"actor": torch.full((2, 3), float(t))}, batch_size=[2])
        tr.actions = torch.full((2, 2), float(t))
        tr.rewards = torch.ones(2)
        tr.dones = torch.zeros(2)
        tr.values = torch.zeros(2, 1)
        tr.actions_log_prob = torch.zeros(2, 1)
        tr.distribution_params = (torch.zeros(2, 2),)
        tr.teacher_actions = torch.full((2, 2), float(10 + t))
        tr.hold_mask = torch.tensor([t == 0, t == 1])
        tr.behavior_ids = torch.tensor([t, t + 1])
        storage.add_transition(tr)
    storage.compute_returns(torch.zeros(2, 1), 0.99, 0.95)
    return storage


def test_rejects_non_rl_storage():
    obs = TensorDict({"actor": torch.zeros(3)}, batch_size=[1])
    with pytest.raises(ValueError, match="training_type='rl'"):
        GeneralistAnchorStorage("distillation", 1, 1, obs, (2,))


def test_anchor_metadata_is_carried_through_minibatch():
    storage = _storage()
    batch = next(storage.mini_batch_generator(1, 1))
    assert batch.teacher_actions.shape == (4, 2)
    assert batch.hold_mask.dtype is torch.bool
    assert batch.behavior_ids.dtype is torch.long
    pairs = {(int(a[0]), int(m[0]), int(b[0])) for a, m, b in zip(batch.teacher_actions, batch.hold_mask, batch.behavior_ids)}
    assert pairs == {(10, 1, 0), (10, 0, 1), (11, 1, 1), (11, 0, 2)}


def test_missing_anchor_field_fails_fast():
    storage = _storage()
    tr = RolloutStorage.Transition()
    tr.observations = TensorDict({"actor": torch.zeros(2, 3)}, batch_size=[2])
    with pytest.raises(ValueError, match="teacher_actions"):
        storage.add_transition(tr)
