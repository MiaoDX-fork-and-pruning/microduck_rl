import pytest
import torch

from mjlab_microduck.generalist_anchor import AnchorSchedule
from mjlab_microduck.generalist_hybrid_ppo import GeneralistHybridPPO


def _algorithm(schedule=None):
    # Exercise the schedule hook without requiring actor/critic construction.
    algorithm = object.__new__(GeneralistHybridPPO)
    algorithm.device = torch.device("cpu")
    algorithm.anchor_schedule = schedule
    weights = schedule.weights if schedule is not None else [1.0, 2.0, 3.0]
    algorithm.anchor_weights = torch.tensor(weights, dtype=torch.float32)
    return algorithm


def test_schedule_update_synchronizes_live_anchor_weights():
    schedule = AnchorSchedule([1.0, 2.0, 3.0], [10.0, 10.0, 10.0], threshold=0.9, decay=0.5)
    algorithm = _algorithm(schedule)

    result = algorithm.update_anchor_schedule([9.0, None, 0.0])

    assert result == [0.5, 2.0, 3.0]
    assert torch.equal(algorithm.anchor_weights, torch.tensor(result))
    # The one-time unlock behavior is preserved across repeated calls.
    assert algorithm.update_anchor_schedule([10.0, 9.0, 10.0]) == [0.5, 1.0, 1.5]


def test_no_schedule_is_a_valid_noop_but_validates_shape():
    algorithm = _algorithm()
    assert algorithm.update_anchor_schedule([None, 0.0, 100.0]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="length mismatch"):
        algorithm.update_anchor_schedule([1.0])

