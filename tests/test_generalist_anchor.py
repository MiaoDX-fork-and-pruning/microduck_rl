import pytest
import torch

from mjlab_microduck.generalist_anchor import (
    AnchorSchedule,
    action_anchor_loss,
    per_behavior_anchor_error,
)


def test_anchor_loss_masks_transition_rows():
    student = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    teacher = torch.zeros_like(student)
    loss, count = action_anchor_loss(student, teacher, torch.tensor([True, False]))
    assert count.item() == 1
    assert loss.item() == pytest.approx(0.5)


def test_anchor_loss_empty_selection_is_differentiable_zero():
    student = torch.ones(2, 3, requires_grad=True)
    loss, count = action_anchor_loss(student, torch.zeros_like(student), torch.zeros(2, dtype=torch.bool))
    assert count.item() == 0 and loss.item() == 0.0
    loss.backward()
    assert student.grad is not None


def test_per_behavior_error_reports_nan_for_missing_behavior():
    student = torch.tensor([[1.0], [2.0], [4.0]])
    teacher = torch.zeros_like(student)
    result = per_behavior_anchor_error(student, teacher, torch.tensor([0, 0, 2]), torch.ones(3, dtype=torch.bool), 3)
    assert result[0].item() == pytest.approx(2.5)
    assert torch.isnan(result[1])
    assert result[2].item() == pytest.approx(16.0)


def test_schedule_decays_once_only_after_threshold():
    schedule = AnchorSchedule([1.0, 0.8], [10.0, 20.0])
    assert schedule.update([8.9, None]) == [1.0, 0.8]
    assert schedule.update([9.0, 18.0]) == [0.5, 0.4]
    assert schedule.update([100.0, 100.0]) == [0.5, 0.4]
