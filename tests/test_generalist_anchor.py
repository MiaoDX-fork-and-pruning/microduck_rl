import pytest
import torch

from mjlab_microduck.generalist_anchor import (
    AnchorSchedule,
    action_anchor_loss,
    per_behavior_anchor_error,
    weighted_action_anchor_loss,
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


def test_weighted_anchor_uses_active_behavior_coefficient_and_hold_mask():
    student = torch.tensor([[1.0, 1.0], [2.0, 0.0], [9.0, 9.0]])
    teacher = torch.zeros_like(student)
    loss, count = weighted_action_anchor_loss(
        student,
        teacher,
        behavior_ids=torch.tensor([0, 1, 2]),
        hold_mask=torch.tensor([True, True, False]),
        behavior_weights=torch.tensor([0.5, 2.0, 100.0]),
    )
    # Row MSEs are 1 and 2; weights remain coefficients over the held rows.
    assert count.item() == 2
    assert loss.item() == pytest.approx((0.5 * 1.0 + 2.0 * 2.0) / 2.0)


def test_uniform_weighted_anchor_matches_scaled_unweighted_loss():
    student = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    teacher = torch.zeros_like(student)
    hold = torch.tensor([True, True])
    unweighted, _ = action_anchor_loss(student, teacher, hold)
    weighted, _ = weighted_action_anchor_loss(
        student, teacher, torch.tensor([0, 1]), hold, torch.tensor([0.25, 0.25])
    )
    assert weighted.item() == pytest.approx(0.25 * unweighted.item())


def test_weighted_anchor_rejects_unconfigured_held_behavior():
    with pytest.raises(ValueError, match="no configured anchor weight"):
        weighted_action_anchor_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.tensor([2]),
            torch.tensor([True]),
            torch.tensor([0.1, 0.2]),
        )
