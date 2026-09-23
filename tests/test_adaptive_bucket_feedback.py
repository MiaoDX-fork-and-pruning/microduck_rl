"""Contract tests for command-conditioned adaptive training evidence."""

from __future__ import annotations

import pytest
import torch

from mjlab_microduck.tasks.adaptive_runner import (
    COMMAND_FEEDBACK_BUCKETS,
    BucketFeedbackTracker,
)


def test_bucket_feedback_records_sample_counts_and_dt_scaled_weighted_mass():
    tracker = BucketFeedbackTracker(
        ("track_linear_velocity", "linear_velocity_error_l1"),
        device="cpu",
        step_dt=0.02,
    )
    bucket_ids = torch.tensor([0, 1, 1, 5, 6, -1])
    reward_rates = torch.tensor(
        [[2.0, -0.5], [1.0, -1.0], [3.0, -2.0], [4.0, -0.25], [5.0, -0.75], [9.0, 9.0]]
    )
    tracker.record(bucket_ids, reward_rates)

    payload = tracker.state_dict()
    assert payload["sample_count"] == {
        "zero": 1,
        "forward": 2,
        "lateral": 0,
        "yaw": 0,
        "turn-left": 0,
        "turn-right": 1,
        "nominal": 1,
    }
    assert payload["unclassified_count"] == 1
    assert payload["sample_fraction"]["forward"] == pytest.approx(2 / 5)
    assert payload["weighted_reward_mass"]["forward"] == {
        "track_linear_velocity": pytest.approx(0.08),
        "linear_velocity_error_l1": pytest.approx(-0.06),
    }
    assert payload["weighted_reward_abs_mass"]["forward"]["linear_velocity_error_l1"] == pytest.approx(0.06)
    assert payload["weighted_reward_mass"]["nominal"] == {
        "track_linear_velocity": pytest.approx(0.1),
        "linear_velocity_error_l1": pytest.approx(-0.015),
    }
    assert payload["bucket_names"] == list(COMMAND_FEEDBACK_BUCKETS)


def test_bucket_feedback_snapshot_resets_only_after_copy_and_roundtrips():
    tracker = BucketFeedbackTracker(("reward",), device="cpu", step_dt=0.02)
    tracker.record(torch.tensor([2, 2]), torch.tensor([[1.0], [2.0]]))
    snapshot = tracker.snapshot()
    assert snapshot["sample_count"]["lateral"] == 2
    assert tracker.state_dict()["sample_count"] == dict.fromkeys(COMMAND_FEEDBACK_BUCKETS, 0)

    restored = BucketFeedbackTracker(("reward",), device="cpu", step_dt=0.02)
    restored.load_state_dict(snapshot)
    assert restored.state_dict() == snapshot


def test_bucket_feedback_rejects_wrong_reward_schema():
    tracker = BucketFeedbackTracker(("reward",), device="cpu", step_dt=0.02)
    payload = tracker.state_dict()
    payload["term_names"] = ["different"]
    with pytest.raises(ValueError, match="reward term mismatch"):
        tracker.load_state_dict(payload)


def test_bucket_feedback_resumes_when_adaptive_term_is_added():
    source = BucketFeedbackTracker(("reward",), device="cpu", step_dt=0.02)
    source.record(torch.tensor([0]), torch.tensor([[2.0]]))
    payload = source.state_dict()

    restored = BucketFeedbackTracker(("reward", "new_adaptive_cost"), device="cpu", step_dt=0.02)
    restored.load_state_dict(payload)
    assert restored.state_dict()["weighted_reward_mass"]["zero"] == {
        "reward": pytest.approx(0.04),
        "new_adaptive_cost": pytest.approx(0.0),
    }


def test_bucket_feedback_rejects_removed_persisted_term():
    source = BucketFeedbackTracker(("reward", "old_term"), device="cpu", step_dt=0.02)
    payload = source.state_dict()
    restored = BucketFeedbackTracker(("reward",), device="cpu", step_dt=0.02)
    with pytest.raises(ValueError, match="reward term mismatch"):
        restored.load_state_dict(payload)


def test_bucket_feedback_buffers_remain_mutable_when_created_in_rollout_mode():
    with torch.inference_mode():
        tracker = BucketFeedbackTracker(("reward",), device="cpu", step_dt=0.02)
    assert not tracker.sample_count.is_inference()
    tracker.record(torch.tensor([0]), torch.tensor([[1.0]]))
    tracker.reset()
