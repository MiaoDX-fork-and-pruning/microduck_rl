import pytest
import torch

from mjlab_microduck.generalist_teachers import compare_action_batches


def test_compare_action_batches_reports_phase_a_thresholds():
    native = torch.zeros(2, 14)
    reconstructed = native.clone()
    reconstructed[0, 0] = 1e-5
    report = compare_action_batches(native, reconstructed)
    assert report["finite"] is True
    assert report["max_abs"] == pytest.approx(1e-5)
    assert report["passed"] is True


def test_compare_action_batches_rejects_wrong_shape():
    with pytest.raises(ValueError):
        compare_action_batches(torch.zeros(1, 13), torch.zeros(1, 13))
