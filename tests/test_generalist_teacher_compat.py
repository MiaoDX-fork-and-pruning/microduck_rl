import pytest
import torch
import numpy as np

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


def test_frozen_teacher_uses_exported_normalizer_denominator():
    from mjlab_microduck.generalist_teachers import FrozenG0Teachers
    import onnxruntime as ort
    from mjlab_microduck.generalist_schema import make_conditioned_observation
    data = np.load('/tmp/g0-p0-repro/velstand_flat/canonical.npz')
    x = make_conditioned_observation(data['observation'][:2], data['requested_command'][:2], 'stand')
    reconstructed = FrozenG0Teachers()(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession('artifacts/specialists/velstand_flat/policy.onnx', providers=['CPUExecutionProvider'])
    native = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None].astype(np.float32)})[0] for row in data['observation'][:2]])
    assert np.max(np.abs(reconstructed - native)) <= 1e-5
