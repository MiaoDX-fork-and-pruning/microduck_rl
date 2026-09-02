import numpy as np
import pytest

from mjlab_microduck.generalist_schema import OBS_DIM, make_conditioned_observation, legacy_command_fields


def test_conditioned_layout_is_71d_and_preserves_proprioception():
    obs = np.arange(61, dtype=np.float32)[None, :]
    cmd = np.zeros((1, 13), dtype=np.float32)
    out = make_conditioned_observation(obs, cmd, "locomotion")
    assert out.shape == (1, OBS_DIM)
    np.testing.assert_array_equal(out[0, :48], obs[0, :48])
    np.testing.assert_array_equal(out[0, 48:54], [0, 1, 0, 0, 0, 0])


def test_adapter_rejects_bad_contract():
    with pytest.raises(ValueError):
        make_conditioned_observation(np.zeros((2, 60)), np.zeros((2, 13)), "stand")


def test_documented_offsets_and_legacy_fields():
    obs = np.zeros((1, 61), dtype=np.float32)
    cmd = np.zeros((1, 13), dtype=np.float32)
    cmd[0, 0] = 1.0
    out = make_conditioned_observation(obs, cmd, "sit_stand")
    assert out[0, 48:54].sum() == 1
    assert out[0, 67:69].tolist() == [0.0, 0.0]
    assert out[0, 69] == 1.0
    phase, posture, side = legacy_command_fields(np.array([[0.2, 0.8] + [0.0] * 11], np.float32), "ground_pick")
    np.testing.assert_allclose(phase, [[0.2, 0.8]])
    np.testing.assert_array_equal(posture, [[0.0]])
    np.testing.assert_array_equal(side, [[0.0]])


def test_non_finite_input_is_rejected():
    with pytest.raises(ValueError):
        make_conditioned_observation(np.full((1, 61), np.nan), np.zeros((1, 13)), "stand")


def test_golden_vector_round_trip(tmp_path):
    x = make_conditioned_observation(np.zeros((2, 61), np.float32), np.zeros((2, 13), np.float32), "stand")
    path = tmp_path / "golden.npz"
    np.savez(path, inputs=x, actions=np.zeros((2, 14), np.float32))
    with np.load(path, allow_pickle=False) as saved:
        assert saved["inputs"].shape == (2, 71)
        assert saved["actions"].shape == (2, 14)
