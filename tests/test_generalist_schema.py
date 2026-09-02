import numpy as np
import pytest

from mjlab_microduck.generalist_schema import OBS_DIM, make_conditioned_observation


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
