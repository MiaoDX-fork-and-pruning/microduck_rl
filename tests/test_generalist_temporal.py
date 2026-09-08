import numpy as np
import pytest
import torch

from mjlab_microduck.generalist_temporal import H4Actor, History, OBS_DIM, build_windows, make_window


def test_h4_shape_and_first_frame_padding():
    frames = np.arange(3 * 71, dtype=np.float32).reshape(3, 71)
    ids = np.zeros(3, dtype=np.int64)
    result = make_window(frames, end=0, segment_ids=ids)
    assert result.shape == (215,)
    np.testing.assert_array_equal(result[: 3 * 48], np.tile(frames[0, :48], 3))
    np.testing.assert_array_equal(result[-23:], frames[0, 48:])


def test_h4_resets_and_pads_at_segment_boundary():
    frames = np.zeros((5, 71), dtype=np.float32)
    ids = np.array([0, 0, 0, 1, 1])
    frames[:, :48] = np.arange(5, dtype=np.float32)[:, None]
    result = make_window(frames, end=3, segment_ids=ids)
    np.testing.assert_array_equal(result[: 4 * 48], np.tile(frames[3, :48], 4))


def test_h4_windows_are_deterministic_and_newest_condition_only():
    frames = np.zeros((5, 71), dtype=np.float32)
    frames[:, :48] = np.arange(5, dtype=np.float32)[:, None]
    frames[:, 48:] = np.arange(5 * 23, dtype=np.float32).reshape(5, 23)
    windows = build_windows(frames, np.zeros(5, dtype=np.int64))
    np.testing.assert_array_equal(windows, build_windows(frames, np.zeros(5, dtype=np.int64)))
    np.testing.assert_array_equal(windows[3, -23:], frames[3, 48:])
    np.testing.assert_array_equal(windows[3, -23 - 48:-23], frames[3, :48])


def test_h4_actor_is_shared_215_to_14():
    model = H4Actor()
    output = model(torch.zeros(2, OBS_DIM))
    assert output.shape == (2, 14)
    assert sum(p.numel() for p in model.action_head.parameters()) > 0


def test_online_history_matches_batch_windows_and_reset():
    frames = np.arange(5 * 71, dtype=np.float32).reshape(5, 71)
    history = History()
    online = []
    ids = np.array([0, 0, 0, 1, 1])
    for index, frame in enumerate(frames):
        if index == 0 or ids[index] != ids[index - 1]:
            history.reset()
        online.append(history.append(frame))
    np.testing.assert_array_equal(np.stack(online), build_windows(frames, ids))
