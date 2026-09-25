from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp


def _env(num_envs: int = 12):
    asset = SimpleNamespace(
        num_joints=14,
        data=SimpleNamespace(encoder_bias=torch.zeros(num_envs, 14))
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        scene={"robot": asset},
    )


def test_sensor_corner_event_covers_coupled_startup_tails() -> None:
    env = _env()
    mdp.randomize_sensor_corners(
        env,
        None,
        fraction=0.5,
        max_angle_deg=6.0,
        bias_range=(-0.015, 0.015),
    )

    assert env._imu_misalign_quat.shape == (12, 4)
    corner_q = env._imu_misalign_quat[:6]
    # Six signed coordinate axes at the configured 6-degree boundary.
    torch.testing.assert_close(corner_q[:, 1:].norm(dim=1), torch.full((6,), 0.05233596))
    assert torch.all(env.scene["robot"].data.encoder_bias[:6].abs() == 0.015)
    assert torch.all(env.scene["robot"].data.encoder_bias[6:] == 0)


@pytest.mark.parametrize(
    "fraction, angle, bias_range",
    [(-0.1, 6.0, (-0.015, 0.015)), (1.1, 6.0, (-0.015, 0.015)),
     (0.25, float("nan"), (-0.015, 0.015)), (0.25, 6.0, (0.015, -0.015))],
)
def test_sensor_corner_event_rejects_invalid_ranges(fraction, angle, bias_range) -> None:
    with pytest.raises(ValueError, match="sensor corner"):
        mdp.randomize_sensor_corners(
            _env(), None, fraction=fraction, max_angle_deg=angle, bias_range=bias_range
        )


def test_sensor_realization_resamples_only_selected_reset_slice() -> None:
    env = _env()
    env._imu_misalign_quat = torch.zeros(12, 4)
    env._imu_misalign_quat[:, 0] = 1.0
    env.scene["robot"].data.encoder_bias[6:] = 0.25
    torch.manual_seed(20260923)

    mdp.randomize_sensor_realization(
        env,
        torch.arange(6),
        fraction=1.0,
        max_angle_deg=6.0,
        bias_range=(-0.015, 0.015),
    )

    q = env._imu_misalign_quat
    torch.testing.assert_close(q[6:], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(6, 1))
    assert torch.all(q[:6, 1:].norm(dim=1) <= 0.0523361)
    bias = env.scene["robot"].data.encoder_bias
    assert torch.all((bias[:6] >= -0.015) & (bias[:6] <= 0.015))
    assert torch.all(bias[6:] == 0.25)


@pytest.mark.parametrize(
    "fraction, angle, bias_range",
    [(-0.1, 6.0, (-0.015, 0.015)), (1.1, 6.0, (-0.015, 0.015)),
     (0.25, float("nan"), (-0.015, 0.015)), (0.25, 6.0, (0.015, -0.015))],
)
def test_sensor_realization_event_rejects_invalid_ranges(fraction, angle, bias_range) -> None:
    with pytest.raises(ValueError, match="sensor reset"):
        mdp.randomize_sensor_realization(
            _env(), None, fraction=fraction, max_angle_deg=angle, bias_range=bias_range
        )
