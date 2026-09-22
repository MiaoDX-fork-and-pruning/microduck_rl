from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp


def _env(num_envs: int = 12):
    asset = SimpleNamespace(
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
