"""Feedback tracking keeps a gradient without taxing zero axes at tiny scales."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    ADAPTIVE_RECIPES,
    FEEDBACK_LINEAR_DEADBAND_M_S,
    FEEDBACK_LINEAR_L1_WEIGHT,
    FEEDBACK_LINEAR_MIN_SCALE_M_S,
    FEEDBACK_TRACKING_TAU_S,
    FEEDBACK_YAW_DEADBAND_RAD_S,
    FEEDBACK_YAW_L1_WEIGHT,
    FEEDBACK_YAW_MIN_SCALE_RAD_S,
    make_microduck_adaptive_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def _env(command, *, linear=None, angular=None):
    command = torch.tensor(command, dtype=torch.float64)
    zeros = torch.zeros_like(command)
    data = SimpleNamespace(
        root_link_lin_vel_b=zeros if linear is None else torch.tensor(linear, dtype=zeros.dtype),
        root_link_ang_vel_b=zeros if angular is None else torch.tensor(angular, dtype=zeros.dtype),
    )
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda name: command if name == "twist" else None),
        scene={"robot": SimpleNamespace(data=data)},
        step_dt=0.02,
        common_step_counter=0,
        episode_length_buf=torch.zeros(len(command), dtype=torch.long),
    )


def _linear(env):
    return mdp.command_normalized_linear_velocity_l1(
        env,
        command_name="twist",
        minimum_scale=FEEDBACK_LINEAR_MIN_SCALE_M_S,
        deadband=FEEDBACK_LINEAR_DEADBAND_M_S,
    )


def _yaw(env):
    return mdp.command_normalized_yaw_velocity_l1(
        env,
        command_name="twist",
        minimum_scale=FEEDBACK_YAW_MIN_SCALE_RAD_S,
        deadband=FEEDBACK_YAW_DEADBAND_RAD_S,
    )


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("axis", [0, 1])
def test_linear_tracking_improves_toward_either_command_direction(axis, sign):
    command = torch.zeros((4, 3), dtype=torch.float64)
    command[:, axis] = sign * 0.12
    actual = torch.zeros_like(command)
    actual[:, axis] = sign * torch.tensor([-0.12, 0.0, 0.06, 0.12])
    # Vertical bouncing is already priced by the canonical Gaussian; it must
    # not contaminate the additional planar acquisition signal.
    actual[:, 2] = 3.0
    reward = _linear(_env(command.tolist(), linear=actual.tolist()))
    assert torch.all(reward <= 0)
    assert torch.all(torch.diff(reward) > 0)
    torch.testing.assert_close(reward[-1], torch.tensor(0.0, dtype=torch.float64), atol=1e-6, rtol=0)


def test_zero_linear_command_has_deadband_then_finite_gait_scale_penalty():
    env = _env(
        [[0.0, 0.0, 0.0]] * 4,
        linear=[[0.0, 0.0, 4.0], [0.01, -0.01, 4.0], [0.07, 0.0, 4.0], [0.0, -0.13, 4.0]],
    )
    torch.testing.assert_close(
        _linear(env),
        torch.tensor(
            [0.0, -((2**0.5 * 0.01 - 0.01) / 0.12), -0.5, -1.0],
            dtype=torch.float64,
        ),
    )


def test_linear_normalizer_uses_command_speed_for_both_axes():
    env = _env(
        [[0.12, 0.0, 0.0], [0.24, 0.0, 0.0], [0.144, 0.192, 0.0]],
        linear=[[0.06, 0.0, 0.0], [0.12, 0.0, 0.0], [0.072, 0.096, 0.0]],
    )
    # The same aligned residual fraction has equal cost for short, long, and
    # diagonal commands; the scalar command norm sets the denominator.
    torch.testing.assert_close(_linear(env), torch.full((3,), -0.5, dtype=torch.float64))


def test_linear_orthogonal_gait_motion_is_free_for_active_command():
    env = _env(
        [[0.24, 0.0, 0.0], [0.24, 0.0, 0.0]],
        # Same aligned error (−0.12 m/s), with a large orthogonal oscillation
        # added to the second rollout.
        linear=[[0.12, 0.0, 0.0], [0.12, 0.8, 0.0]],
    )
    torch.testing.assert_close(_linear(env), torch.full((2,), -0.5, dtype=torch.float64))


def test_linear_active_command_prices_aligned_error_inside_old_deadband():
    env = _env([[0.12, 0.0, 0.8]], linear=[[0.115, 3.0, 0.0]])
    # The 0.005 m/s aligned miss is intentionally visible; only the near-zero
    # command branch uses the linear deadband.
    torch.testing.assert_close(_linear(env), torch.tensor([-0.005 / 0.12], dtype=torch.float64))


@pytest.mark.parametrize("sign", [-1, 1])
def test_yaw_tracking_improves_toward_either_command_direction(sign):
    env = _env(
        [[0.12, 0.0, sign * 0.8]] * 4,
        angular=[[3.0, -2.0, sign * value] for value in [-0.8, 0.0, 0.4, 0.8]],
    )
    reward = _yaw(env)
    assert torch.all(reward <= 0)
    assert torch.all(torch.diff(reward) > 0)
    assert reward[-1] == 0


def test_zero_yaw_command_has_no_extra_penalty():
    env = _env(
        [[0.12, 0.0, 0.0], [0.12, 0.0, 0.04], [0.12, 0.0, -0.05], [0.12, 0.0, 0.0]],
        angular=[[3.0, -2.0, value] for value in [0.0, 0.05, -0.05, 0.85]],
    )
    torch.testing.assert_close(_yaw(env), torch.zeros(4, dtype=torch.float64))


def test_yaw_normalization_tracks_command_magnitude_and_ignores_roll_pitch():
    env = _env(
        [[0.0, 0.0, 0.8], [0.0, 0.0, 1.6], [0.0, 0.0, -1.6]],
        angular=[[3.0, -2.0, 0.35], [6.0, -4.0, 0.75], [-3.0, 2.0, -0.75]],
    )
    torch.testing.assert_close(_yaw(env), torch.full((3,), -0.5, dtype=torch.float64))


def test_yaw_moving_command_keeps_small_error_deadband():
    env = _env([[0.12, 0.0, 0.8]], angular=[[0.0, 0.0, 0.825]])
    assert _yaw(env).item() == 0


def _averaged_linear(env, tau_s=FEEDBACK_TRACKING_TAU_S):
    return mdp.command_normalized_linear_velocity_l1(
        env, command_name="twist", minimum_scale=0.12, deadband=0.01, tau_s=tau_s
    )


def _averaged_yaw(env, tau_s=FEEDBACK_TRACKING_TAU_S):
    return mdp.command_normalized_yaw_velocity_l1(
        env, command_name="twist", minimum_scale=0.8, deadband=0.05, tau_s=tau_s
    )


def test_averaged_tracking_prefers_progress_with_gait_sway_to_standing():
    # Walking in the commanded direction is better on average, but the lateral
    # sway of a biped crosses the target. Instantaneous L1 ranks it worse than
    # standing; averaging must reverse that ranking without changing actions.
    env = _env([[0.0, 0.12, 0.8]] * 2)
    data = env.scene["robot"].data
    averaged, instantaneous, yaw = [], [], []
    for step in range(1, 501):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        ripple = torch.sin(torch.tensor(2 * torch.pi * 2 * step * env.step_dt))
        data.root_link_lin_vel_b[0, 1] = 0.06 + 0.20 * ripple
        data.root_link_ang_vel_b[0, 2] = 0.8 + 1.8 * ripple
        if step > 100:
            instantaneous.append(_linear(env))
        lin, ang = _averaged_linear(env), _averaged_yaw(env)
        if step > 100:
            averaged.append(lin)
            yaw.append(ang)
    assert torch.stack(instantaneous).mean(0)[0] < torch.stack(instantaneous).mean(0)[1]
    for samples in (averaged, yaw):
        rewards = torch.stack(samples)
        assert torch.isfinite(rewards).all() and torch.all(rewards <= 0)
        assert rewards.mean(0)[0] > rewards.mean(0)[1] + 0.25


def test_average_isolated_across_resets_and_command_changes_and_not_double_updated():
    env = _env([[0.12, 0.0, 0.8]] * 3)
    command = env.command_manager.get_command("twist")
    data = env.scene["robot"].data
    _averaged_linear(env)
    env.common_step_counter = 1
    env.episode_length_buf[:] = 20
    data.root_link_lin_vel_b[:, 0] = 0.12
    data.root_link_ang_vel_b[:, 2] = 0.8
    # Row 0 starts a new episode; row 1 switches command; row 2 continues.
    env.episode_length_buf[0] = 1
    command[1] = torch.tensor([-0.12, 0.0, -0.8], dtype=command.dtype)
    lin = _averaged_linear(env)
    ang = _averaged_yaw(env)
    assert lin[0] == 0 and ang[0] == 0
    assert lin[1] == -2.0  # New command is immediately charged, no grace period.
    assert ang[1].item() == pytest.approx(-1.55 / 0.8)
    assert -1.0 < lin[2] < -0.9
    # Reading the two reward terms must not advance their shared average twice.
    torch.testing.assert_close(_averaged_linear(env), lin)
    torch.testing.assert_close(_averaged_yaw(env), ang)
    env.common_step_counter += 1
    env.episode_length_buf += 1
    assert _averaged_linear(env)[2] > lin[2]


def test_idle_speed_is_still_penalized_instantaneously():
    env = _env([[0.0, 0.0, 0.0]])
    assert _averaged_linear(env).item() == 0
    env.common_step_counter = 1
    env.episode_length_buf[:] = 20
    env.scene["robot"].data.root_link_lin_vel_b[0, 1] = 0.13
    assert _averaged_linear(env).item() == pytest.approx(-1.0)


@pytest.mark.parametrize("tau", [-0.1, float("nan"), float("inf")])
def test_invalid_averaging_time_constant_rejected(tau):
    with pytest.raises(ValueError, match="tau_s"):
        _averaged_linear(_env([[0.12, 0.0, 0.0]]), tau_s=tau)


def test_only_feedback_adds_positive_weight_self_negating_terms():
    canonical = make_microduck_velocity_env_cfg()
    original_rewards = deepcopy(canonical.rewards)
    feedback = make_microduck_adaptive_velocity_env_cfg(command_exposure=True)
    expected_terms = {
        "linear_velocity_error_l1": (mdp.command_normalized_linear_velocity_l1, FEEDBACK_LINEAR_L1_WEIGHT),
        "yaw_velocity_error_l1": (mdp.command_normalized_yaw_velocity_l1, FEEDBACK_YAW_L1_WEIGHT),
    }
    assert set(feedback.rewards) == set(original_rewards) | set(expected_terms)
    assert {name: feedback.rewards[name] for name in original_rewards} == original_rewards
    env = _env([[0.0, 0.12, 0.8]])
    for name, (func, weight) in expected_terms.items():
        term = feedback.rewards[name]
        assert term.func is func
        assert term.weight == weight > 0
        assert (term.func(env, **term.params) * term.weight).item() < 0
    assert FEEDBACK_LINEAR_MIN_SCALE_M_S >= 0.12
    assert FEEDBACK_YAW_MIN_SCALE_RAD_S >= 0.8
    assert feedback.observations == canonical.observations
    assert feedback.actions == canonical.actions
    assert feedback.events == canonical.events
    assert canonical.rewards == original_rewards
    assert make_microduck_velocity_env_cfg().rewards == original_rewards
    for axis, diagnostic, use_feedback, _ in ADAPTIVE_RECIPES:
        if not use_feedback:
            cfg = make_microduck_adaptive_velocity_env_cfg(axis_mode=axis, diagnostic_mode=diagnostic)
            assert not set(expected_terms).intersection(cfg.rewards)

    # Factory construction and later live-recipe changes must not alias either
    # an existing canonical config or the next config built from that factory.
    feedback.rewards["track_linear_velocity"].weight = 99.0
    feedback.rewards["track_linear_velocity"].params["std"] = 99.0
    assert canonical.rewards == original_rewards
    assert make_microduck_velocity_env_cfg().rewards == original_rewards
    assert make_microduck_adaptive_velocity_env_cfg().rewards == original_rewards
