from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.adaptive_curriculum import TransitionExposure
from mjlab_microduck.tasks.adaptive_runner import AdaptiveMicroduckOnPolicyRunner
from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    make_microduck_adaptive_velocity_env_cfg,
)


def _term(*, probability=0.0, mode="forward", num_envs=64):
    cfg = make_microduck_adaptive_velocity_env_cfg(
        command_exposure=True, transition_probability=probability
    ).commands["twist"]
    cfg.heading_command = False
    cfg.bucket_probabilities = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    cfg.transition_duration_s = (1.0, 1.0)
    cfg.transition_forward_fraction = (0.5, 0.5)
    cfg.transition_bootstrap_mode = mode
    env = SimpleNamespace(
        device="cpu", num_envs=num_envs, common_step_counter=0, step_dt=0.02
    )
    term = object.__new__(mdp.AdaptiveVelocityCommand)
    term.cfg = cfg
    term._env = env
    term.vel_command_b = torch.full((num_envs, 3), 9.0)
    term.vel_command_w = term.vel_command_b.clone()
    term.time_left = torch.full((num_envs,), 10.0)
    term.metrics = {"error_vel_xy": torch.zeros(num_envs), "error_vel_yaw": torch.zeros(num_envs)}
    term.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(num_envs, 3),
            root_link_ang_vel_b=torch.zeros(num_envs, 3),
            heading_w=torch.zeros(num_envs),
        )
    )
    for name in ("is_standing_env", "is_world_env", "is_heading_env", "is_forward_env"):
        setattr(term, name, torch.zeros(num_envs, dtype=torch.bool))
    return term


def test_transition_zero_is_exactly_legacy_command_stream():
    term = _term(probability=0.0)
    ids = torch.arange(term.num_envs)
    term._resample_command(ids)
    assert torch.all(term.command[:, :2] == 0.0)
    assert torch.all(term.command[:, 2].abs() > 0.0)
    assert not torch.any(term._transition_active)


def test_zero_to_yaw_uses_forward_bootstrap_then_restores_target():
    term = _term(probability=0.4)
    ids = torch.arange(term.num_envs)
    torch.manual_seed(7)
    term._resample_command(ids)
    active = term._transition_active.clone()
    target = term._transition_target.clone()
    assert torch.any(active)
    assert torch.all(term.command[active, 0] > 0.0)
    assert torch.all(term.command[active, 2] == 0.0)
    assert torch.all(term.bucket_ids[active] == 1)
    term._transition_elapsed[active] = term._transition_duration[active]
    term._update_command()
    assert not torch.any(term._transition_active)
    torch.testing.assert_close(term.command, target)
    assert torch.all(term.bucket_ids == 3)


def test_zero_bootstrap_holds_idle_command_then_restores_yaw_target():
    term = _term(probability=0.4, mode="zero")
    ids = torch.arange(term.num_envs)
    torch.manual_seed(7)
    term._resample_command(ids)
    active = term._transition_active.clone()
    target = term._transition_target.clone()
    assert torch.any(active)
    assert torch.all(term.command[active] == 0.0)
    assert torch.all(term.bucket_ids[active] == 0)
    assert torch.all(term.is_standing_env[active])
    term._transition_elapsed[active] = term._transition_duration[active]
    term._update_command()
    assert not torch.any(term._transition_active)
    torch.testing.assert_close(term.command, target)
    assert torch.all(term.bucket_ids == 3)
    assert not torch.any(term.is_standing_env[active])
    term._env.common_step_counter = 1
    term.compute(0.02)
    torch.testing.assert_close(term.command[active], target[active])


def test_reset_compute_does_not_advance_transition_timer():
    term = _term(probability=0.4)
    term._resample_command(torch.arange(term.num_envs))
    active = term._transition_active.clone()
    term.compute(0.0)
    assert torch.all(term._transition_elapsed == 0.0)
    term._env.common_step_counter = 1
    term.compute(0.02)
    torch.testing.assert_close(
        term._transition_elapsed[active], torch.full((int(active.sum()),), 0.02)
    )


def test_partial_transition_start_leaves_other_environments_untouched():
    term = _term(probability=0.4, num_envs=64)
    term._resample_command(torch.arange(term.num_envs))
    term._transition_active[2] = True
    term._transition_elapsed[2] = 0.7
    before = term._transition_elapsed.clone()
    term._set_transition_state(torch.tensor([0]), torch.tensor([3]))
    assert term._transition_active[2]
    assert term._transition_elapsed[2] == pytest.approx(float(before[2]))


def test_transition_exposure_is_bounded_adaptive_and_checkpointable():
    exposure = TransitionExposure(0.20)
    exposure.update({"yaw": 0.0, "turn-left": 0.7, "turn-right": 0.9})
    assert exposure.probability == pytest.approx(0.25)
    payload = exposure.state_dict()
    restored = TransitionExposure()
    restored.load_state_dict(payload)
    assert restored.state_dict() == payload
    assert exposure.repair(("yaw",))
    assert exposure.probability == pytest.approx(0.30)
    for _ in range(20):
        exposure.update({"yaw": 0.95, "turn-left": 0.95, "turn-right": 0.95})
    assert 0.0 <= exposure.probability <= 0.40


def test_transition_exposure_rejects_invalid_state():
    with pytest.raises(ValueError):
        TransitionExposure(0.41)
    exposure = TransitionExposure()
    with pytest.raises(ValueError):
        exposure.load_state_dict({"version": 1, "probability": 0.5, "windows": 0, "repairs": 0})


def test_config_is_zero_by_default_and_transition_is_opt_in():
    base = make_microduck_adaptive_velocity_env_cfg()
    assert not base.adaptive_transition_acquisition
    assert base.adaptive_transition_probability == 0.0
    assert not hasattr(base.commands["twist"], "transition_probability")
    feedback = make_microduck_adaptive_velocity_env_cfg(
        command_exposure=True, transition_probability=0.20
    )
    assert feedback.adaptive_transition_acquisition
    assert feedback.adaptive_transition_probability == pytest.approx(0.20)
    assert feedback.commands["twist"].transition_probability == pytest.approx(0.20)
    assert feedback.adaptive_transition_bootstrap_mode == "forward"
    zero = make_microduck_adaptive_velocity_env_cfg(
        command_exposure=True, transition_probability=0.0
    )
    assert zero.adaptive_transition_acquisition
    assert zero.commands["twist"].transition_probability == 0.0
    zero_mode = make_microduck_adaptive_velocity_env_cfg(
        command_exposure=True, transition_probability=0.20,
        transition_bootstrap_mode="zero",
    )
    assert zero_mode.adaptive_transition_bootstrap_mode == "zero"
    assert zero_mode.adaptive_transition_bootstrap_mode_override
    with pytest.raises(ValueError, match="command_exposure"):
        make_microduck_adaptive_velocity_env_cfg(transition_probability=0.20)


def test_runner_checkpoint_payload_persists_transition_exposure():
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.env = SimpleNamespace(
        num_envs=64,
        cfg=SimpleNamespace(task_id="fake", adaptive_evaluator_schema_version=2),
    )
    runner.cfg = {"num_steps_per_env": 24}
    runner.current_learning_iteration = 4
    runner.completed_iterations = 5
    runner.capability_gate = None
    runner.command_exposure = None
    runner.transition_exposure = TransitionExposure(0.30)
    runner.final_com_fraction = 0.0
    runner.env.cfg.adaptive_sensor_reset_fraction = 0.0
    runner.bucket_feedback = None
    runner.last_known_good_checkpoint = None
    runner.last_known_good_buckets = ()
    runner.evaluation_events = []
    runner.last_evaluation_provenance = None
    payload = runner.adaptive_checkpoint_info()["adaptive_curriculum"]
    restored = TransitionExposure()
    restored.load_state_dict(payload["transition_exposure"])
    assert restored.state_dict() == runner.transition_exposure.state_dict()
    assert payload["sensor_reset_fraction"] == 0.0
