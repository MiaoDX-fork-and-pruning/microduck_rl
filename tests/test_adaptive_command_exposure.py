"""Feedback must change actual command exposure without losing anchor coverage."""
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.evaluation.capability import BUCKETS
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.adaptive_curriculum import CommandExposure
from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import make_microduck_adaptive_velocity_env_cfg


def _command(n=30000):
    cfg = make_microduck_adaptive_velocity_env_cfg(command_exposure=True).commands["twist"]
    cfg.heading_command = False  # Unit sampler has no robot heading tensor.
    term = object.__new__(mdp.AdaptiveVelocityCommand)
    term.cfg = cfg
    term._env = SimpleNamespace(device="cpu", num_envs=n)
    term.vel_command_b = torch.full((n, 3), 9.0)
    term.vel_command_w = term.vel_command_b.clone()
    for name in ("is_standing_env", "is_world_env", "is_heading_env", "is_forward_env"):
        setattr(term, name, torch.zeros(n, dtype=torch.bool))
    return term


def _masks(command):
    x, y, yaw = command.unbind(1)
    return {
        "zero": (x == 0) & (y == 0) & (yaw == 0),
        "forward": (x > 0) & (y == 0) & (yaw == 0),
        "lateral": (x == 0) & (y != 0) & (yaw == 0),
        "yaw": (x == 0) & (y == 0) & (yaw != 0),
        "turn-left": (x > 0) & (y == 0) & (yaw > 0),
        "turn-right": (x > 0) & (y == 0) & (yaw < 0),
        "nominal": (x != 0) & (y != 0) & (yaw != 0),
    }


def _transition_term(targets: torch.Tensor) -> mdp.AdaptiveVelocityCommand:
    """Build the command term around its transition state machine only."""
    cfg = make_microduck_adaptive_velocity_env_cfg(
        command_exposure=True,
        transition_probability=0.4,
        transition_bootstrap_mode="zero",
    ).commands["twist"]
    cfg.heading_command = False
    cfg.transition_probability = 1.0
    cfg.transition_duration_s = (1.0, 1.0)
    cfg.transition_forward_fraction = (0.5, 0.5)
    n = targets.shape[0]
    term = object.__new__(mdp.AdaptiveVelocityCommand)
    term.cfg = cfg
    term._env = SimpleNamespace(device="cpu", num_envs=n, common_step_counter=0)
    term.vel_command_b = targets.clone()
    term.vel_command_w = targets.clone()
    for name in ("is_standing_env", "is_world_env", "is_heading_env", "is_forward_env"):
        setattr(term, name, torch.zeros(n, dtype=torch.bool))
    term.bucket_ids = torch.full((n,), -1, dtype=torch.long)
    term.time_left = torch.full((n,), 10.0)
    term._update_metrics = lambda: None
    return term


def _start_zero_transition(term: mdp.AdaptiveVelocityCommand) -> None:
    ids = torch.arange(term.num_envs)
    buckets = torch.tensor([3 + (idx % 3) for idx in range(term.num_envs)])
    term._set_transition_state(ids, buckets)


def test_failed_lateral_gets_more_real_samples_and_retains_nominal_and_anchor_floors():
    term = _command()
    exposure = CommandExposure()
    assert exposure.focus_bucket == "forward"
    assert exposure.probabilities == pytest.approx(
        {name: (0.28 if name == "forward" else 0.20 if name == "zero" else 0.08) for name in BUCKETS}
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
    ids = torch.arange(term.num_envs)
    torch.manual_seed(17)
    term._resample_command(ids)
    initial = _masks(term.command)["lateral"].float().mean()
    scores = {bucket: (0.0 if bucket == "lateral" else 1.0) for bucket in BUCKETS}
    for _ in range(20):
        before = exposure.probabilities.copy()
        exposure.update(scores)
        assert max(abs(exposure.probabilities[b] - before[b]) for b in BUCKETS) <= 0.050001
    exposure.apply(env)
    assert exposure.focus_bucket == "lateral"
    assert exposure.probabilities["lateral"] > exposure.probabilities["forward"]
    assert exposure.probabilities["zero"] >= 0.20
    assert all(value >= 0.08 for name, value in exposure.probabilities.items() if name != "zero")
    torch.manual_seed(17)
    term._resample_command(ids)
    term._update_command()
    masks = _masks(term.command)
    assert term.bucket_ids.shape == (term.num_envs,)
    assert torch.all((term.bucket_ids >= 0) & (term.bucket_ids < len(BUCKETS) + 1))
    assert torch.equal(term.bucket_ids == 0, masks["zero"])
    assert torch.equal(term.bucket_ids == 1, masks["forward"])
    assert torch.equal(term.bucket_ids == 2, masks["lateral"])
    assert torch.equal(term.bucket_ids == 3, masks["yaw"])
    assert torch.equal(term.bucket_ids == 4, masks["turn-left"])
    assert torch.equal(term.bucket_ids == 5, masks["turn-right"])
    assert torch.equal(term.bucket_ids == 6, masks["nominal"])
    assert masks["lateral"].float().mean() > initial + 0.12
    assert masks["zero"].float().mean() >= 0.18
    for bucket in BUCKETS:
        if bucket != "zero":
            assert masks[bucket].float().mean() >= 0.07
    assert masks["nominal"].float().mean() == pytest.approx(0.20, abs=0.015)
    assert torch.any(term.command[masks["lateral"], 1] < 0)
    assert torch.any(term.command[masks["lateral"], 1] > 0)
    assert torch.any(term.command[masks["nominal"], 0] < 0)
    assert torch.equal(term.is_standing_env, masks["zero"])
    assert torch.equal(term.vel_command_w, term.vel_command_b)


def test_zero_transition_slews_between_bootstrap_and_target_and_finishes_exactly():
    target = torch.tensor([[0.0, 0.0, 0.8], [0.22, 0.0, -0.8]])
    term = _transition_term(target)
    _start_zero_transition(term)

    # A reset-time command update must expose the exact zero bootstrap and must
    # not consume transition time, even when dt is explicitly zero.
    term.compute(0.0)
    torch.testing.assert_close(term.vel_command_b, torch.zeros_like(target))
    assert torch.all(term._transition_elapsed == 0.0)
    assert torch.equal(term.bucket_ids, torch.zeros(term.num_envs, dtype=torch.long))

    # The command is continuous at the half-way point, rather than jumping
    # directly from zero to the sampled yaw/turn target.
    term._env.common_step_counter = 1
    term.compute(0.5)
    torch.testing.assert_close(term.vel_command_b, target * 0.5)
    assert torch.all(term._transition_active)
    assert torch.equal(term.bucket_ids, torch.zeros(term.num_envs, dtype=torch.long))

    # Completion writes the original target exactly and transfers attribution
    # to the target bucket.
    term.compute(0.5)
    torch.testing.assert_close(term.vel_command_b, target, rtol=0.0, atol=0.0)
    torch.testing.assert_close(term.vel_command_w, target, rtol=0.0, atol=0.0)
    assert not torch.any(term._transition_active)
    assert torch.equal(term.bucket_ids, torch.tensor([3, 4]))


def test_transition_partial_reset_does_not_mutate_other_env_state():
    target = torch.tensor([[0.0, 0.0, 0.8], [0.22, 0.0, -0.8], [0.18, 0.0, 0.8]])
    term = _transition_term(target)
    _start_zero_transition(term)
    term._transition_elapsed[1] = 0.25
    saved_target = term._transition_target[1].clone()
    saved_bootstrap = term._transition_bootstrap[1].clone()
    saved_bucket = term._transition_target_bucket[1].clone()

    # Resample only env 0 with a new target. Env 1 remains mid-transition and
    # its timer, target, bootstrap, and attribution stay untouched.
    term.vel_command_b[0] = torch.tensor([0.0, 0.0, -0.8])
    term._set_transition_state(torch.tensor([0]), torch.tensor([3]))
    assert term._transition_active[1]
    assert term._transition_elapsed[1].item() == pytest.approx(0.25)
    torch.testing.assert_close(term._transition_target[1], saved_target)
    torch.testing.assert_close(term._transition_bootstrap[1], saved_bootstrap)
    assert term._transition_target_bucket[1] == saved_bucket
    assert term._transition_active[0]
    assert term._transition_target[0, 2].item() == pytest.approx(-0.8)


def test_acquisition_profile_can_start_on_lateral_frontier():
    exposure = CommandExposure(
        initial_focus="lateral",
        frontier_order=("lateral", "forward", "yaw", "turn-left", "turn-right"),
    )
    assert exposure.focus_bucket == "lateral"
    assert exposure.probabilities["lateral"] == pytest.approx(0.28)
    assert exposure.probabilities["zero"] == pytest.approx(0.20)
    assert sum(exposure.probabilities.values()) == pytest.approx(0.80)


def test_frontier_order_is_checkpointed_and_controls_focus_priority():
    order = ("lateral", "forward", "yaw", "turn-left", "turn-right")
    exposure = CommandExposure(frontier_order=order)
    exposure.update({name: 0.0 for name in BUCKETS})
    assert exposure.focus_bucket == "lateral"
    restored = CommandExposure(frontier_order=order)
    restored.load_state_dict(exposure.state_dict())
    assert restored.state_dict() == exposure.state_dict()
    with pytest.raises(ValueError, match="frontier order"):
        CommandExposure().load_state_dict(exposure.state_dict())


def test_invalid_initial_focus_is_rejected():
    with pytest.raises(ValueError, match="unsupported frontier bucket"):
        CommandExposure(initial_focus="sideways")


def test_subset_reset_does_not_change_other_commands_and_survives_update():
    term = _command(n=256)
    ids = torch.arange(0, term.num_envs, 2)
    term._resample_command(ids)
    commands = term.command.clone()
    term._update_command()
    assert torch.equal(term.command, commands)
    assert torch.all(term.command[1::2] == 9)


def test_feedback_state_and_rng_reproduce_sampling_after_resume():
    exposure = CommandExposure()
    exposure.update({name: (0.0 if name == "yaw" else 0.9) for name in BUCKETS})
    restored = CommandExposure()
    restored.load_state_dict(exposure.state_dict())
    scores = {name: (0.0 if name == "forward" else 0.8) for name in BUCKETS}
    exposure.update(scores)
    restored.update(scores)
    assert restored.state_dict() == exposure.state_dict()
    a, b = _command(512), _command(512)
    a.cfg.bucket_probabilities = tuple(exposure.probabilities.values())
    b.cfg.bucket_probabilities = tuple(restored.probabilities.values())
    torch.manual_seed(901)
    state = torch.get_rng_state()
    a._resample_command(torch.arange(512))
    torch.set_rng_state(state)
    b._resample_command(torch.arange(512))
    assert torch.equal(a.command, b.command)


def test_focus_switch_preserves_every_anchor_bucket():
    exposure = CommandExposure()
    exposure.update({name: (0.9 if name != "lateral" else 0.1) for name in BUCKETS})
    assert exposure.focus_bucket == "lateral"
    assert exposure.probabilities["lateral"] > exposure.probabilities["yaw"]
    assert exposure.probabilities["zero"] >= 0.20
    assert all(value >= 0.08 for name, value in exposure.probabilities.items() if name != "zero")
    assert sum(exposure.probabilities.values()) == pytest.approx(0.80)


def test_near_pass_bucket_keeps_frontier_focus_until_gate_threshold():
    exposure = CommandExposure()
    scores = {name: 1.0 for name in BUCKETS}
    scores["forward"] = 0.79
    scores["lateral"] = 0.10
    exposure.update(scores)
    assert exposure.focus_bucket == "forward"
    assert exposure.probabilities["forward"] > exposure.probabilities["lateral"]


def test_configured_frontier_dwell_moves_to_lowest_unmastered_bucket():
    exposure = CommandExposure(
        initial_focus="lateral",
        frontier_order=("lateral", "forward", "yaw", "turn-left", "turn-right"),
        stall_windows=2,
        stall_improvement=0.05,
    )
    scores = {name: 0.9 for name in BUCKETS}
    scores.update(lateral=0.2, yaw=0.0, **{"turn-left": 0.0, "turn-right": 0.0})
    exposure.update(scores)
    assert exposure.focus_bucket == "lateral"
    exposure.update(scores)
    assert exposure.focus_bucket == "lateral"
    exposure.update(scores)
    assert exposure.focus_bucket == "yaw"
    assert exposure.probabilities["yaw"] > 0.08
    assert exposure.probabilities["lateral"] < 0.28
    exposure.update(scores)
    assert exposure.focus_bucket == "yaw"


def test_large_capability_gap_preempts_dwell_and_survives_resume_with_real_sampling():
    # Retained seed-17 state after 6500 updates: yaw collapsed below mastery
    # while lateral kept the focus slice. No mastered-yaw rollback can fire.
    exposure = CommandExposure(
        initial_focus="lateral",
        frontier_order=("lateral", "forward", "yaw", "turn-left", "turn-right"),
        stall_windows=4,
    )
    payload = exposure.state_dict()
    payload.update(
        probabilities={
            "zero": 0.20, "forward": 0.08, "lateral": 0.2593683976351299,
            "yaw": 0.08900390442022896, "turn-left": 0.09162769794464111,
            "turn-right": 0.08,
        },
        windows=34, focus_best_score=0.7959980678393459, focus_stall_count=1,
        retention_repairs=5, last_repair_buckets=["yaw", "turn-left"],
    )
    exposure.load_state_dict(payload)
    scores = {
        "zero": 0.9773, "forward": 0.7374, "lateral": 0.7844,
        "yaw": 0.2485, "turn-left": 0.8356, "turn-right": 0.8147,
    }
    initial = _command()
    initial.cfg.bucket_probabilities = tuple(exposure.probabilities[b] for b in BUCKETS)
    torch.manual_seed(17)
    initial._resample_command(torch.arange(initial.num_envs))
    before = exposure.probabilities.copy()
    exposure.update(scores)
    assert exposure.focus_bucket == "yaw"
    assert exposure.focus_stall_count == 0
    assert exposure.probabilities["yaw"] > before["yaw"]
    assert exposure.probabilities["lateral"] < before["lateral"]
    assert max(abs(exposure.probabilities[b] - before[b]) for b in BUCKETS) <= 0.05
    assert exposure.probabilities["zero"] == pytest.approx(0.20)
    assert min(exposure.probabilities.values()) >= 0.08
    assert sum(exposure.probabilities.values()) == pytest.approx(0.80)
    # This is acquisition feedback, not invented mastery or a retention repair.
    assert exposure.retention_repairs == 5

    restored = CommandExposure(frontier_order=exposure.frontier_order, stall_windows=4)
    restored.load_state_dict(exposure.state_dict())
    assert restored.state_dict() == exposure.state_dict()
    sampled = _command()
    restored.apply(SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: sampled)))
    torch.manual_seed(17)
    sampled._resample_command(torch.arange(sampled.num_envs))
    initial_yaw = (initial.bucket_ids == BUCKETS.index("yaw")).float().mean()
    updated_yaw = (sampled.bucket_ids == BUCKETS.index("yaw")).float().mean()
    assert updated_yaw > initial_yaw + 0.03
    assert (sampled.bucket_ids == len(BUCKETS)).float().mean() == pytest.approx(0.20, abs=0.015)
    # Partial recovery must not immediately hand focus back to the first
    # unmastered bucket; continuing and restarting follow the same decisions.
    scores["yaw"] = 0.60
    exposure.update(scores)
    restored.update(scores)
    assert restored.state_dict() == exposure.state_dict()
    assert restored.focus_bucket == "yaw"


@pytest.mark.parametrize("weak_score", [0.54, 0.60, 0.78])
def test_small_capability_gap_does_not_interrupt_frontier_consolidation(weak_score):
    exposure = CommandExposure(initial_focus="lateral", stall_windows=4)
    scores = {name: 0.9 for name in BUCKETS}
    scores.update(lateral=0.7844, yaw=weak_score)
    exposure.update(scores)
    exposure.update(scores)
    assert exposure.focus_bucket == "lateral"
    assert exposure.focus_stall_count == 1


def test_large_gap_selects_weakest_bucket_with_deterministic_ties():
    exposure = CommandExposure(initial_focus="lateral", stall_windows=4)
    scores = {name: 0.9 for name in BUCKETS}
    scores.update(lateral=0.79, yaw=0.20, **{"turn-left": 0.10, "turn-right": 0.10})
    exposure.update(scores)
    assert exposure.focus_bucket == "turn-left"


def test_checkpoint_requires_and_restores_focus_bucket():
    exposure = CommandExposure()
    exposure.update({name: (0.9 if name != "yaw" else 0.1) for name in BUCKETS})
    payload = exposure.state_dict()
    restored = CommandExposure()
    restored.load_state_dict(payload)
    assert restored.state_dict() == payload
    payload["focus_bucket"] = "unknown"
    with pytest.raises(ValueError, match="focus bucket"):
        restored.load_state_dict(payload)


def test_failed_lateral_never_uses_three_percent_retention_floor():
    exposure = CommandExposure()
    for _ in range(8):
        exposure.update({name: (0.0 if name == "lateral" else 1.0) for name in BUCKETS})
    assert exposure.focus_bucket == "lateral"
    assert min(exposure.probabilities[name] for name in BUCKETS if name != "zero") >= 0.08
    assert exposure.probabilities["zero"] == pytest.approx(0.20)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_feedback_does_not_partially_mutate_state(bad):
    exposure = CommandExposure()
    before = exposure.state_dict()
    with pytest.raises(ValueError):
        exposure.update({name: bad if name == "turn-right" else 0.9 for name in BUCKETS})
    assert exposure.state_dict() == before


def test_equal_deficits_keep_balanced_samples():
    exposure = CommandExposure()
    before = exposure.probabilities.copy()
    exposure.update(dict.fromkeys(BUCKETS, 0.0))
    assert exposure.probabilities == pytest.approx(before)


def test_retention_repair_reallocates_only_bounded_focus_slice_and_keeps_anchors():
    exposure = CommandExposure(
        initial_focus="lateral",
        frontier_order=("lateral", "forward", "yaw", "turn-left", "turn-right"),
    )
    before = exposure.state_dict()
    repaired = exposure.repair(
        ("turn-right",),
        {name: (0.55 if name == "turn-right" else 0.9) for name in BUCKETS},
        feedback={
            "sample_count": {name: 100 for name in BUCKETS},
            "weighted_reward_mass": {
                name: {
                    "track_linear_velocity": 1.0,
                    "track_angular_velocity": 1.0,
                    "upright": 1.0,
                    "yaw_velocity_error_l1": -2.0 if name == "turn-right" else 0.0,
                }
                for name in BUCKETS
            },
        },
    )
    assert repaired == ("turn-right",)
    assert exposure.focus_bucket == "turn-right"
    assert exposure.probabilities["turn-right"] > before["probabilities"]["turn-right"]
    assert exposure.probabilities["zero"] == pytest.approx(0.20)
    assert sum(exposure.probabilities.values()) == pytest.approx(0.80)
    assert exposure.retention_repairs == 1
    restored = CommandExposure(frontier_order=exposure.frontier_order)
    restored.load_state_dict(exposure.state_dict())
    assert restored.state_dict() == exposure.state_dict()


def test_retention_repair_uses_reward_mass_to_rank_multiple_regressions():
    exposure = CommandExposure()
    repaired = exposure.repair(
        ("forward", "yaw"),
        {name: (0.50 if name in ("forward", "yaw") else 0.9) for name in BUCKETS},
        feedback={
            "sample_count": {name: 100 for name in BUCKETS},
            "weighted_reward_mass": {
                name: {
                    "track_linear_velocity": 1.0,
                    "track_angular_velocity": 1.0,
                    "upright": 1.0,
                    "linear_velocity_error_l1": -3.0 if name == "forward" else 0.0,
                }
                for name in BUCKETS
            },
        },
    )
    assert repaired == ("forward", "yaw")
    assert exposure.probabilities["forward"] == pytest.approx(0.255)
    assert exposure.probabilities["yaw"] == pytest.approx(0.105)
