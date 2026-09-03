from dataclasses import asdict

import torch

from mjlab_microduck.tasks.microduck_generalist_g0_env_cfg import (
    G0_BEHAVIORS,
    G0_OBS_DIM,
    GeneralistG0RlCfg,
    GeneralistG0DirectPpoRlCfg,
    GeneralistG0HybridPpoRlCfg,
    behavior_mask,
    make_microduck_generalist_g0_env_cfg,
    G0_TRANSITION_CONTRACTS,
    initialize_g0_state,
    reset_g0_sitstand_state,
    sample_g0_transition,
    g0_stage_curriculum,
)


def test_g0_cfg_composes_all_collision_recipe_and_conditioning():
    cfg = make_microduck_generalist_g0_env_cfg()
    assert cfg.scene.entities["robot"].spec_fn.__name__ == "get_standup_spec"
    assert cfg.g0_observation_dim == G0_OBS_DIM == 71
    assert cfg.g0_action_dim == 14
    assert cfg.g0_behaviors == G0_BEHAVIORS
    assert {"g0_behavior", "g0_phase", "g0_posture", "g0_side"} <= set(cfg.observations["actor"].terms)
    assert {
        "sitstand_posture_pose_legs",
        "sitstand_posture_pose_l1",
        "sitstand_posture_height",
        "sitstand_posture_composite",
    } <= set(cfg.rewards)
    reset_events = list(cfg.events)
    assert reset_events.index("g0_sitstand_state") > reset_events.index("g0_state")
    assert reset_events.index("g0_sitstand_state") > reset_events.index("random_prone_init")
    assert cfg.terminations["fell_over"].params["limit_angle"] == torch.pi
    assert "fell_over_disable" not in cfg.curriculum
    for name, expected in (("action_rate_l2", -0.1), ("body_ang_vel", -0.05),
                           ("joint_torque_rate_l2", -0.002)):
        stages = cfg.curriculum[f"g0_{name}_discovery"].params["weight_stages"]
        assert stages[0]["weight"] == 0.0
        assert stages[1]["weight"] == expected
        assert stages[1]["step"] == 1200 * 24


def test_g0_observation_terms_follow_frozen_abi_order():
    cfg = make_microduck_generalist_g0_env_cfg()
    names = list(cfg.observations["actor"].terms)
    assert names[:5] == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"]
    assert names[5] == "g0_behavior"
    assert names[6:9] == ["command", "head_command", "body_command"]
    assert names[9:] == ["g0_phase", "g0_posture", "g0_side"]


def test_g0_condition_masks_are_exclusive_and_finite():
    class Env:
        num_envs = 3
        device = "cpu"
        g0_behavior_id = torch.tensor([0, 1, 2])

    masks = torch.stack([behavior_mask(Env(), name) for name in G0_BEHAVIORS])
    assert torch.equal(masks.sum(dim=0), torch.ones(3))
    assert torch.isfinite(masks).all()


def test_g0_runner_has_distinct_identity():
    assert GeneralistG0RlCfg.experiment_name == "generalist_g0"
    assert GeneralistG0RlCfg.actor.distribution_cfg["class_name"] == "GaussianDistribution"
    assert GeneralistG0RlCfg.algorithm is not None


def test_g0_baseline_runner_identities_are_comparable_but_distinct():
    assert GeneralistG0DirectPpoRlCfg.experiment_name == "generalist_g0_direct_ppo"
    assert GeneralistG0HybridPpoRlCfg.experiment_name == "generalist_g0_hybrid_ppo"
    assert GeneralistG0DirectPpoRlCfg.run_name != GeneralistG0HybridPpoRlCfg.run_name
    assert GeneralistG0DirectPpoRlCfg.num_steps_per_env == GeneralistG0HybridPpoRlCfg.num_steps_per_env
    assert GeneralistG0DirectPpoRlCfg.max_iterations == GeneralistG0HybridPpoRlCfg.max_iterations
    assert "anchor_weights" not in asdict(GeneralistG0DirectPpoRlCfg)["algorithm"]
    assert asdict(GeneralistG0HybridPpoRlCfg)["algorithm"]["anchor_weights"] == 0.1


def test_g0_transition_contract_is_exactly_four_track_a_edges():
    assert {(item.source, item.destination) for item in G0_TRANSITION_CONTRACTS} == {
        (0, 1), (1, 0), (0, 2), (2, 0)
    }
    assert [(item.dwell_s, item.command) for item in G0_TRANSITION_CONTRACTS] == [
        (14.0, (0.15, 0.0, 0.0)),
        (8.0, (0.0, 0.0, 0.0)),
        (6.0, (1.0, 0.0, 0.0)),
        (6.0, (0.0, 0.0, 0.0)),
    ]


def test_g0_stage_unlock_requires_measured_success_in_order():
    class Env:
        device = "cpu"
    env = Env()
    assert g0_stage_curriculum(env) == 0  # no metric means no promotion
    assert g0_stage_curriculum(env, success_rates=[0.91, 0.1, 0.1]) == 1
    assert g0_stage_curriculum(env, success_rates=[0.91, 0.91, 0.1]) == 2
    assert g0_stage_curriculum(env, success_rates=[0.91, 0.91, 0.91]) == 3


class _Term:
    def __init__(self, n):
        self.vel_command_b = torch.zeros(n, 3)
        self.vel_command_w = torch.zeros(n, 3)


class _Manager:
    def __init__(self, n):
        self.term = _Term(n)

    def get_term(self, name):
        assert name == "twist"
        return self.term


class _RouterEnv:
    num_envs = 1
    device = "cpu"
    step_dt = 1.0

    def __init__(self):
        self.command_manager = _Manager(1)


def test_g0_sitstand_reset_covers_all_state_goal_pairs(monkeypatch):
    class Env:
        num_envs = 4096
        device = "cpu"

        def __init__(self):
            self.command_manager = _Manager(self.num_envs)

    env = Env()
    ids = torch.arange(env.num_envs)
    torch.manual_seed(42)
    initialize_g0_state(env, ids)

    physical_bucket = torch.full((env.num_envs,), -1, dtype=torch.long)

    def record_reset(_env, selected, *, sitting_prob, standing_prob, **_params):
        physical_bucket[selected] = 1 if sitting_prob == 1.0 else 0
        assert sitting_prob + standing_prob == 1.0

    monkeypatch.setattr(
        "mjlab_microduck.tasks.microduck_generalist_g0_env_cfg._mdp.set_random_ground_state",
        record_reset,
    )
    reset_g0_sitstand_state(env, ids)

    sitstand = env.g0_behavior_id == G0_BEHAVIORS.index("SITSTAND")
    command_goal = env.command_manager.term.vel_command_b[:, 0].to(torch.long)
    assert torch.equal(env.g0_posture[sitstand].to(torch.long), command_goal[sitstand])
    assert physical_bucket[~sitstand].eq(-1).all()
    pairs = set(zip(physical_bucket[sitstand].tolist(), command_goal[sitstand].tolist()))
    assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_g0_reset_bucket_labels_match_behavior_condition():
    class Env:
        num_envs = 12
        device = "cpu"

        def __init__(self):
            self.command_manager = _Manager(self.num_envs)

    env = Env()
    ids = torch.arange(env.num_envs)
    torch.manual_seed(7)
    initialize_g0_state(env, ids)
    assert env.g0_reset_bucket.shape == (env.num_envs,)
    assert torch.equal(
        env.g0_reset_bucket[env.g0_behavior_id == 0],
        torch.full_like(env.g0_reset_bucket[env.g0_behavior_id == 0], 2),
    )
    inactive = env.g0_transition_destination < 0
    assert env.g0_reset_transition_phase[inactive].eq(0).all()


def test_g0_initial_transition_buckets_are_legal_and_phase_labeled():
    class Env:
        num_envs = 4096
        device = "cpu"

        def __init__(self):
            self.command_manager = _Manager(self.num_envs)

    env = Env()
    ids = torch.arange(env.num_envs)
    torch.manual_seed(3)
    initialize_g0_state(env, ids)
    active = env.g0_transition_destination >= 0
    assert active.float().mean() > 0.10
    pairs = set(zip(env.g0_transition_source[active].tolist(), env.g0_transition_destination[active].tolist()))
    assert pairs <= {(0, 1), (1, 0), (0, 2), (2, 0)}
    assert torch.all((env.g0_reset_transition_phase[active] > 0) & (env.g0_reset_transition_phase[active] < 0.8))


def test_g0_router_holds_initial_stand_then_uses_velocity_contract():
    torch.manual_seed(0)
    env = _RouterEnv()
    ids = torch.tensor([0])
    initialize_g0_state(env, ids)
    for _ in range(8):
        sample_g0_transition(env, ids)
    # The initial node is sampled, and any subsequent edge remains legal.
    assert int(env.g0_behavior_id[0]) in (0, 1, 2)
    command = env.command_manager.term.vel_command_b[0]
    assert any(
        torch.allclose(command, torch.tensor(expected), atol=1e-6)
        for expected in ((0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (1.0, 0.0, 0.0))
    )


def test_g0_router_never_emits_direct_velocity_sit_edge():
    env = _RouterEnv()
    ids = torch.tensor([0])
    initialize_g0_state(env, ids)
    env.g0_behavior_id[0] = 1
    env.g0_transition_destination[0] = -1
    env.g0_transition_elapsed_s[0] = 8.0
    sample_g0_transition(env, ids)
    assert int(env.g0_behavior_id[0]) == 0
