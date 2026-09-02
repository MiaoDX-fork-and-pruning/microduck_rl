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
    sample_g0_transition,
)


def test_g0_cfg_composes_all_collision_recipe_and_conditioning():
    cfg = make_microduck_generalist_g0_env_cfg()
    assert cfg.scene.entities["robot"].spec_fn.__name__ == "get_standup_spec"
    assert cfg.g0_observation_dim == G0_OBS_DIM == 71
    assert cfg.g0_action_dim == 14
    assert cfg.g0_behaviors == G0_BEHAVIORS
    assert {"g0_behavior", "g0_phase", "g0_posture", "g0_side"} <= set(cfg.observations["actor"].terms)


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


def test_g0_router_holds_initial_stand_then_uses_velocity_contract():
    env = _RouterEnv()
    ids = torch.tensor([0])
    initialize_g0_state(env, ids)
    for _ in range(8):
        sample_g0_transition(env, ids)
    # The first expiry chooses one of the two legal stand edges; either choice
    # must carry its frozen command and expose a live transition phase.
    assert int(env.g0_behavior_id[0]) in (1, 2)
    assert float(env.g0_phase[0, 1]) == 1.0
    command = env.command_manager.term.vel_command_b[0]
    assert any(
        torch.allclose(command, torch.tensor(expected), atol=1e-6)
        for expected in ((0.15, 0.0, 0.0), (1.0, 0.0, 0.0))
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
