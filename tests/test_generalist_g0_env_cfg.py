import torch

from mjlab_microduck.tasks.microduck_generalist_g0_env_cfg import (
    G0_BEHAVIORS,
    G0_OBS_DIM,
    GeneralistG0RlCfg,
    behavior_mask,
    make_microduck_generalist_g0_env_cfg,
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
