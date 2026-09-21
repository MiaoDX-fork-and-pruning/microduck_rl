from types import SimpleNamespace

import torch

from mjlab_microduck.tasks.mdp import adaptive_strictification_profile


def _term(weight: float, **params):
    return SimpleNamespace(weight=weight, params=dict(params))


def test_strictification_profile_mutates_live_manager_configs() -> None:
    command = SimpleNamespace(rel_forward_envs=0.2, rel_lateral_envs=0.0)
    rewards = {
        "track_linear_velocity": _term(2.0),
        "track_angular_velocity": _term(2.0),
        "pose": _term(1.0),
        "air_time": _term(3.0),
        "action_rate_l2": _term(-0.2),
    }
    root_height = _term(1.0, min_height=0.055)
    env = SimpleNamespace(
        common_step_counter=0,
        command_manager=SimpleNamespace(get_term_cfg=lambda _: command),
        reward_manager=SimpleNamespace(get_term_cfg=lambda name: rewards[name]),
        termination_manager=SimpleNamespace(get_term_cfg=lambda _: root_height),
    )
    stages = [
        {
            "step": 0,
            "rel_forward_envs": 0.0,
            "rel_lateral_envs": 0.25,
            "track_linear_velocity": 4.0,
            "track_angular_velocity": 6.0,
            "pose": 0.5,
            "air_time": 1.0,
            "root_height": 0.0,
            "action_rate_l2": -0.1,
        },
        {
            "step": 12000,
            "rel_forward_envs": 0.2,
            "rel_lateral_envs": 0.0,
            "track_linear_velocity": 2.0,
            "track_angular_velocity": 2.0,
            "pose": 1.0,
            "air_time": 3.0,
            "root_height": 0.055,
            "action_rate_l2": -0.2,
        },
    ]

    value = adaptive_strictification_profile(env, torch.empty(0, dtype=torch.long), stages)
    assert value.item() == 0.0
    assert (command.rel_forward_envs, command.rel_lateral_envs) == (0.0, 0.25)
    assert [rewards[name].weight for name in rewards] == [4.0, 6.0, 0.5, 1.0, -0.1]
    assert root_height.params["min_height"] == 0.0

    env.common_step_counter = 12000
    value = adaptive_strictification_profile(env, torch.empty(0, dtype=torch.long), stages)
    assert value.item() == 12000.0
    assert (command.rel_forward_envs, command.rel_lateral_envs) == (0.2, 0.0)
    assert [rewards[name].weight for name in rewards] == [2.0, 2.0, 1.0, 3.0, -0.2]
    assert root_height.params["min_height"] == 0.055
