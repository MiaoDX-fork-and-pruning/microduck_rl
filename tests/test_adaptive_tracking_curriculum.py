from types import SimpleNamespace

import torch
import pytest

from mjlab_microduck.tasks.mdp import velocity_tracking_std_curriculum


def test_tracking_std_curriculum_applies_exact_cumulative_boundaries() -> None:
    term = SimpleNamespace(params={"std": 0.31622776601683794})
    env = SimpleNamespace(
        common_step_counter=0,
        reward_manager=SimpleNamespace(get_term_cfg=lambda _: term),
    )
    stages = [
        {"step": 0, "std": 0.31622776601683794},
        {"step": 12000, "std": 0.22},
        {"step": 24000, "std": 0.16},
    ]

    value = velocity_tracking_std_curriculum(env, torch.empty(0, dtype=torch.long), "track_linear_velocity", stages)
    assert value.item() == pytest.approx(stages[0]["std"])
    assert term.params["std"] == pytest.approx(stages[0]["std"])

    env.common_step_counter = 12000
    value = velocity_tracking_std_curriculum(env, torch.empty(0, dtype=torch.long), "track_linear_velocity", stages)
    assert value.item() == pytest.approx(stages[1]["std"])
    assert term.params["std"] == pytest.approx(stages[1]["std"])

    env.common_step_counter = 24000
    value = velocity_tracking_std_curriculum(env, torch.empty(0, dtype=torch.long), "track_linear_velocity", stages)
    assert value.item() == pytest.approx(stages[2]["std"])
    assert term.params["std"] == pytest.approx(stages[2]["std"])
