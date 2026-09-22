from types import SimpleNamespace

import pytest

from mjlab_microduck.tasks.adaptive_curriculum import AdaptiveActionRateRelief
from mjlab_microduck.tasks.mdp import reward_weight


def test_relief_bootstraps_on_yaw_deficit_and_expires_bounded() -> None:
    relief = AdaptiveActionRateRelief(active_windows=2, cooldown_windows=1)
    metrics = {"yaw": 0.0, "turn-left": 0.75, "turn-right": 0.70}

    relief.bootstrap(metrics)
    assert relief.active is True
    assert relief.remaining_windows == 2
    assert relief.triggers == 1

    relief.update(metrics)
    assert relief.active is True
    assert relief.remaining_windows == 1
    relief.update(metrics)
    assert relief.active is False
    assert relief.cooldown_remaining == 1
    assert relief.last_reason == "bounded_relief_expired"

    relief.update(metrics)
    assert relief.active is False
    assert relief.cooldown_remaining == 0
    relief.update(metrics)
    assert relief.active is True
    assert relief.triggers == 2


def test_relief_releases_when_yaw_frontier_is_mastered() -> None:
    relief = AdaptiveActionRateRelief(active_windows=4)
    relief.bootstrap({"yaw": 0.2, "turn-left": 0.4, "turn-right": 0.3})
    relief.update({"yaw": 0.85, "turn-left": 0.82, "turn-right": 0.81})
    assert relief.active is False
    assert relief.remaining_windows == 0
    assert relief.last_reason == "yaw_frontier_released"


def test_relief_state_round_trip_rejects_different_contract() -> None:
    source = AdaptiveActionRateRelief(relief_weight=-0.25)
    source.bootstrap({"yaw": 0.1, "turn-left": 0.7, "turn-right": 0.6})
    restored = AdaptiveActionRateRelief(relief_weight=-0.25)
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()
    with pytest.raises(ValueError, match="relief_weight mismatch"):
        AdaptiveActionRateRelief(relief_weight=-0.2).load_state_dict(source.state_dict())


def test_reward_weight_preserves_relief_until_runner_clears_override() -> None:
    term = SimpleNamespace(weight=-1.0)
    env = SimpleNamespace(
        _adaptive_action_rate_weight=-0.2,
        reward_manager=SimpleNamespace(get_term_cfg=lambda _: term),
        common_step_counter=999999,
    )
    stages = [{"step": 0, "weight": -0.1}, {"step": 100, "weight": -1.0}]
    reward_weight(env, None, "action_rate_l2", stages)
    assert term.weight == -0.2

    env._adaptive_action_rate_weight = None
    reward_weight(env, None, "action_rate_l2", stages)
    assert term.weight == -1.0
