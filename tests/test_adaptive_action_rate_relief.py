from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks.adaptive_curriculum import AdaptiveActionRateRelief
from mjlab_microduck.tasks.mdp import adaptive_action_rate_l2, reward_weight


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


def test_relief_never_hardens_early_canonical_smoothing() -> None:
    term = SimpleNamespace(weight=-0.1)
    env = SimpleNamespace(
        _adaptive_action_rate_weight=-0.2,
        reward_manager=SimpleNamespace(get_term_cfg=lambda _: term),
        common_step_counter=24,
    )
    reward_weight(env, None, "action_rate_l2", [{"step": 0, "weight": -0.1}])
    assert term.weight == -0.1


def test_legacy_bootstrap_does_not_keep_live_relief_state() -> None:
    relief = AdaptiveActionRateRelief()
    relief.bootstrap({"yaw": 0.0, "turn-left": 0.7, "turn-right": 0.7})
    relief.update({"yaw": 0.0, "turn-left": 0.7, "turn-right": 0.7})
    relief.bootstrap({"yaw": 0.85, "turn-left": 0.85, "turn-right": 0.85})
    assert not relief.active
    assert relief.remaining_windows == 0
    assert relief.triggers == 0


def test_pure_yaw_relief_does_not_trigger_on_a_mixed_turn_deficit() -> None:
    relief = AdaptiveActionRateRelief(scope="pure_yaw")
    relief.bootstrap({"yaw": 0.9, "turn-left": 0.0, "turn-right": 0.0})
    assert not relief.active
    relief.update({"yaw": 0.2, "turn-left": 0.0, "turn-right": 0.0})
    assert relief.active
    restored = AdaptiveActionRateRelief(scope="pure_yaw")
    restored.load_state_dict(relief.state_dict())
    assert restored.state_dict() == relief.state_dict()
    restored.update({"yaw": 0.85, "turn-left": 0.0, "turn-right": 0.0})
    assert not restored.active


def test_legacy_global_relief_cannot_silently_become_pure_yaw() -> None:
    source = AdaptiveActionRateRelief()
    source.bootstrap({"yaw": 0.1, "turn-left": 0.9, "turn-right": 0.9})
    legacy = source.state_dict()
    legacy["version"] = 1
    legacy.pop("scope")
    restored = AdaptiveActionRateRelief()
    restored.load_state_dict(legacy)
    assert restored.state_dict() == source.state_dict()
    with pytest.raises(ValueError, match="scope mismatch"):
        AdaptiveActionRateRelief(scope="pure_yaw").load_state_dict(legacy)


@pytest.mark.parametrize("canonical_weight", [-1.0, -0.1, 0.0])
def test_pure_yaw_relief_preserves_other_command_costs_and_actions(canonical_weight) -> None:
    commands = torch.tensor([
        [0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.0, 0.12, 0.0],
        [0.0, 0.0, 0.8], [0.08, 0.0, 0.8], [0.08, 0.0, -0.8],
        [0.0, 0.0, -0.8],
    ])
    actions = torch.ones(7, 14)
    previous = torch.zeros_like(actions)
    term = SimpleNamespace(weight=canonical_weight)
    env = SimpleNamespace(
        common_step_counter=24,
        action_manager=SimpleNamespace(action=actions, prev_action=previous),
        command_manager=SimpleNamespace(get_command=lambda _: commands),
        reward_manager=SimpleNamespace(get_term_cfg=lambda _: term),
    )
    relief = AdaptiveActionRateRelief(scope="pure_yaw")
    relief.bootstrap({"yaw": 0.0})
    relief.apply(env)
    reward_weight(env, None, "action_rate_l2", [{"step": 0, "weight": canonical_weight}])
    assert term.weight == canonical_weight
    expected = torch.full((7,), 14.0 * canonical_weight)
    expected[[3, 6]] = 14.0 * max(canonical_weight, -0.2)
    torch.testing.assert_close(term.weight * adaptive_action_rate_l2(env), expected)
    torch.testing.assert_close(actions, torch.ones_like(actions))
    torch.testing.assert_close(previous, torch.zeros_like(previous))

    relief.update({"yaw": 0.9})
    relief.apply(env)
    torch.testing.assert_close(
        term.weight * adaptive_action_rate_l2(env),
        torch.full((7,), 14.0 * canonical_weight),
    )
