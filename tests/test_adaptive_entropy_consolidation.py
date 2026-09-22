from copy import deepcopy

import pytest

from mjlab_microduck.evaluation.capability import BUCKETS
from mjlab_microduck.tasks.adaptive_curriculum import EntropyConsolidation
from mjlab_microduck.tasks.adaptive_curriculum import CommandExposure


def acquired():
    return dict(zip(BUCKETS, (0.95, 0.735, 0.796, 0.669, 0.830, 0.812), strict=True))


def active():
    controller = EntropyConsolidation()
    controller.begin(acquired(), entropy_coef=0.01, checkpoint="baseline.pt",
                     completed_iterations=6500, window_updates=250, focus_bucket="yaw")
    return controller


def test_trigger_requires_current_acquisition_and_an_unmastered_frontier():
    controller = EntropyConsolidation()
    assert controller.ready(acquired(), 0.01)
    assert not controller.ready({**acquired(), "yaw": 0.59}, 0.01)
    assert not controller.ready({**acquired(), "zero": 0.79}, 0.01)
    assert not controller.ready(dict.fromkeys(BUCKETS, 0.9), 0.01)
    assert not controller.ready(acquired(), 0.0)


@pytest.mark.parametrize("metrics,retained,phase,reason", [
    (dict.fromkeys(BUCKETS, 0.78), True, "retained", "weakest_capability_improved"),
    (dict.fromkeys(BUCKETS, 0.81), True, "retained", "weakest_capability_improved"),
    (acquired(), True, "rejected", "insufficient_capability_gain"),
    (dict.fromkeys(BUCKETS, 0.90), False, "rejected", "preservation_failed"),
    (None, False, "rejected", "evaluation_failed"),
])
def test_one_window_always_terminates_with_evidence(metrics, retained, phase, reason):
    controller = active()
    with pytest.raises(ValueError, match="has not completed"):
        controller.finish(metrics, gate_retained=retained, completed_iterations=6749)
    controller.finish(metrics, gate_retained=retained, completed_iterations=6750)
    assert controller.phase == phase and controller.reason == reason
    assert controller.terminal
    assert not controller.ready(acquired(), 0.01)
    restored = EntropyConsolidation()
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()


def test_active_resume_keeps_original_deadline():
    controller = active()
    restored = EntropyConsolidation()
    restored.load_state_dict(controller.state_dict())
    assert restored.deadline == 6750
    assert restored.baseline_metrics == acquired()
    assert restored.focus_bucket == "yaw"


def test_consolidation_focus_reallocates_only_one_bounded_slice():
    exposure = CommandExposure(initial_focus="lateral")
    before = exposure.state_dict()
    selected = exposure.consolidation_focus("yaw", acquired())
    after = exposure.state_dict()
    assert selected == "yaw"
    assert before["focus_bucket"] == "lateral"
    assert after["focus_bucket"] == "yaw"
    assert after["windows"] == before["windows"] + 1
    assert after["retention_repairs"] == before["retention_repairs"] == 0
    assert after["probabilities"]["yaw"] > before["probabilities"]["yaw"]
    assert sum(after["probabilities"].values()) == pytest.approx(sum(before["probabilities"].values()))


@pytest.mark.parametrize("terminal", [False, True])
def test_original_consolidation_checkpoints_keep_budget_and_terminal_verdict(terminal):
    original = active()
    if terminal:
        original.finish(acquired(), gate_retained=True, completed_iterations=6750)
    state = original.state_dict()
    state["version"] = 1
    state.pop("focus_bucket")
    restored = EntropyConsolidation()
    restored.load_state_dict(state)
    assert restored.deadline == 6750
    assert restored.terminal == terminal
    assert restored.phase == original.phase
    assert restored.focus_bucket is None


@pytest.mark.parametrize("change", [
    {"phase": "retained"}, {"window_updates": True}, {"original_entropy_coef": None},
    {"baseline_metrics": dict.fromkeys(BUCKETS, 0.1)}, {"window_updates": 0},
])
def test_corrupt_attempt_state_cannot_reset_its_budget(change):
    state = deepcopy(active().state_dict())
    state.update(change)
    with pytest.raises((ValueError, TypeError)):
        EntropyConsolidation().load_state_dict(state)
