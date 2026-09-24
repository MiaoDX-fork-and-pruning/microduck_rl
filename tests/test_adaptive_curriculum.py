from __future__ import annotations

import pytest

from mjlab_microduck.tasks.adaptive_curriculum import (
    ADAPTIVE_AXIS_CONFIGS,
    AxisConfig,
    CapabilityGate,
    apply_stage_to_env,
    GateOutcome,
)


def _gate() -> CapabilityGate:
    return CapabilityGate(
        (AxisConfig("com", (0.003, 0.01, 0.015), 0.8, 0.6, pass_windows=2, fail_windows=2, min_dwell_steps=10),),
        critical_buckets=("zero", "forward", "yaw"),
        ema_alpha=1.0,
    )


def test_advances_after_consecutive_windows_and_logs_trace() -> None:
    gate = _gate()
    assert gate.update(0, {"zero": 0.9, "forward": 0.85, "yaw": 0.82}, checkpoint="p0", seed=7) is None
    transition = gate.update(1, {"zero": 0.9, "forward": 0.85, "yaw": 0.82}, checkpoint="p1", seed=7)
    assert transition is not None
    assert transition.axis == "com"
    assert transition.old_stage == 0 and transition.new_stage == 1
    assert transition.checkpoint == "p1"


def test_dwell_and_deadband_prevent_chatter() -> None:
    gate = _gate()
    gate.update(0, {"zero": 0.9, "forward": 0.9, "yaw": 0.9})
    gate.update(1, {"zero": 0.9, "forward": 0.9, "yaw": 0.9})
    assert gate.update(5, {"zero": 0.2, "forward": 0.2, "yaw": 0.2}) is None
    assert gate.update(10, {"zero": 0.2, "forward": 0.2, "yaw": 0.2}) is None
    assert gate.states["com"].current_stage == 1


def test_preservation_gate_blocks_frontier_when_mastered_bucket_drops() -> None:
    gate = _gate()
    gate.update(0, {"zero": 0.95, "forward": 0.9, "yaw": 0.9})
    gate.update(1, {"zero": 0.95, "forward": 0.9, "yaw": 0.9})
    # The lower-tail score still passes, but the previously mastered zero
    # bucket dropped by more than the 5% preservation tolerance.
    assert gate.update(20, {"zero": 0.7, "forward": 0.9, "yaw": 0.9}) is None
    assert gate.states["com"].current_stage == 1


def test_state_round_trip_reproduces_trace() -> None:
    gate = _gate()
    gate.update(0, {"zero": 0.9, "forward": 0.9, "yaw": 0.9})
    payload = gate.state_dict()
    restored = _gate()
    restored.load_state_dict(payload)
    assert restored.state_dict() == payload


def test_freeze_at_final_moves_all_axes_without_erasing_audit_history() -> None:
    gate = CapabilityGate(
        (
            AxisConfig("com", (0.003, 0.01, 0.015), 0.8, 0.6, pass_windows=1, fail_windows=1),
            AxisConfig("head", (0.003, 0.005), 0.8, 0.6, pass_windows=1, fail_windows=1),
        ),
        critical_buckets=("zero", "forward", "yaw"),
        ema_alpha=1.0,
    )
    gate.update(10, {"zero": 0.9, "forward": 0.9, "yaw": 0.9})
    trace = gate.trace.copy()
    gate.freeze_at_final(step=240)
    assert gate.stage_value("com") == 0.015
    assert gate.stage_value("head") == 0.005
    assert gate.trace == trace
    assert all(state.pass_count == 0 and state.fail_count == 0 for state in gate.states.values())
    assert all(state.last_transition_step == 240 for state in gate.states.values())


def test_freeze_at_final_can_isolate_one_owned_axis() -> None:
    gate = CapabilityGate(
        (
            AxisConfig("com", (0.003, 0.015), 0.8, 0.6),
            AxisConfig("head", (0.003, 0.010), 0.8, 0.6),
        ),
        critical_buckets=("zero", "forward", "yaw"),
    )
    gate.freeze_at_final(step=24, axis_names=("com",))
    assert gate.stage_value("com") == 0.015
    assert gate.stage_value("head") == 0.003
    with pytest.raises(ValueError, match="nonempty owned subset"):
        gate.freeze_at_final(step=24, axis_names=("other",))


def test_missing_or_nonfinite_bucket_is_rejected() -> None:
    gate = _gate()
    with pytest.raises(KeyError):
        gate.update(0, {"zero": 1.0})
    with pytest.raises(ValueError):
        gate.update(0, {"zero": float("nan"), "forward": 1.0, "yaw": 1.0})


def test_decision_api_distinguishes_hold_and_preservation_failure() -> None:
    gate = _gate()
    assert gate.decide(0, {"zero": 0.9, "forward": 0.9, "yaw": 0.9}).outcome == GateOutcome.HOLD
    gate.decide(1, {"zero": 0.9, "forward": 0.9, "yaw": 0.9})
    gate.decide(20, {"zero": 0.8, "forward": 0.9, "yaw": 0.9})
    decision = gate.decide(21, {"zero": 0.8, "forward": 0.9, "yaw": 0.9})
    assert decision.outcome == GateOutcome.PRESERVATION_FAILURE


def test_preservation_failures_identifies_only_previously_mastered_buckets() -> None:
    gate = _gate()
    gate.decide(0, {"zero": 0.9, "forward": 0.9, "yaw": 0.9})
    assert gate.preservation_failures({"zero": 0.7, "forward": 0.9, "yaw": 0.9}) == (
        "zero",
    )
    assert gate.preservation_failures({"zero": 0.86, "forward": 0.9, "yaw": 0.9}) == ()


def test_default_axes_match_canonical_com_endpoints() -> None:
    assert ADAPTIVE_AXIS_CONFIGS[0].name == "com_range"
    assert ADAPTIVE_AXIS_CONFIGS[0].stages == (0.003, 0.005, 0.010, 0.015)
    assert ADAPTIVE_AXIS_CONFIGS[1].stages[-1] == 0.010


def test_stage_binding_updates_live_event_manager_cfg() -> None:
    class EventCfg:
        def __init__(self):
            self.params = {}

    class EventManager:
        def __init__(self):
            self.cfg = EventCfg()

        def get_term_cfg(self, name):
            assert name == "randomize_com"
            return self.cfg

    class Env:
        event_manager = EventManager()

    apply_stage_to_env(Env(), "com_range", 0.01)
    assert Env.event_manager.cfg.params["ranges"] == (-0.01, 0.01)
