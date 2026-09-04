from __future__ import annotations

from scripts.isaaclab.velocity_flat_battery_spec import CASES, SEED, STEPS_PER_CASE, evaluate_case


def test_common_battery_has_fixed_cross_backend_contract() -> None:
    assert SEED == 2026
    assert STEPS_PER_CASE == 300
    assert [case.name for case in CASES] == [
        "zero", "forward", "lateral", "yaw", "turn_left", "turn_right"
    ]


def test_common_battery_requires_signed_response() -> None:
    yaw = next(case for case in CASES if case.name == "turn_left")
    passed, failures = evaluate_case(
        yaw,
        finite=True,
        mean_actual_xy=(0.0, 0.0),
        mean_actual_yaw=0.2,
        max_tilt=0.2,
        reset_fraction=0.0,
    )
    assert not passed
    assert failures == ["command_response"]


def test_common_battery_passes_bounded_directional_response() -> None:
    forward = next(case for case in CASES if case.name == "forward")
    passed, failures = evaluate_case(
        forward,
        finite=True,
        mean_actual_xy=(0.1, 0.0),
        mean_actual_yaw=0.0,
        max_tilt=0.2,
        reset_fraction=0.0,
    )
    assert passed
    assert failures == []
