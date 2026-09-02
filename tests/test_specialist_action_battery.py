import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "run_specialist_action_battery",
    Path(__file__).parents[1] / "scripts" / "run_specialist_action_battery.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_velocity_battery_has_every_speed_and_both_input_modes():
    cases = _MODULE.command_cases("velocity_flat", smoke=False)
    assert len(cases) == 12
    assert {case["speed"] for case in cases} == set(_MODULE.COMMAND_SPEEDS)
    assert {case["input_mode"] for case in cases} == set(_MODULE.COMMAND_MODES)


def test_smoke_battery_is_one_direct_step_case():
    assert _MODULE.command_cases("velocity_rollers", smoke=True) == [
        {"id": "vx_0.03_direct_step", "speed": 0.03, "input_mode": "direct_step"}
    ]


def test_profiles_keep_walk_all_collisions_and_rollers_separate():
    assert _MODULE.POLICY_PROFILES["velocity_flat"] == "walk_all_collisions"
    assert _MODULE.POLICY_PROFILES["standup_flat"] == "walk_all_collisions"
    assert _MODULE.POLICY_PROFILES["velocity_rollers"] == "rollers"
    assert len(set(_MODULE.PROFILE_SCENES.values())) == 2


def test_command_ema_is_distinct_from_direct_step():
    requested = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    previous = np.zeros(3, dtype=np.float32)
    direct = _MODULE.apply_command_input(requested, previous, "direct_step")
    ema = _MODULE.apply_command_input(requested, previous, "command_ema", alpha=0.1)
    np.testing.assert_allclose(direct, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(ema, [0.02, 0.0, 0.0])
    with pytest.raises(ValueError, match="unsupported"):
        _MODULE.apply_command_input(requested, previous, "scheduler")


def test_failure_classification_is_per_case():
    passed, reason = _MODULE.classify_case(
        "velocity_flat", True, [0.1, 0.0, 0.0], 0.2, 0.1, False
    )
    assert passed and reason is None
    passed, reason = _MODULE.classify_case(
        "velocity_flat", True, [0.0, 0.0, 0.0], 0.2, 0.1, False
    )
    assert not passed and reason == "insufficient_forward_displacement"
    passed, reason = _MODULE.classify_case(
        "standup_flat", True, [0.0, 0.0, 0.0], 1.4, 1.3, False
    )
    assert not passed and reason == "ended_fallen"


def test_low_speed_deadband_may_hold_still_but_higher_speed_must_move():
    low = _MODULE.classify_case(
        "velocity_flat", True, [0.0, 0.0, 0.0], 0.1, 0.1, False, 0.05
    )
    moving = _MODULE.classify_case(
        "velocity_flat", True, [0.0, 0.0, 0.0], 0.1, 0.1, False, 0.08
    )
    assert low == (True, None)
    assert moving == (False, "insufficient_forward_displacement")


def test_sitstand_case_contains_a_commanded_rise():
    case = _MODULE.command_cases("sitstand_flat", smoke=False)[0]
    assert _MODULE.requested_command("sitstand_flat", case, 0, 100)[0] == 1.0
    assert _MODULE.requested_command("sitstand_flat", case, 50, 100)[0] == 0.0


def test_standup_separates_primary_sit_to_stand_from_recovery_probe():
    cases = _MODULE.command_cases("standup_flat", smoke=False)
    assert [case["id"] for case in cases] == ["sit_to_stand", "prone_recovery_probe"]
    assert cases[0]["acceptance"] == "primary"
    assert cases[1]["acceptance"] == "probe"
    assert [case["id"] for case in _MODULE.command_cases("standup_flat", smoke=True)] == ["sit_to_stand"]


def test_every_accepted_specialist_has_an_explicit_profile():
    entries = _MODULE.load_manifest(
        Path(__file__).parents[1] / "artifacts" / "specialist_artifact_manifest.json"
    )
    assert len(entries) == 13
    assert {entry["id"] for entry in entries} == set(_MODULE.POLICY_PROFILES)
