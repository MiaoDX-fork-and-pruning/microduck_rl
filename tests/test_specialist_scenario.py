import importlib.util
import json
from pathlib import Path
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "specialist_scenario",
    Path(__file__).parents[1] / "scripts" / "specialist_scenario.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
compile_scenario = _MODULE.compile_scenario
scenario_events = _MODULE.scenario_events


def _scenario() -> dict:
    path = Path(__file__).parents[1] / "docs" / "specialist_demo_scenario.json"
    return json.loads(path.read_text())


def _track_b_scenario() -> dict:
    path = Path(__file__).parents[1] / "docs" / "specialist_demo_track_b_scenario.json"
    return json.loads(path.read_text())


def test_canonical_scenario_expands_to_exact_50_hz_schedule():
    frames = compile_scenario(_scenario())
    assert len(frames) == 90 * 50
    assert [frames[index].policy_id for index in (0, 400, 1100, 1500, 1800)] == [
        "velstand_flat", "velocity_flat", "velstand_flat", "sitstand_flat", "sitstand_flat"
    ]
    assert frames[1500].command == (1.0, 0.0, 0.0) + (0.0,) * 10
    assert frames[1800].command == (0.0,) * 13
    assert frames[-1].step == 4499
    assert frames[-1].time_s == 4499 / 50


def test_accepts_full_13d_command_without_remapping():
    scenario = _scenario()
    command = [float(index) for index in range(13)]
    scenario["transitions"][0]["command"] = command
    assert compile_scenario(scenario)[0].command == tuple(command)


def test_events_include_policy_and_same_policy_command_changes():
    events = scenario_events(compile_scenario(_scenario()))
    assert [(event.step, event.policy_id, event.command[0]) for event in events] == [
        (0, "velstand_flat", 0.0),
        (400, "velocity_flat", 0.15),
        (1100, "velstand_flat", 0.0),
        (1500, "sitstand_flat", 1.0),
        (1800, "sitstand_flat", 0.0),
        (2100, "velstand_flat", 0.0),
        (2400, "ground_pick_flat", 0.0),
        (2900, "velstand_flat", 0.0),
        (3200, "ball_kick_flat", 0.0),
        (3500, "velstand_flat", 0.0),
        (3800, "roulade_flat", 0.0),
        (4200, "velstand_flat", 0.0),
    ]


def test_rejects_discontinuous_policy_chain():
    scenario = _scenario()
    scenario["transitions"][1]["from"] = "wrong_policy"
    with pytest.raises(ValueError, match="does not continue previous policy"):
        compile_scenario(scenario)


def test_rejects_switch_time_off_command_grid():
    scenario = _scenario()
    scenario["transitions"][1]["at_s"] = 8.001
    with pytest.raises(ValueError, match="must align to command_rate_hz"):
        compile_scenario(scenario)


def test_requires_integer_seed_and_fixed_50_hz_rate():
    scenario = _scenario()
    scenario["seed"] = 1.5
    with pytest.raises(ValueError, match="seed must be an integer"):
        compile_scenario(scenario)
    scenario = _scenario()
    scenario["command_rate_hz"] = 40
    with pytest.raises(ValueError, match="must be 50"):
        compile_scenario(scenario)


def test_rejects_switch_before_minimum_dwell():
    scenario = _scenario()
    scenario["transitions"][0]["min_dwell_s"] = 9
    with pytest.raises(ValueError, match="switches before min_dwell_s"):
        compile_scenario(scenario)


def test_unsupported_records_are_validated_but_not_compiled():
    scenario = _scenario()
    frames = compile_scenario(scenario)
    unsupported_ids = {item["to"] for item in scenario["unsupported_transitions"]}
    assert unsupported_ids.isdisjoint(frame.policy_id for frame in frames)
    scenario["unsupported_transitions"][0]["at_s"] = 1
    with pytest.raises(ValueError, match="must not contain executable fields"):
        compile_scenario(scenario)


def test_track_b_declares_one_compatible_session_and_is_deterministic():
    scenario = _track_b_scenario()
    assert scenario["compatibility"] == {
        "scene": "microduck_walk_rollers_flat",
        "session": "track_b_rollers_flat",
    }
    first = compile_scenario(scenario)
    second = compile_scenario(scenario)
    assert first == second
    assert len(first) == 60 * 50
    assert set(frame.policy_id for frame in first) == {"velocity_rollers", "roller_crouch"}


def test_requires_structured_measurable_outcome():
    scenario = _scenario()
    scenario["transitions"][0]["expected_outcome"] = "upright"
    with pytest.raises(ValueError, match="expected_outcome must be an object"):
        compile_scenario(scenario)


@pytest.mark.parametrize("command", [[0.0], [0.0] * 12, [0.0] * 14, [0.0, "x", 0.0]])
def test_rejects_invalid_command_blocks(command):
    scenario = _scenario()
    scenario["transitions"][0]["command"] = command
    with pytest.raises(ValueError, match="command"):
        compile_scenario(scenario)
