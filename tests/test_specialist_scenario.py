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


def test_canonical_scenario_expands_to_exact_50_hz_schedule():
    frames = compile_scenario(_scenario())
    assert len(frames) == 90 * 50
    assert [frames[index].policy_id for index in (0, 400, 1100, 1500, 1800)] == [
        "stand", "velocity_flat", "stand", "sitstand_flat", "sitstand_flat"
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
        (0, "stand", 0.0),
        (400, "velocity_flat", 0.15),
        (1100, "stand", 0.0),
        (1500, "sitstand_flat", 1.0),
        (1800, "sitstand_flat", 0.0),
        (2100, "stand", 0.0),
        (2400, "ground_pick_flat", 0.0),
        (2900, "stand", 0.0),
        (3200, "ball_kick_flat", 0.0),
        (3500, "stand", 0.0),
        (3800, "roulade_flat", 0.0),
        (4200, "stand", 0.0),
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


@pytest.mark.parametrize("command", [[0.0], [0.0] * 12, [0.0] * 14, [0.0, "x", 0.0]])
def test_rejects_invalid_command_blocks(command):
    scenario = _scenario()
    scenario["transitions"][0]["command"] = command
    with pytest.raises(ValueError, match="command"):
        compile_scenario(scenario)
