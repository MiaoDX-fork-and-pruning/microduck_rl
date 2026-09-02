import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "infer_policy",
    Path(__file__).parents[1] / "scripts" / "infer_policy.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Session:
    def __init__(self, name):
        self.name = name

    def get_inputs(self):
        return [SimpleNamespace(name=f"{self.name}_input")]

    def get_outputs(self):
        return [SimpleNamespace(name=f"{self.name}_output")]


def _policy():
    policy = _MODULE.PolicyInference.__new__(_MODULE.PolicyInference)
    policy.new_cmd_obs = True
    policy.standing_session = _Session("stand")
    policy.walking_session = _Session("walk")
    policy.sit_session = _Session("sitstand")
    policy.is_sitstand = True
    policy.ground_pick_session = _Session("ground_pick")
    policy.behavior_sessions = {
        "kick_right": _Session("kick_right"),
        "roulade": _Session("roulade"),
    }
    policy.ground_pick_mode = False
    policy.ground_pick_phase = 0.0
    policy.behavior_mode = None
    policy.sit_mode = False
    policy.slope_mode = False
    policy.vel_cmd = np.zeros(3, dtype=np.float32)
    policy.head_offset = np.zeros(4, dtype=np.float32)
    policy.body_cmd = np.zeros(6, dtype=np.float32)
    policy.command = np.zeros(13, dtype=np.float32)
    policy._place_ball = lambda behavior: setattr(policy, "placed_ball", behavior)
    return policy


def _command(twist=(0.0, 0.0, 0.0)):
    return tuple(twist) + (0.0,) * 10


def test_routes_canonical_track_a_policy_and_command_events():
    policy = _policy()
    policy.activate_specialist_policy("stand", _command())
    assert policy.current_policy == "standing"

    policy.activate_specialist_policy("velocity_flat", _command((0.15, 0.0, 0.0)))
    assert policy.current_policy == "walking"
    np.testing.assert_allclose(policy.command[:3], [0.15, 0.0, 0.0])

    policy.activate_specialist_policy("sitstand_flat", _command((1.0, 0.0, 0.0)))
    assert policy.sit_mode is True and policy.command[0] == 1.0
    policy.activate_specialist_policy("sitstand_flat", _command())
    assert policy.sit_mode is False and policy.command[0] == 0.0

    policy.activate_specialist_policy("ground_pick_flat", _command())
    assert policy.ground_pick_mode is True
    policy.activate_specialist_policy("ball_kick_flat", _command())
    assert policy.behavior_mode == "kick_right" and policy.placed_ball == "kick_right"
    policy.activate_specialist_policy("roulade_flat", _command())
    assert policy.behavior_mode == "roulade"


def test_router_rejects_missing_or_unknown_policy_sessions():
    policy = _policy()
    policy.standing_session = None
    with pytest.raises(ValueError, match="was not loaded"):
        policy.activate_specialist_policy("stand", _command())
    with pytest.raises(ValueError, match="unsupported"):
        policy.activate_specialist_policy("spin", _command())


def test_preflight_checks_every_scheduled_policy_before_simulation():
    policy = _policy()
    policy.behavior_sessions.pop("roulade")
    with pytest.raises(ValueError, match="roulade_flat"):
        policy.validate_specialist_policies(
            policy_id for policy_id in ["stand", "velocity_flat", "roulade_flat"])
    with pytest.raises(ValueError, match="spin"):
        policy.validate_specialist_policies(["stand", "spin"])


def test_router_rejects_non_13d_commands():
    with pytest.raises(ValueError, match="finite 13D"):
        _policy().activate_specialist_policy("stand", (0.0, 0.0, 0.0))
