from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import torch

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


def test_battery_freezes_complete_command_block_after_reset() -> None:
    # The executable harness imports IsaacLab's AppLauncher at module import
    # time.  Stub only that boundary so this test can exercise the deterministic
    # command writer on CPU without launching Isaac Sim.
    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = object
    isaaclab_module = types.ModuleType("isaaclab")
    previous = {name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.app")}
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = app_module
    try:
        import importlib

        battery = importlib.import_module("scripts.isaaclab.velocity_flat_command_battery")
        terms = {}
        for name, dim in (("base_velocity", 3), ("head_pose", 4), ("body_pose", 6)):
            terms[name] = SimpleNamespace(
                command=torch.full((2, dim), 9.0),
                time_left=torch.zeros(2),
                is_standing_env=torch.ones(2, dtype=torch.bool),
            )
        manager = SimpleNamespace(get_term=lambda name: terms[name])
        env = SimpleNamespace(command_manager=manager)

        battery._set_command(env, (0.2, 0.0, -0.7))

        assert torch.equal(terms["base_velocity"].command[0], torch.tensor([0.2, 0.0, -0.7]))
        assert torch.equal(terms["head_pose"].command, torch.zeros(2, 4))
        assert torch.equal(terms["body_pose"].command, torch.zeros(2, 6))
        for term in terms.values():
            assert torch.isinf(term.time_left).all()
        assert not terms["base_velocity"].is_standing_env.any()
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
