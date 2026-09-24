"""The real CPU model must enforce the same current cap in both rehearsal paths."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
from bam.model import load_model

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from infer_policy import apply_xl330_current_limit  # noqa: E402
import run_specialist_action_battery as battery  # noqa: E402


def model():
    return mujoco.MjModel.from_xml_path(str(battery.PROFILE_SCENES["walk_all_collisions"]))


@pytest.mark.parametrize("amps", [1.75, 0.8])
def test_current_limit_caps_actual_servo_force(amps):
    robot = model()
    data = mujoco.MjData(robot)
    data.ctrl[:] = 10.0
    mujoco.mj_forward(robot, data)
    expected = float(load_model(motor_name="xl330", model="m6").kt.value) * amps
    assert np.abs(data.actuator_force).max() > expected

    limit = apply_xl330_current_limit(robot, amps)
    mujoco.mj_forward(robot, data)

    assert limit == pytest.approx(expected)
    assert robot.nu == 14
    np.testing.assert_allclose(np.abs(data.actuator_force), expected, atol=1e-12)


@pytest.mark.parametrize("amps", [None, 0.0, -1.0])
def test_disabled_override_preserves_xml_limits(amps):
    robot = model()
    before = robot.actuator_forcerange.copy()
    assert apply_xl330_current_limit(robot, amps) is None
    np.testing.assert_array_equal(robot.actuator_forcerange, before)


@pytest.mark.parametrize("amps", [float("nan"), float("inf"), -float("inf")])
def test_invalid_current_does_not_change_the_model(amps):
    robot = model()
    before = robot.actuator_forcerange.copy()
    with pytest.raises(ValueError, match="finite"):
        apply_xl330_current_limit(robot, amps)
    np.testing.assert_array_equal(robot.actuator_forcerange, before)


def test_headless_battery_applies_runtime_default_before_loading_policy(monkeypatch):
    expected = float(load_model(motor_name="xl330", model="m6").kt.value) * 1.75

    class ModelChecked(Exception):
        pass

    def check_model(robot, data, onnx_path):
        np.testing.assert_allclose(robot.actuator_forcerange[:, 1], expected)
        np.testing.assert_allclose(robot.actuator_forcerange[:, 0], -expected)
        assert robot.actuator_forcelimited.all()
        raise ModelChecked

    monkeypatch.setattr(battery, "_load_policy", check_model)
    case = battery.command_cases("adaptive_velocity", smoke=False)[0]
    with pytest.raises(ModelChecked):
        battery.run_case("adaptive_velocity", Path("unused.onnx"), case, seed=17, duration_s=6.0)
