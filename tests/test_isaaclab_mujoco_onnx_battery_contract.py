from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/isaaclab/mujoco_onnx_battery.py"


def test_headless_mujoco_battery_uses_shared_cases_and_deployable_abi() -> None:
    source = SCRIPT.read_text()
    assert "velocity_flat_battery_spec import CASES" in source
    assert '"schema": "mujoco_onnx_battery.v1"' in source
    assert '"actor_obs_dim": 61' in source
    assert '"action_dim": 14' in source
    assert "mujoco.viewer" not in source


def test_headless_battery_records_directional_and_finite_state_metrics() -> None:
    source = SCRIPT.read_text()
    for token in (
        "mean_actual_vel_xy_m_s",
        "mean_actual_vel_yaw_rad_s",
        "max_tilt_rad",
        "world_displacement_m",
        "evaluate_case(",
    ):
        assert token in source
