from __future__ import annotations

from pathlib import Path


def test_physics_battery_exposes_required_bringup_cases_and_boundaries() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/physics_battery.py").read_text()
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert "MICRODUCK_CFG" in source
    assert '"home_settle"' in source
    assert '"free_fall"' in source
    assert '"target_step"' in source
    assert '"nan_soak"' in source
    assert '"contact_force_parity": "pending"' in source
    assert '"slip_parity": "pending"' in source
    assert '"friction_bridge": "motor_only_external_effort_unavailable"' in source
    assert "physx_friction_coefficients" in source
    assert "write_joint_friction_coefficient_to_sim_index" in source
    assert "torch.isfinite" in source
    assert "root_pos_w" in source
    assert "root_quat_w" in source
    assert "root_link_pos_w" in source
    assert "root_link_quat_w" in source
