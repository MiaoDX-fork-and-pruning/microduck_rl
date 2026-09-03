from pathlib import Path


def test_friction_sweep_is_fixed_root_and_reports_bridge_boundary() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/friction_sweep.py").read_text()
    assert "kinematic_state_reset" in source
    assert "write_root_pose_to_sim_index" in source
    assert "policy_target_to_sim" in source
    assert "for scale in (0.5, 1.0, 1.5)" in source
    assert "physx_friction_coefficients" in source
    assert "write_joint_friction_coefficient_to_sim_index" in source
    assert '"friction_bridge": "motor_only_external_effort_unavailable"' in source
    assert "torch.isfinite" in source
