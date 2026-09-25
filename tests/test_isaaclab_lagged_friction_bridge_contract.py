from pathlib import Path


def test_lagged_bridge_probe_keeps_sampling_and_application_phases_explicit() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/lagged_friction_bridge_probe.py").read_text()
    assert "get_dof_projected_joint_forces" in source
    assert "get_dof_actuation_forces" in source
    assert "scene.update(dt)" in source
    assert 'mode in ("motor_only", "one_step_lag")' in source
    assert '"lag_steps": 1' in source
    assert '"production_wiring": "diagnostic_only_not_applied"' in source
