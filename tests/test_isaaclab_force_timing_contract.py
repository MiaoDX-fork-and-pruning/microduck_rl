from pathlib import Path


def test_force_timing_probe_covers_physx_force_getters_and_step_phases() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/force_timing_probe.py").read_text()
    assert "get_dof_projected_joint_forces" in source
    assert "get_link_incoming_joint_force" in source
    assert 'capture("after_write_before_step")' in source
    assert 'capture("after_step_before_scene_update")' in source
    assert 'capture("after_scene_update")' in source
    assert 'target[:, 0] += 0.35' in source
    assert 'timing_summary' in source
    assert 'force getters are sampled from the previous solved state' in source
