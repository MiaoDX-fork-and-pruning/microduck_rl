from pathlib import Path


ROOT = Path(__file__).parents[1]
TRACE = ROOT / "scripts/isaaclab/velocity_flat_directional_trace.py"


def test_directional_trace_reuses_strict_contract_and_intermediate_fields() -> None:
    source = TRACE.read_text()
    assert "velocity_flat_battery_spec" in source
    assert '"zero", "forward", "lateral", "yaw"' in source
    assert '"raw_policy_action"' in source
    assert '"bam_delayed_target"' in source
    assert '"bam_applied_effort"' in source
    assert '"bam_delay_lag"' in source
    assert '"bam_delay_initialized"' in source
    assert '"root_lin_vel_b"' in source
    assert "clip_actions=agent_cfg.clip_actions" in source
