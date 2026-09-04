from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/isaaclab/foot_material_runtime_probe.py"


def test_foot_material_probe_uses_production_task_and_cached_physx_term() -> None:
    source = SCRIPT.read_text()
    assert 'gym.make("IsaacLab-Velocity-Flat-MicroDuck"' in source
    assert 'get_term_cfg("randomize_foot_material")' in source
    assert '"_velocity_flat_foot_material_term"' in source
    assert '"_RandomizeRigidBodyMaterialPhysx"' in source
    assert "get_material_properties()" in source
    assert "event_cfg.func(" in source
    assert "OvArticulation" not in source
    assert "_RandomizeRigidBodyMaterialOvPhysx" not in source


def test_foot_material_probe_checks_both_ankles_and_low_high_readback() -> None:
    source = SCRIPT.read_text()
    assert '{"ankle_left", "ankle_right"}' in source
    assert '(("low", 0.1), ("high", 1.2))' in source
    assert '"unselected_shapes_unchanged"' in source
    assert '"low_high_readbacks_differ"' in source
    assert '"status": "READBACK_PROVEN"' in source
    assert '"response": "not_tested"' in source
