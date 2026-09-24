from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_sensor_source_trace_covers_raw_sources_and_processing_contract() -> None:
    mjlab = (ROOT / "scripts/mjlab_sensor_source_trace.py").read_text()
    isaaclab = (ROOT / "scripts/isaaclab/sensor_source_trace.py").read_text()
    comparator = (ROOT / "scripts/compare_sensor_source_trace.py").read_text()
    for source in (mjlab, isaaclab):
        assert '"delay_index": 0' in source
        assert '"reset_behavior"' in source
        assert '"subtree_angular_momentum"' in source
    assert '"subtree_angular_momentum", "named_root_angmom", "subtree_angmom_adapter"' in comparator
    assert '"processing": {"mjlab": mj.get("processing", {}), "isaaclab": isaac.get("processing", {})}' in comparator


def test_sensor_source_comparator_reports_max_and_mean_error(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "backend": "mjlab",
        "processing": {"delay_index": 0},
        "source_notes": {},
        "records": [{"raw": {"root_ang_vel_b_adapter": [0.0], "projected_gravity_b_adapter": [1.0], "joint_pos": [0.0], "joint_vel": [0.0], "foot_site_pos_w": [0.0], "foot_site_vel_w": [0.0], "contact_found": [0.0], "contact_force": [0.0], "named_root_angmom": [0.0]}}],
    }
    observed = json.loads(json.dumps(report))
    observed["backend"] = "isaaclab"
    observed["records"][0]["raw"]["root_ang_vel_b_adapter"] = [0.25]
    observed["records"][0]["raw"]["subtree_angmom_adapter"] = [0.5]
    mj_path = tmp_path / "mj.json"
    isaac_path = tmp_path / "isaac.json"
    out_path = tmp_path / "comparison.json"
    mj_path.write_text(json.dumps(report))
    isaac_path.write_text(json.dumps(observed))

    from scripts.compare_sensor_source_trace import compare

    result = compare(mj_path, isaac_path, out_path)
    assert result["signals"]["gyro_adapter"]["max_abs_error"] == 0.25
    assert result["signals"]["subtree_angular_momentum"]["max_abs_error"] == 0.5
