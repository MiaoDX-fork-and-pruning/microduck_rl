from __future__ import annotations

from pathlib import Path


def test_articulation_probe_has_stage_markers_and_real_step() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/articulation_probe.py").read_text()
    assert "ISAACLAB_ARTICULATION_PROBE:app_ready" in source
    assert "ISAACLAB_ARTICULATION_PROBE:articulation_ready" in source
    assert "scene.write_data_to_sim()" in source
    assert "sim.step()" in source
