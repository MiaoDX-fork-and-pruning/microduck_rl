from __future__ import annotations

from pathlib import Path


def test_articulation_probe_has_stage_markers_and_real_step() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/articulation_probe.py").read_text()
    assert "ISAACLAB_ARTICULATION_PROBE:app_ready" in source
    assert "ISAACLAB_ARTICULATION_PROBE:articulation_ready" in source
    assert "scene.write_data_to_sim()" in source
    assert "sim.step()" in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert "AppLauncher(args)" in source


def test_articulation_probe_exposes_native_reset_diagnostics() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/articulation_probe.py").read_text()
    assert "--no-collisions" in source
    assert "--cartpole" in source
    assert "--reset-timeout" in source
    assert "collisions_removed=" in source
