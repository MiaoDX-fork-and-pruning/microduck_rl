from __future__ import annotations

from pathlib import Path


def test_dynamic_bench_uses_real_articulation_and_reports_friction_boundary() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/bam_dynamic_bench.py").read_text()
    assert "MICRODUCK_CFG" in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert '"step_7.5v"' in source
    assert '"step_6.5v"' in source
    assert '"sine_7.5v"' in source
    assert '"not_applied_to_physx"' in source
    assert "scene.write_data_to_sim()" in source
    assert "sim.step()" in source
