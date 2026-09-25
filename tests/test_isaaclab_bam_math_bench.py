from __future__ import annotations

from scripts.isaaclab.bam_math_bench import run_bench


def test_bam_math_bench_is_numerically_exact() -> None:
    report = run_bench()
    assert report["samples"] == 225
    assert report["voltage_max_abs_error"] < 1e-8
    assert report["torque_max_abs_error"] < 1e-8
    assert report["friction_nominal_max_abs_error"] < 1e-8
