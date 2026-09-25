from __future__ import annotations

from scripts.isaaclab.velocity_flat_backend_neutral_fixture import run_fixture


def test_backend_neutral_fixture_matches_mjlab_after_subset_reset() -> None:
    report = run_fixture(seed=2026, num_envs=4, steps=10)
    assert report["passed"] is True
    assert report["max_target_error"] <= 1e-6
    # IsaacLab's runtime tensors are float32, matching the reference Torch
    # backend; these bounds allow only normal single-precision roundoff.
    assert report["max_voltage_error"] <= 5e-7
    assert report["max_torque_error"] <= 1e-6
    assert report["rows"][5]["reset_env_ids"] == [1]
    assert all(row["finite"] for row in report["rows"])
