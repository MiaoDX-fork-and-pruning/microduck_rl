from pathlib import Path


def test_student_rollout_entrypoint_exists():
    assert (Path(__file__).parents[1] / "scripts" / "rollout_generalist_bc.py").exists()
