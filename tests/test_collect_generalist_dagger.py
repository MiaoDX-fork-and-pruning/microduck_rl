from pathlib import Path

def test_dagger_entrypoint_exists():
    assert (Path(__file__).parents[1] / "scripts" / "collect_generalist_dagger.py").exists()
