from pathlib import Path

def test_switching_battery_entrypoint_exists():
    assert (Path(__file__).parents[1] / "scripts" / "run_specialist_switching_battery.py").exists()


def test_switching_report_requires_physical_stability():
    import json
    report = json.loads((Path(__file__).parents[1] / "artifacts/generalist-v0/specialist-switch-track-a.json").read_text())
    assert report["reset_count"] == 0
    assert len(report["policy_sequence"]) >= 2
    assert report["passed"] is False  # frozen pre-fix evidence remains a regression fixture
    assert report["failure_reason"] == "physical_fall_during_switching_sequence"


def test_switching_runner_preserves_runtime_command_contracts(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from run_specialist_switching_battery import run
    root = Path(__file__).parents[1]
    a = run(root / "docs/specialist_demo_scenario.json", False, tmp_path / "a.json")
    b = run(root / "docs/specialist_demo_track_b_scenario.json", True, tmp_path / "b.json")
    for report in (a, b):
        assert report["passed"] is True
        assert report["reset_count"] == 0
        assert report["world_displacement_m"]
