from pathlib import Path

def test_switching_battery_entrypoint_exists():
    assert (Path(__file__).parents[1] / "scripts" / "run_specialist_switching_battery.py").exists()


def test_switching_report_requires_physical_stability():
    import json
    report = json.loads((Path(__file__).parents[1] / "artifacts/generalist-v0/specialist-switch-track-a.json").read_text())
    assert report["reset_count"] == 0
    assert len(report["policy_sequence"]) >= 2
    assert report["passed"] is False
    assert report["failure_reason"] == "physical_fall_during_switching_sequence"
