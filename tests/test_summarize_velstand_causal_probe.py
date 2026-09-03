import json, subprocess, sys
from pathlib import Path


def test_phase_b_failure_is_fail_closed(tmp_path: Path):
    compat = tmp_path / "compat.json"
    rollout = tmp_path / "rollout.json"
    output = tmp_path / "diagnosis.json"
    compat.write_text(json.dumps({"passed": True, "max_abs": 1e-7, "mean_abs": 1e-8}))
    rollout.write_text(json.dumps({"profiles": [{"behavior": "stand", "passed": False,
        "finite": True, "max_tilt_rad": 1.5, "stability_gate_rad": 1.13}]}))
    subprocess.run([sys.executable, "scripts/summarize_velstand_causal_probe.py",
                    "--compatibility", str(compat), "--rollout", str(rollout),
                    "--output", str(output)], check=True)
    report = json.loads(output.read_text())
    assert report["phase_a"]["passed"] is True
    assert report["phase_b"]["passed"] is False
    assert report["diagnosis"] == "state_distribution_coverage_or_model_capacity"
    assert report["recommendation"].startswith("Do not start merged PPO")
