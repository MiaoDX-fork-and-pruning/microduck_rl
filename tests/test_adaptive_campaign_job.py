"""Launcher delegates state ownership and consumes a concrete completion record."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location("adaptive_campaign_job", Path(__file__).parents[1] / "scripts/run_adaptive_campaign_job.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def _write_result(path, checkpoint, *, start=5, completed=7, task="task", events=None,
                  fraction=None, transition=False):
    state = {"task_id": task, "completed_iterations": completed,
             "num_envs": 64, "evaluation_seed": 20260815,
             "evaluation_events": events or [{"kind": "hold"}]}
    if fraction is not None:
        state["final_com_fraction"] = fraction
    if transition:
        state["transition_exposure"] = {"probability": 0.2}
        state["transition_bootstrap_mode"] = "zero"
    torch.save({"iter": completed - 1, "infos": {"adaptive_curriculum": state}}, checkpoint)
    result = {"version": 1, "status": "completed", "task_id": task,
              "completed_iterations": completed, "start_completed_iterations": start,
              "num_envs": 64, "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "adaptive_state": state}
    path.write_text(json.dumps(result))
    return result


def test_completion_uses_manifest_path_and_rejects_tampered_artifact(tmp_path):
    checkpoint = tmp_path / "explicit-final.pt"
    result = tmp_path / "training-result.json"
    _write_result(result, checkpoint)
    args = dict(task_id="task", completed_iterations=7, start_iterations=5, num_envs=64)
    found, state = campaign.read_training_result(result, **args)
    assert found == checkpoint
    assert state["completed_iterations"] == 7
    checkpoint.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.read_training_result(result, **args)


def test_gate_cohort_seed_coverage_cannot_overlap_heldout(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "campaign", "--branch", "feedback", "--seed", "17",
        "--output", str(tmp_path / "run"), "--iterations", "1",
        "--gate-seed", "100", "--gate-cohort-size", "3",
        "--heldout-seed", "106",
    ])
    with pytest.raises(SystemExit):
        campaign.main()


@pytest.mark.parametrize("saved_fraction,requested_fraction,expected_fraction", [
    (None, None, 0.0), (0.2, None, 0.2), (None, 0.2, 0.2), (0.2, 0.0, 0.0),
])
def test_resumed_job_smokes_fresh_then_trains_only_remaining_budget(
    monkeypatch, tmp_path, saved_fraction, requested_fraction, expected_fraction
):
    task, _ = campaign.TASKS["feedback"]
    checkpoint = tmp_path / "checkpoint with spaces.pt"
    _write_result(tmp_path / "prior-result.json", checkpoint, start=0, completed=5, task=task,
                  fraction=saved_fraction, transition=True)
    output = tmp_path / "continuation"
    monkeypatch.setenv("MICRODUCK_SOURCE_SHA", "reviewed-source")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", "/unrelated/inherited.pt")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION", "0.1")
    argv = ["campaign", "--branch", "feedback", "--seed", "17",
        "--output", str(output), "--iterations", "7", "--num-envs", "64",
        "--gate-interval", "1", "--resume", str(checkpoint)]
    if requested_fraction is not None:
        argv += ["--final-com-fraction", str(requested_fraction)]
    argv += ["--transition-probability", "0.2", "--transition-mode", "zero"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION"] == str(expected_fraction)
        if Path(command[0]).name == "train":
            phase = Path(kwargs["cwd"]).name
            env = kwargs["env"]
            if phase == "smoke":
                assert "MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT" not in env
                assert env["MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL"] == "0"
                assert command[command.index("--agent.max-iterations") + 1] == "5"
            else:
                assert env["MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT"] == str(checkpoint)
                assert command[command.index("--agent.max-iterations") + 1] == "2"
                final = output / "training" / "explicit-final.pt"
                _write_result(Path(env["MICRODUCK_ADAPTIVE_RESULT_FILE"]), final, task=task,
                              fraction=expected_fraction, transition=True)
        else:
            for name in (
                "MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY",
                "MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE",
                "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE",
                "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE",
                "MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION",
            ):
                assert name not in kwargs["env"]
            report = Path(command[command.index("--output") + 1])
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"aggregate": {"passed": False, "score": 0.0}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    assert campaign.main() == 0
    assert len(calls) == 4
    result = json.loads((output / "campaign-result.json").read_text())
    assert result["total_training_transitions"] == 7 * 24 * 64
    assert result["segment_training_transitions"] == 2 * 24 * 64
    assert result["heldout_aggregate"]["passed"] is False
    assert result["cpu_transfer_aggregate"]["passed"] is False
    assert result["final_com_fraction"] == expected_fraction
