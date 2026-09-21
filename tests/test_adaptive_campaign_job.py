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


def _write_result(path, checkpoint, *, start=5, completed=7, task="task", events=None):
    state = {"task_id": task, "completed_iterations": completed,
             "num_envs": 64, "evaluation_seed": 20260815,
             "evaluation_events": events or [{"kind": "hold"}]}
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


def test_resumed_job_smokes_fresh_then_trains_only_remaining_budget(monkeypatch, tmp_path):
    task, _ = campaign.TASKS["feedback"]
    checkpoint = tmp_path / "checkpoint with spaces.pt"
    _write_result(tmp_path / "prior-result.json", checkpoint, start=0, completed=5, task=task)
    output = tmp_path / "continuation"
    monkeypatch.setenv("MICRODUCK_SOURCE_SHA", "reviewed-source")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", "/unrelated/inherited.pt")
    monkeypatch.setattr(sys, "argv", ["campaign", "--branch", "feedback", "--seed", "17",
        "--output", str(output), "--iterations", "7", "--num-envs", "64",
        "--gate-interval", "1", "--resume", str(checkpoint)])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
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
                _write_result(Path(env["MICRODUCK_ADAPTIVE_RESULT_FILE"]), final, task=task)
        else:
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
