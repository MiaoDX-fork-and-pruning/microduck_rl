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
                  fraction=None, sensor_reset_fraction=None, transition=False, distribution="final"):
    state = {"task_id": task, "completed_iterations": completed,
             "num_envs": 64, "evaluation_seed": 20260815,
             "evaluation_events": events or [{"kind": "hold"}],
             "evaluation_distribution": distribution}
    if fraction is not None:
        state["final_com_fraction"] = fraction
    if sensor_reset_fraction is not None:
        state["sensor_reset_fraction"] = sensor_reset_fraction
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


def _write_consolidation_stop(path, checkpoint, task="task"):
    from mjlab_microduck.evaluation.capability import BUCKETS
    from mjlab_microduck.tasks.adaptive_curriculum import EntropyConsolidation

    result = _write_result(path, checkpoint, task=task)
    controller = EntropyConsolidation()
    controller.begin({**dict.fromkeys(BUCKETS, 0.7), "zero": 0.9},
                     entropy_coef=0.01, checkpoint="baseline.pt",
                     completed_iterations=5, window_updates=2, focus_bucket="yaw")
    controller.finish(dict.fromkeys(BUCKETS, 0.9), gate_retained=True, completed_iterations=7)
    result["adaptive_state"]["entropy_consolidation"] = controller.state_dict()
    torch.save({"iter": 6, "infos": {"adaptive_curriculum": result["adaptive_state"]}}, checkpoint)
    result.update(status="stopped", requested_completed_iterations=10,
                  stop_reason="entropy_consolidation_terminal",
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    path.write_text(json.dumps(result))
    return result


def test_early_completion_requires_a_verified_terminal_attempt(tmp_path):
    checkpoint = tmp_path / "retained.pt"
    path = tmp_path / "result.json"
    result = _write_consolidation_stop(path, checkpoint)
    args = {"task_id": "task", "completed_iterations": 10, "start_iterations": 5, "num_envs": 64}
    found, state = campaign.read_training_result(path, **args)
    assert found == checkpoint and state["completed_iterations"] == 7
    result["stop_reason"] = "interrupted"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="without a completed consolidation"):
        campaign.read_training_result(path, **args)


def test_early_stopped_campaign_evaluates_retained_actor_and_counts_actual_budget(monkeypatch, tmp_path):
    task, _ = campaign.TASKS["lateral-drive"]
    checkpoint = tmp_path / "start.pt"
    _write_result(tmp_path / "prior.json", checkpoint, start=0, completed=5, task=task)
    output = tmp_path / "campaign"
    monkeypatch.setenv("MICRODUCK_SOURCE_SHA", "test-source")
    monkeypatch.setattr(sys, "argv", [
        "campaign", "--branch", "lateral-drive", "--seed", "17", "--num-envs", "64",
        "--iterations", "10", "--resume", str(checkpoint), "--output", str(output),
        "--entropy-consolidation", "--gate-interval", "2",
    ])
    evaluated = []

    def run(command, **kwargs):
        if Path(command[0]).name == "train":
            assert command[command.index("--env.adaptive-entropy-consolidation") + 1] == "True"
            if Path(kwargs["cwd"]).name == "training":
                _write_consolidation_stop(Path(kwargs["env"]["MICRODUCK_ADAPTIVE_RESULT_FILE"]),
                                          output / "retained.pt", task)
        else:
            evaluated.append(command[command.index("--checkpoint") + 1])
            report = Path(command[command.index("--output") + 1])
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"aggregate": {"passed": False}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    assert campaign.main() == 0
    result = json.loads((output / "campaign-result.json").read_text())
    assert result["iterations"] == 10 and result["completed_iterations"] == 7
    assert result["segment_training_transitions"] == 2 * 24 * 64
    assert result["total_training_transitions"] == 7 * 24 * 64
    assert evaluated == [str(output / "retained.pt")] * 2


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
@pytest.mark.parametrize("gate_distribution", ["final", "stage"])
def test_resumed_job_smokes_fresh_then_trains_only_remaining_budget(
    monkeypatch, tmp_path, saved_fraction, requested_fraction, expected_fraction, gate_distribution
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
    if gate_distribution == "stage":
        argv += ["--gate-distribution", "stage", "--rebaseline-gate"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION"] == str(expected_fraction)
        if Path(command[0]).name == "train":
            phase = Path(kwargs["cwd"]).name
            env = kwargs["env"]
            assert env["MICRODUCK_ADAPTIVE_EVALUATION_DISTRIBUTION"] == gate_distribution
            assert "--distribution {distribution}" in env["MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND"]
            if phase == "smoke":
                assert "MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT" not in env
                assert env["MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL"] == "0"
                assert command[command.index("--agent.max-iterations") + 1] == "5"
            else:
                assert env["MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT"] == str(checkpoint)
                assert command[command.index("--agent.max-iterations") + 1] == "2"
                final = output / "training" / "explicit-final.pt"
                _write_result(Path(env["MICRODUCK_ADAPTIVE_RESULT_FILE"]), final, task=task,
                              fraction=expected_fraction, transition=True, distribution=gate_distribution)
        else:
            for name in (
                "MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY",
                "MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE",
                "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE",
                "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE",
                "MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION",
                "MICRODUCK_ADAPTIVE_EVALUATION_DISTRIBUTION",
                "MICRODUCK_ADAPTIVE_ALLOW_DISTRIBUTION_MIGRATION",
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
    assert result["gate_distribution"] == gate_distribution


def test_campaign_cannot_silently_reuse_final_gate_evidence_for_stage_gate(monkeypatch, tmp_path):
    task, _ = campaign.TASKS["feedback"]
    checkpoint = tmp_path / "final.pt"
    _write_result(tmp_path / "prior.json", checkpoint, task=task)
    monkeypatch.setattr(sys, "argv", [
        "campaign", "--branch", "feedback", "--seed", "17", "--iterations", "8",
        "--num-envs", "64", "--resume", str(checkpoint),
        "--output", str(tmp_path / "run"), "--gate-distribution", "stage",
    ])
    with pytest.raises(SystemExit):
        campaign.main()
    assert not (tmp_path / "run").exists()


def test_resumed_job_recreates_sensor_reset_setting_from_checkpoint(
    monkeypatch, tmp_path
):
    task, _ = campaign.TASKS["feedback"]
    checkpoint = tmp_path / "sensor-reset-checkpoint.pt"
    _write_result(
        tmp_path / "prior-result.json",
        checkpoint,
        start=0,
        completed=5,
        task=task,
        sensor_reset_fraction=1.0,
    )
    output = tmp_path / "continuation"
    monkeypatch.setenv("MICRODUCK_SOURCE_SHA", "reviewed-source")
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "campaign", "--branch", "feedback", "--seed", "17",
        "--output", str(output), "--iterations", "7", "--num-envs", "64",
        "--gate-interval", "1", "--resume", str(checkpoint),
    ])

    def run(command, **kwargs):
        env = kwargs["env"]
        if Path(command[0]).name == "train":
            assert env["MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION"] == "1.0"
            phase = Path(kwargs["cwd"]).name
            if phase == "training":
                final = output / "training" / "model_6.pt"
                _write_result(
                    Path(env["MICRODUCK_ADAPTIVE_RESULT_FILE"]),
                    final,
                    start=5,
                    completed=7,
                    task=task,
                    sensor_reset_fraction=1.0,
                )
        else:
            assert "MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION" not in env
            report = Path(command[command.index("--output") + 1])
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"aggregate": {"passed": False}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    assert campaign.main() == 0
    config = json.loads((output / "campaign-config.json").read_text())
    assert config["sensor_reset_fraction"] == "1.0"


def test_resumed_job_preserves_pure_yaw_scope_in_smoke_and_training_only(monkeypatch, tmp_path):
    task, _ = campaign.TASKS["lateral-drive"]
    checkpoint = tmp_path / "pure-yaw.pt"
    _write_result(tmp_path / "prior.json", checkpoint, task=task, completed=5)
    saved = torch.load(checkpoint, weights_only=False)
    saved["infos"]["adaptive_curriculum"]["action_rate_relief"] = {"version": 2, "scope": "pure_yaw"}
    torch.save(saved, checkpoint)
    output = tmp_path / "continuation"
    monkeypatch.setenv("MICRODUCK_SOURCE_SHA", "reviewed-source")
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "campaign", "--branch", "lateral-drive", "--seed", "17",
        "--output", str(output), "--iterations", "7", "--num-envs", "64",
        "--gate-interval", "1", "--resume", str(checkpoint),
    ])

    def run(command, **kwargs):
        env = kwargs["env"]
        if Path(command[0]).name == "train":
            assert env["MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE"] == "pure_yaw"
            if Path(kwargs["cwd"]).name == "training":
                _write_result(Path(env["MICRODUCK_ADAPTIVE_RESULT_FILE"]),
                              output / "training/model_6.pt", task=task)
        else:
            assert "MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE" not in env
            report = Path(command[command.index("--output") + 1])
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"aggregate": {"passed": False}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    assert campaign.main() == 0
    config = json.loads((output / "campaign-config.json").read_text())
    assert config["action_rate_relief_scope"] == "pure_yaw"
