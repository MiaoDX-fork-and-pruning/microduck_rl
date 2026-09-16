from pathlib import Path

import pytest

from mjlab_microduck.tasks.adaptive_runner import AdaptiveMicroduckOnPolicyRunner
from mjlab_microduck.tasks.adaptive_curriculum import CapabilityGate, ADAPTIVE_AXIS_CONFIGS
from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report


def _report(path: Path, mode="composed"):
    raw = {}
    for name in BUCKETS:
        key = "zero_drift_m" if name == "zero" else "angular_tracking_error_rad_s" if name in ("yaw", "turn-left", "turn-right") else "tracking_error_m_s"
        raw[name] = {"survival_fraction": 1.0, "tilt_p95_rad": 0.1, key: 0.0}
    return build_capability_report(raw, axis_mode=mode, metadata={
        "task_id": "adaptive_velocity", "source_sha": "src", "evaluator_config_sha256": "cfg",
        "checkpoint": str(path), "checkpoint_sha256": "hash", "policy_format": "onnx",
        "seed_set_id": "0", "generated_at": "now",
    })


def _runner(tmp_path):
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.evaluation_interval = 1
    runner.evaluation_seed = 0
    runner.current_learning_iteration = 1
    runner.evaluation_events = []
    runner.last_known_good_checkpoint = None
    runner.capability_gate = None
    runner.env = type("Env", (), {"cfg": type("Cfg", (), {"adaptive_evaluator_schema_version": 2})()})()
    runner.capability_gate = CapabilityGate(ADAPTIVE_AXIS_CONFIGS, critical_buckets=BUCKETS, axis_mode="composed")
    return runner


def test_invalid_report_fails_closed(tmp_path):
    runner = _runner(tmp_path)
    class Evaluator:
        def evaluate(self, **kwargs):
            raise RuntimeError("battery unavailable")
    runner.evaluator = Evaluator()
    runner._evaluate_window(str(tmp_path / "model.pt"))
    assert runner.evaluation_events[0]["kind"] == "evaluation_error"


def test_report_validator_rejects_wrong_axis(tmp_path):
    runner = _runner(tmp_path)
    class Gate:
        axis_mode = "composed"
        axis_order = ("com_range", "head_com_range")
    runner.capability_gate = Gate()
    report = _report(tmp_path / "model.pt", mode="all_static")
    with pytest.raises(ValueError, match="axis contract"):
        runner._validate_report(report, str(tmp_path / "model.pt"))


def test_report_validator_requires_existing_checkpoint(tmp_path):
    runner = _runner(tmp_path)
    report = _report(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="does not exist"):
        runner._validate_report(report, str(tmp_path / "missing.pt"))


def test_rollback_requires_recorded_known_good_checkpoint(tmp_path):
    runner = _runner(tmp_path)
    runner.last_known_good_checkpoint = str(tmp_path / "good.pt")
    with pytest.raises(ValueError, match="known-good"):
        runner.rollback(str(tmp_path / "other.pt"))
