from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mjlab_microduck.evaluation.capability import BUCKETS, CapabilityReport, build_capability_report

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_adaptive_checkpoint_battery.py"
_SPEC = importlib.util.spec_from_file_location("run_adaptive_checkpoint_battery", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
battery = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = battery
_SPEC.loader.exec_module(battery)


def _raw_metrics() -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in BUCKETS:
        key = (
            "zero_drift_m"
            if name == "zero"
            else "angular_tracking_error_rad_s"
            if name in {"yaw", "turn-left", "turn-right"}
            else "tracking_error_m_s"
        )
        result[name] = {"survival_fraction": 1.0, "tilt_p95_rad": 0.1, key: 0.0}
    return result


def _battery_payload(onnx: Path, seed: int, *, source_sha: str = "src") -> dict:
    report = build_capability_report(
        _raw_metrics(),
        axis_mode="composed",
        metadata={
            "task_id": "Mjlab-Velocity-Flat-Adaptive-MicroDuck",
            "source_sha": source_sha,
            "evaluator_config_sha256": "config",
            "checkpoint": str(onnx),
            "checkpoint_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            "policy_format": "onnx",
            "seed_set_id": f"adaptive-default-{seed}",
            "generated_at": "now",
        },
    )
    return {
        **report.payload,
        "seed": seed,
        "cases": [
            {"bucket": bucket, "steps": 300, "finite_61d_14d": True}
            for bucket in BUCKETS
        ],
    }


def _fake_subprocess(monkeypatch, tmp_path: Path, *, battery_code: int = 0, seed: int = 17):
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setenv("MICRODUCK_SOURCE_SHA", "test-source")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "export.py" in command[1]:
            onnx = Path(command[command.index("--onnx-file") + 1])
            onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx.write_bytes(b"exported-onnx")
            return subprocess.CompletedProcess(command, 0)
        battery_dir = Path(command[command.index("--output") + 1])
        onnx = Path(command[command.index("--onnx") + 1])
        battery_dir.mkdir(parents=True, exist_ok=True)
        (battery_dir / "capability.json").write_text(
            json.dumps(_battery_payload(onnx, seed)), encoding="utf-8"
        )
        import numpy as np

        for bucket in battery.BUCKETS:
            np.savez_compressed(
                battery_dir / f"{bucket}.npz",
                observation=np.zeros((300, 61), dtype=np.float32),
                raw_action=np.zeros((300, 14), dtype=np.float32),
                applied_action=np.zeros((300, 14), dtype=np.float32),
            )
        return subprocess.CompletedProcess(command, battery_code)

    monkeypatch.setattr(battery.subprocess, "run", run)
    return calls


def test_valid_failed_battery_is_a_report_and_binds_both_artifacts(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model_250.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "eval" / "capability.json"
    calls = _fake_subprocess(monkeypatch, tmp_path, battery_code=2)

    payload = battery.run_checkpoint_battery(
        checkpoint=checkpoint,
        task_id="Mjlab-Velocity-Flat-Adaptive-MicroDuck",
        axis_mode="composed",
        seed_set_id="held-out-v1",
        evaluation_seed=17,
        output=output,
    )

    assert payload["seed"] == 17
    assert payload["metadata"]["checkpoint"] == str(checkpoint)
    assert payload["metadata"]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    onnx = Path(payload["metadata"]["exported_onnx"])
    assert onnx.exists()
    assert payload["metadata"]["exported_onnx_sha256"] == hashlib.sha256(onnx.read_bytes()).hexdigest()
    assert payload["metadata"]["seed_set_id"] == "held-out-v1"
    assert payload["metadata"]["evaluation_seed"] == 17
    CapabilityReport.from_dict(payload)
    export_call = calls[0][0]
    assert export_call[2] == "Mjlab-Velocity-Flat-Adaptive-MicroDuck"
    assert export_call[export_call.index("--num-envs") + 1] == "1"
    assert export_call[export_call.index("--device") + 1] == "cuda:0"


def test_battery_exception_exit_is_not_accepted(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = _fake_subprocess(monkeypatch, tmp_path, battery_code=3)

    with pytest.raises(subprocess.CalledProcessError):
        battery.run_checkpoint_battery(
            checkpoint=checkpoint,
            task_id="Mjlab-Velocity-Flat-Adaptive-MicroDuck",
            axis_mode="composed",
            seed_set_id="held-out-v1",
            evaluation_seed=17,
            output=tmp_path / "eval" / "capability.json",
        )
    assert len(calls) == 2


def test_missing_source_sha_is_rejected(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.delenv("MICRODUCK_SOURCE_SHA", raising=False)
    with pytest.raises(ValueError, match="MICRODUCK_SOURCE_SHA"):
        battery.run_checkpoint_battery(
            checkpoint=checkpoint,
            task_id="Mjlab-Velocity-Flat-Adaptive-MicroDuck",
            axis_mode="composed",
            seed_set_id="held-out-v1",
            evaluation_seed=17,
            output=tmp_path / "eval" / "capability.json",
        )


def test_missing_trace_is_rejected(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    _fake_subprocess(monkeypatch, tmp_path, battery_code=2)

    original = battery._validate_battery_traces
    monkeypatch.setattr(
        battery,
        "_validate_battery_traces",
        lambda battery_dir, payload: original(battery_dir / "missing", payload),
    )
    with pytest.raises(FileNotFoundError, match="missing zero trace"):
        battery.run_checkpoint_battery(
            checkpoint=checkpoint,
            task_id="Mjlab-Velocity-Flat-Adaptive-MicroDuck",
            axis_mode="composed",
            seed_set_id="held-out-v1",
            evaluation_seed=17,
            output=tmp_path / "eval" / "capability.json",
        )


def test_seed_set_and_evaluation_seed_are_required_and_consistent(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    _fake_subprocess(monkeypatch, tmp_path, seed=18)

    with pytest.raises(ValueError, match="evaluation seed mismatch"):
        battery.run_checkpoint_battery(
            checkpoint=checkpoint,
            task_id="Mjlab-Velocity-Flat-Adaptive-MicroDuck",
            axis_mode="composed",
            seed_set_id="held-out-v1",
            evaluation_seed=17,
            output=tmp_path / "eval" / "capability.json",
        )
    with pytest.raises(ValueError, match="seed_set_id"):
        battery.run_checkpoint_battery(
            checkpoint=checkpoint,
            task_id="Mjlab-Velocity-Flat-Adaptive-MicroDuck",
            axis_mode="composed",
            seed_set_id="",
            evaluation_seed=17,
            output=tmp_path / "eval2" / "capability.json",
        )
