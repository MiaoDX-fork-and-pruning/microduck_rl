import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "validate_specialist_artifacts",
    Path(__file__).parents[1] / "scripts" / "validate_specialist_artifacts.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
validate = _MODULE.validate


def _inputs(tmp_path):
    artifacts = {
        "checkpoint": tmp_path / "model.pt",
        "onnx": tmp_path / "policy.onnx",
        "metadata": tmp_path / "metadata.json",
        "evaluation_report": tmp_path / "evaluation.json",
        "diagnostic_video": tmp_path / "diagnostic.mp4",
        "parity_report": tmp_path / "parity.json",
    }
    artifacts["checkpoint"].write_bytes(b"checkpoint")
    artifacts["onnx"].write_bytes(b"onnx")
    artifacts["diagnostic_video"].write_bytes(b"video")
    artifacts["metadata"].write_text(json.dumps({"observation_dim": 61, "action_dim": 14}))
    artifacts["evaluation_report"].write_text(json.dumps({
        "accepted": True,
        "finite": True,
        "success_rate": 0.9,
        "main_task_metric": 1.2,
        "penalty_terms": {"action_rate": -0.1},
        "video_review": "Completes the expected motion without falling.",
    }))
    artifacts["parity_report"].write_text(json.dumps({
        "passed": True, "observation_dim": 61, "action_dim": 14,
    }))
    manifest = {
        "policies": [{
            "id": "velstand_flat",
            "task": "Mjlab-VelStand-Flat-MicroDuck",
            "accepted": True,
            "observation_dim": 61,
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in artifacts.items()},
        }]
    }
    scenario = {
        "seed": 1234,
        "duration_s": 10,
        "command_rate_hz": 50,
        "compatibility": {"scene": "test_scene", "session": "test_session"},
        "transitions": [{
            "at_s": 0,
            "from": "velstand_flat",
            "to": "velstand_flat",
            "min_dwell_s": 10,
            "non_interruptible_s": 0,
            "expected_outcome": {"metric": "upright", "operator": "gte", "threshold": 0.8},
        }],
        "unsupported_transitions": [],
    }
    manifest_path = tmp_path / "manifest.json"
    scenario_path = tmp_path / "scenario.json"
    manifest_path.write_text(json.dumps(manifest))
    scenario_path.write_text(json.dumps(scenario))
    return manifest_path, scenario_path, manifest, scenario


def test_valid_specialist_handoff(tmp_path):
    manifest_path, scenario_path, _, _ = _inputs(tmp_path)
    assert validate(manifest_path, scenario_path)["valid"] is True


def test_rejects_artifact_hash_mismatch(tmp_path):
    manifest_path, scenario_path, manifest, _ = _inputs(tmp_path)
    manifest["policies"][0]["sha256"]["onnx"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sha256 mismatch for onnx"):
        validate(manifest_path, scenario_path)


def test_rejects_incomplete_policy_evidence(tmp_path):
    manifest_path, scenario_path, manifest, _ = _inputs(tmp_path)
    del manifest["policies"][0]["artifacts"]["evaluation_report"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing artifact evaluation_report"):
        validate(manifest_path, scenario_path)


def test_rejects_unaccepted_or_positive_penalty_evidence(tmp_path):
    manifest_path, scenario_path, manifest, _ = _inputs(tmp_path)
    policy = manifest["policies"][0]
    evaluation_path = Path(policy["artifacts"]["evaluation_report"])
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["accepted"] = False
    evaluation["penalty_terms"]["action_rate"] = 0.1
    evaluation_path.write_text(json.dumps(evaluation))
    policy["sha256"]["evaluation_report"] = hashlib.sha256(
        evaluation_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="accepted must be true"):
        validate(manifest_path, scenario_path)


def test_rejects_parity_or_metadata_abi_mismatch(tmp_path):
    manifest_path, scenario_path, manifest, _ = _inputs(tmp_path)
    policy = manifest["policies"][0]
    parity_path = Path(policy["artifacts"]["parity_report"])
    parity_path.write_text(json.dumps({"passed": False, "observation_dim": 71, "action_dim": 14}))
    policy["sha256"]["parity_report"] = hashlib.sha256(parity_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parity_report.passed must be true"):
        validate(manifest_path, scenario_path)


def test_rejects_non_increasing_scenario_times(tmp_path):
    manifest_path, scenario_path, _, scenario = _inputs(tmp_path)
    scenario["transitions"].append({
        "at_s": 0,
        "from": "velstand_flat",
        "to": "velstand_flat",
        "min_dwell_s": 1,
        "non_interruptible_s": 0,
        "expected_outcome": {"metric": "upright", "operator": "gte", "threshold": 0.8},
    })
    scenario_path.write_text(json.dumps(scenario))
    with pytest.raises(ValueError, match="at_s must increase"):
        validate(manifest_path, scenario_path)


def test_validates_unsupported_policy_ids_without_compiling_them(tmp_path):
    manifest_path, scenario_path, _, scenario = _inputs(tmp_path)
    scenario["unsupported_transitions"] = [{
        "from": "velstand_flat",
        "to": "unknown_policy",
        "reason": "not part of the frozen inventory",
    }]
    scenario_path.write_text(json.dumps(scenario))
    with pytest.raises(ValueError, match=r"unsupported_transitions\[0\] references unknown policy"):
        validate(manifest_path, scenario_path)


def test_rejects_unknown_executable_policy_even_with_legacy_reason(tmp_path):
    manifest_path, scenario_path, _, scenario = _inputs(tmp_path)
    scenario["transitions"][0]["to"] = "unknown_policy"
    scenario["transitions"][0]["unsupported_reason"] = "legacy bypass"
    scenario_path.write_text(json.dumps(scenario))
    with pytest.raises(ValueError, match=r"transitions\[0\] references unknown policy"):
        validate(manifest_path, scenario_path)
