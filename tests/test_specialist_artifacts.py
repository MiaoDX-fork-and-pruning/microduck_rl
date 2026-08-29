import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "validate_specialist_artifacts",
    Path(__file__).parents[1] / "scripts" / "validate_specialist_artifacts.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate = _MODULE.validate


def _inputs(tmp_path):
    checkpoint = tmp_path / "model.pt"
    onnx = tmp_path / "policy.onnx"
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"onnx")
    manifest = {
        "policies": [{
            "id": "stand",
            "observation_dim": 61,
            "artifacts": {"checkpoint": str(checkpoint), "onnx": str(onnx)},
            "sha256": {
                "checkpoint": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "onnx": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            },
        }]
    }
    scenario = {
        "duration_s": 10,
        "command_rate_hz": 50,
        "transitions": [{
            "at_s": 0,
            "from": "stand",
            "to": "stand",
            "expected_outcome": "upright",
        }],
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


def test_rejects_non_increasing_scenario_times(tmp_path):
    manifest_path, scenario_path, _, scenario = _inputs(tmp_path)
    scenario["transitions"].append({
        "at_s": 0,
        "from": "stand",
        "to": "stand",
        "expected_outcome": "upright",
    })
    scenario_path.write_text(json.dumps(scenario))
    with pytest.raises(ValueError, match="at_s must increase"):
        validate(manifest_path, scenario_path)
