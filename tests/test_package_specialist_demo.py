import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "package_specialist_demo",
    Path(__file__).parents[1] / "scripts" / "package_specialist_demo.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
package = _MODULE.package


def _inputs(tmp_path: Path, *, policies: int = 2):
    source_root = tmp_path / "prepared"
    entries = []
    for index in range(policies):
        policy_id = f"policy_{index}"
        source = source_root / policy_id
        source.mkdir(parents=True)
        checkpoint = source / "checkpoint.pt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        (source / "policy.onnx").write_bytes(f"onnx-{index}".encode())
        (source / "diagnostic_video.mp4").write_bytes(f"video-{index}".encode())
        (source / "metadata.json").write_text(json.dumps({
            "observation_dim": 61,
            "action_dim": 14,
            "source_commit": "facd4f4",
            "image_digest": "sha256:" + "a" * 64,
            "export_command": f"uv run scripts/export.py Task{index} --checkpoint model.pt",
        }))
        (source / "evaluation_report.json").write_text(json.dumps({
            "accepted": True,
            "finite": True,
            "success_rate": 0.9,
            "main_task_metric": 1.0,
            "penalty_terms": {"action_rate": -0.1},
            "video_review": "Expected behavior observed.",
        }))
        (source / "parity_report.json").write_text(json.dumps({
            "passed": True, "observation_dim": 61, "action_dim": 14,
        }))
        entries.append({
            "id": policy_id,
            "task": f"Task{index}",
            "checkpoint": f"/immutable/model-{index}.pt",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "acceptance_report": f"/immutable/evaluation-{index}.json",
            "accepted": True,
        })
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "schema_version": 1,
        "accepted_count": policies,
        "expected_count": policies,
        "source_inventory": "prior.json",
        "entries": entries,
    }))
    return inventory, source_root


def test_packages_complete_inventory_and_records_provenance(tmp_path):
    inventory, source_root = _inputs(tmp_path)
    output = tmp_path / "artifacts"

    result = package(inventory, output, source_root=source_root)

    manifest_path = output / "specialist_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert result["policies"] == 2
    assert result["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert manifest["source_inventory_sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()
    assert len(manifest["manifest_payload_sha256"]) == 64
    policy = manifest["policies"][0]
    assert policy["provenance"]["source_commit"] == "facd4f4"
    assert policy["provenance"]["source_inventory_entry"]["checkpoint"].startswith("/immutable/")
    for name, path in policy["artifacts"].items():
        artifact = Path(path)
        assert artifact.is_file()
        assert policy["sha256"][name] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_rejects_checkpoint_not_matching_frozen_inventory_without_output(tmp_path):
    inventory, source_root = _inputs(tmp_path)
    (source_root / "policy_0" / "checkpoint.pt").write_bytes(b"replacement")
    output = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="checkpoint sha256"):
        package(inventory, output, source_root=source_root)

    assert not (output / "specialist_artifact_manifest.json").exists()
    assert not (output / "specialists").exists()


@pytest.mark.parametrize(
    ("file_name", "replacement", "message"),
    [
        ("metadata.json", {"observation_dim": 71, "action_dim": 14}, "observation_dim"),
        ("evaluation_report.json", {"accepted": False}, "accepted and finite"),
        ("parity_report.json", {"passed": False}, "passed must be true"),
    ],
)
def test_rejects_invalid_evidence_contract(tmp_path, file_name, replacement, message):
    inventory, source_root = _inputs(tmp_path)
    (source_root / "policy_0" / file_name).write_text(json.dumps(replacement))

    with pytest.raises(ValueError, match=message):
        package(inventory, tmp_path / "artifacts", source_root=source_root)


def test_accepts_explicit_per_policy_sources_and_rejects_partial_set(tmp_path):
    inventory, source_root = _inputs(tmp_path)
    source = source_root / "policy_0"

    with pytest.raises(ValueError, match="policy_1: no prepared source directory"):
        package(inventory, tmp_path / "artifacts", policy_sources=[f"policy_0={source}"])


def test_refuses_to_overwrite_existing_evidence(tmp_path):
    inventory, source_root = _inputs(tmp_path, policies=1)
    output = tmp_path / "artifacts"
    package(inventory, output, source_root=source_root)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        package(inventory, output, source_root=source_root)
