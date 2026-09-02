#!/usr/bin/env python3
"""Validate and stage prepared specialist-demo artifacts.

This command is deliberately offline.  It neither downloads checkpoints nor
exports policies; each source directory must already contain the six canonical
files listed in ``SOURCE_FILES``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


SOURCE_FILES = {
    "checkpoint": "checkpoint.pt",
    "onnx": "policy.onnx",
    "metadata": "metadata.json",
    "evaluation_report": "evaluation_report.json",
    "diagnostic_video": "diagnostic_video.mp4",
    "parity_report": "parity_report.json",
}
OBSERVATION_DIM = 61
ACTION_DIM = 14


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_evidence(policy_id: str, files: dict[str, Path]) -> dict[str, Any]:
    metadata = _read_object(files["metadata"], f"{policy_id} metadata")
    evaluation = _read_object(files["evaluation_report"], f"{policy_id} evaluation report")
    # Remediated acceptance reports wrap battery fields under ``evaluation``;
    # CloudML reports historically emitted the same fields at the top level.
    if isinstance(evaluation.get("evaluation"), dict):
        nested = dict(evaluation["evaluation"])
        nested.setdefault("accepted", evaluation.get("accepted"))
        nested.setdefault("finite", nested.get("finite", evaluation.get("finite")))
        nested.setdefault("video_review", evaluation.get("video", {}).get("review", ""))
        evaluation = nested
    parity = _read_object(files["parity_report"], f"{policy_id} parity report")

    if metadata.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError(f"{policy_id}: metadata.observation_dim must be 61")
    if metadata.get("action_dim") != ACTION_DIM:
        raise ValueError(f"{policy_id}: metadata.action_dim must be 14")
    for field in ("source_commit", "image_digest", "export_command"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"{policy_id}: metadata.{field} must be non-empty")

    if evaluation.get("accepted") is not True or evaluation.get("finite") is not True:
        raise ValueError(f"{policy_id}: evaluation must be accepted and finite")
    rate = evaluation.get("success_rate")
    if not _number(rate) or not 0 <= rate <= 1:
        raise ValueError(f"{policy_id}: evaluation.success_rate must be in [0, 1]")
    if not _number(evaluation.get("main_task_metric")):
        raise ValueError(f"{policy_id}: evaluation.main_task_metric must be numeric")
    penalties = evaluation.get("penalty_terms")
    if not isinstance(penalties, dict) or any(not _number(v) or v > 0 for v in penalties.values()):
        raise ValueError(f"{policy_id}: evaluation penalties must be numeric and <= 0")
    if not isinstance(evaluation.get("video_review"), str) or not evaluation["video_review"].strip():
        raise ValueError(f"{policy_id}: evaluation.video_review must be non-empty")

    if parity.get("passed") is not True:
        raise ValueError(f"{policy_id}: parity_report.passed must be true")
    if parity.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError(f"{policy_id}: parity_report.observation_dim must be 61")
    if parity.get("action_dim") != ACTION_DIM:
        raise ValueError(f"{policy_id}: parity_report.action_dim must be 14")
    return metadata


def _source_map(source_root: Path | None, policy_sources: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in policy_sources:
        policy_id, separator, raw_path = item.partition("=")
        if not separator or not policy_id or not raw_path:
            raise ValueError("--policy-source must be POLICY_ID=DIR")
        if policy_id in result:
            raise ValueError(f"duplicate source for policy {policy_id}")
        result[policy_id] = Path(raw_path)
    if source_root is not None:
        result["*"] = source_root
    return result


def package(
    inventory_path: Path,
    output_root: Path,
    *,
    source_root: Path | None = None,
    policy_sources: list[str] | None = None,
) -> dict[str, Any]:
    inventory = _read_object(inventory_path, "inventory")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("inventory.entries must be a non-empty list")
    if inventory.get("accepted_count") != len(entries) or inventory.get("expected_count") != len(entries):
        raise ValueError("inventory counts must match entries")

    sources = _source_map(source_root, policy_sources or [])
    expected_ids = {entry.get("id") for entry in entries}
    explicit_ids = set(sources) - {"*"}
    unknown = explicit_ids - expected_ids
    if unknown:
        raise ValueError(f"sources contain unknown policies: {', '.join(sorted(unknown))}")

    prepared: list[tuple[dict[str, Any], dict[str, Path], dict[str, Any]]] = []
    seen: set[str] = set()
    for entry in entries:
        policy_id = entry.get("id")
        if not isinstance(policy_id, str) or not policy_id or policy_id in seen:
            raise ValueError(f"duplicate or missing inventory policy id: {policy_id!r}")
        seen.add(policy_id)
        if entry.get("accepted") is not True:
            raise ValueError(f"{policy_id}: inventory entry is not accepted")
        source_dir = sources.get(policy_id)
        if source_dir is None and source_root is not None:
            source_dir = source_root / policy_id
        if source_dir is None:
            raise ValueError(f"{policy_id}: no prepared source directory")
        files = {name: source_dir / filename for name, filename in SOURCE_FILES.items()}
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise ValueError(f"{policy_id}: missing prepared artifacts: {', '.join(missing)}")
        checkpoint_hash = sha256(files["checkpoint"])
        if checkpoint_hash != entry.get("checkpoint_sha256"):
            raise ValueError(f"{policy_id}: checkpoint sha256 does not match frozen inventory")
        metadata = _validate_evidence(policy_id, files)
        prepared.append((entry, files, metadata))

    specialists_root = output_root / "specialists"
    manifest_path = output_root / "specialist_artifact_manifest.json"
    occupied = [specialists_root / policy_id for policy_id in seen if (specialists_root / policy_id).exists()]
    if manifest_path.exists() or occupied:
        raise ValueError("output already exists; refusing to overwrite staged evidence")

    output_root.mkdir(parents=True, exist_ok=True)
    specialists_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".specialist-package-", dir=output_root) as raw_tmp:
        tmp_root = Path(raw_tmp)
        policies = []
        for entry, files, metadata in prepared:
            policy_id = entry["id"]
            tmp_policy = tmp_root / policy_id
            tmp_policy.mkdir()
            artifact_paths: dict[str, str] = {}
            hashes: dict[str, str] = {}
            for name, source in files.items():
                destination = tmp_policy / SOURCE_FILES[name]
                shutil.copy2(source, destination)
                artifact_paths[name] = str(specialists_root / policy_id / SOURCE_FILES[name])
                hashes[name] = sha256(destination)
            policies.append({
                "id": policy_id,
                "task": entry["task"],
                "accepted": True,
                "observation_dim": OBSERVATION_DIM,
                "action_dim": ACTION_DIM,
                "artifacts": artifact_paths,
                "sha256": hashes,
                "provenance": {
                    "source_inventory_entry": entry,
                    "source_directory": str(files["checkpoint"].parent.resolve()),
                    "source_commit": metadata["source_commit"],
                    "image_digest": metadata["image_digest"],
                    "export_command": metadata["export_command"],
                },
            })

        manifest: dict[str, Any] = {
            "version": 1,
            "source_inventory": str(inventory_path),
            "source_inventory_sha256": sha256(inventory_path),
            "policies": policies,
        }
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        manifest["manifest_payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        tmp_manifest = tmp_root / manifest_path.name
        tmp_manifest.write_text(manifest_text, encoding="utf-8")
        for policy_id in seen:
            os.replace(tmp_root / policy_id, specialists_root / policy_id)
        os.replace(tmp_manifest, manifest_path)

    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "policies": len(policies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path(
        "cloudml/specialist-final-checkpoints-remediated-facd4f4.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--policy-source", action="append", default=[], metavar="POLICY_ID=DIR")
    args = parser.parse_args()
    if args.source_root is None and not args.policy_source:
        parser.error("provide --source-root or at least one --policy-source")
    try:
        result = package(args.inventory, args.output_root, source_root=args.source_root,
                         policy_sources=args.policy_source)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
