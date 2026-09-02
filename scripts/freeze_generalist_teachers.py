#!/usr/bin/env python3
"""Freeze and verify the immutable teacher contract used by G0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TEACHERS = ("velstand_flat", "velocity_flat", "sitstand_flat")
SOURCE_MANIFEST = Path("artifacts/specialist_artifact_manifest.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source: Path = SOURCE_MANIFEST) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in data.get("policies", [])}
    policies = []
    for teacher_id in TEACHERS:
        policy = by_id.get(teacher_id)
        if not policy or not policy.get("accepted"):
            raise ValueError(f"accepted teacher missing: {teacher_id}")
        artifacts = policy["artifacts"]
        metadata = json.loads(Path(artifacts["metadata"]).read_text())
        policies.append({
            "id": teacher_id,
            "task": policy["task"],
            "observation_dim": policy["observation_dim"],
            "action_dim": policy["action_dim"],
            "source_commit": policy["provenance"]["source_commit"],
            "export_command": metadata["export_command"],
            "evaluation_command": "uv run scripts/evaluate_specialist_policy.py",
            "artifacts": {"checkpoint": artifacts["checkpoint"], "onnx": artifacts["onnx"],
                          "metadata": artifacts["metadata"], "evaluation_report": artifacts["evaluation_report"],
                          "parity_report": artifacts["parity_report"]},
            "sha256": {k: policy["sha256"][k] for k in ("checkpoint", "onnx", "metadata", "evaluation_report", "parity_report")},
        })
    return {"schema": "generalist-g0-teacher-manifest", "version": 1,
            "teachers": policies,
            "contract": {"observation_dim": 61, "action_dim": 14, "input_dim": 71,
                          "behavior_order": ["VELSTAND", "VELOCITY", "SITSTAND"],
                          "normalization_baked_into_onnx": True,
                          "legal_edges": ["VELSTAND<->VELOCITY", "VELSTAND<->SITSTAND"]}}


def verify(manifest: dict) -> None:
    if [p["id"] for p in manifest["teachers"]] != list(TEACHERS):
        raise ValueError("teacher order or membership mismatch")
    for policy in manifest["teachers"]:
        for name, raw in policy["artifacts"].items():
            path = Path(raw)
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != policy["sha256"][name]:
                raise ValueError(f"hash mismatch: {policy['id']}:{name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE_MANIFEST)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    manifest = freeze(args.source)
    if args.verify:
        verify(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
