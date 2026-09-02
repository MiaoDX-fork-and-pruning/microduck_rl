#!/usr/bin/env python3
"""Validate the immutable specialist fallback contract used by G0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_generalist_teachers import TEACHERS, verify

REQUIRED_ARTIFACTS = ("checkpoint", "onnx", "metadata", "evaluation_report", "parity_report")


def validate_fallback(manifest_path: Path, *, expected_source_commit: str | None = None) -> dict:
    """Validate frozen specialists and return a compact ABI report.

    This deliberately checks the fallback manifest itself, rather than the
    generalist model, so changing the student cannot silently alter specialist
    identity, dimensions, or deployment artifacts.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "generalist-g0-teacher-manifest" or manifest.get("version") != 1:
        raise ValueError("invalid frozen specialist manifest schema")
    contract = manifest.get("contract", {})
    if contract.get("observation_dim") != 61 or contract.get("action_dim") != 14:
        raise ValueError("specialist ABI must remain 61D observations and 14D actions")
    teachers = manifest.get("teachers", [])
    if [p.get("id") for p in teachers] != list(TEACHERS):
        raise ValueError("frozen fallback must contain the three specialists in canonical order")
    for policy in teachers:
        if policy.get("observation_dim") != 61 or policy.get("action_dim") != 14:
            raise ValueError(f"specialist ABI mismatch: {policy.get('id')}")
        if set(policy.get("artifacts", {})) != set(REQUIRED_ARTIFACTS):
            raise ValueError(f"specialist artifact set mismatch: {policy.get('id')}")
        if expected_source_commit is not None and policy.get("source_commit") != expected_source_commit:
            raise ValueError(f"specialist source commit changed: {policy.get('id')}")
    verify(manifest)
    return {"valid": True, "preserved": True, "specialists": list(TEACHERS),
            "observation_dim": 61, "action_dim": 14}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    print(json.dumps(validate_fallback(args.manifest, expected_source_commit=args.source_commit), indent=2))


if __name__ == "__main__":
    main()
