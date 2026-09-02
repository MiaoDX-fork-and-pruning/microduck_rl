#!/usr/bin/env python3
"""Validate specialist policy manifests and deterministic demo scenarios.

The validator is intentionally offline: training and rollout production happen
on CloudML, while this command makes the handoff auditable before a gallery is
built or a policy is used for a switching demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from specialist_scenario import compile_scenario

EXPECTED_OBS_DIM = 61
EXPECTED_ACTION_DIM = 14
REQUIRED_ARTIFACTS = {
    "checkpoint",
    "onnx",
    "metadata",
    "evaluation_report",
    "diagnostic_video",
    "parity_report",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, policy_id: str, name: str, errors: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{policy_id}: invalid {name}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{policy_id}: {name} must contain a JSON object")
        return {}
    return value


def _validate_evidence(policy_id: str, artifacts: dict, errors: list[str]) -> None:
    evaluation_path = Path(artifacts["evaluation_report"])
    if evaluation_path.exists():
        evaluation = _load_json(evaluation_path, policy_id, "evaluation_report", errors)
        if evaluation.get("accepted") is not True:
            errors.append(f"{policy_id}: evaluation_report.accepted must be true")
        if evaluation.get("finite") is not True:
            errors.append(f"{policy_id}: evaluation_report.finite must be true")
        success_rate = evaluation.get("success_rate")
        if not isinstance(success_rate, (int, float)) or isinstance(success_rate, bool) \
                or not 0.0 <= success_rate <= 1.0:
            errors.append(f"{policy_id}: evaluation_report.success_rate must be in [0, 1]")
        if not isinstance(evaluation.get("main_task_metric"), (int, float)) \
                or isinstance(evaluation.get("main_task_metric"), bool):
            errors.append(f"{policy_id}: evaluation_report.main_task_metric must be numeric")
        penalties = evaluation.get("penalty_terms")
        if not isinstance(penalties, dict) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value > 0
            for value in penalties.values()
        ):
            errors.append(f"{policy_id}: evaluation_report penalties must be numeric and <= 0")
        if not isinstance(evaluation.get("video_review"), str) \
                or not evaluation["video_review"].strip():
            errors.append(f"{policy_id}: evaluation_report.video_review must be non-empty")

    parity_path = Path(artifacts["parity_report"])
    if parity_path.exists():
        parity = _load_json(parity_path, policy_id, "parity_report", errors)
        if parity.get("passed") is not True:
            errors.append(f"{policy_id}: parity_report.passed must be true")
        if parity.get("observation_dim") != EXPECTED_OBS_DIM:
            errors.append(f"{policy_id}: parity_report.observation_dim must be 61")
        if parity.get("action_dim") != EXPECTED_ACTION_DIM:
            errors.append(f"{policy_id}: parity_report.action_dim must be 14")

    metadata_path = Path(artifacts["metadata"])
    if metadata_path.exists():
        metadata = _load_json(metadata_path, policy_id, "metadata", errors)
        if metadata.get("observation_dim") != EXPECTED_OBS_DIM:
            errors.append(f"{policy_id}: metadata.observation_dim must be 61")
        if metadata.get("action_dim") != EXPECTED_ACTION_DIM:
            errors.append(f"{policy_id}: metadata.action_dim must be 14")


def validate(manifest_path: Path, scenario_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    policies = manifest.get("policies", [])
    if not policies:
        errors.append("manifest.policies must contain at least one policy")
    policy_ids = set()
    for policy in policies:
        policy_id = policy.get("id")
        if not policy_id or policy_id in policy_ids:
            errors.append(f"duplicate or missing policy id: {policy_id!r}")
        policy_ids.add(policy_id)
        if policy.get("observation_dim", EXPECTED_OBS_DIM) != EXPECTED_OBS_DIM:
            errors.append(f"{policy_id}: observation_dim must be 61")
        if not isinstance(policy.get("task"), str) or not policy["task"].strip():
            errors.append(f"{policy_id}: task must be non-empty")
        if policy.get("accepted") is not True:
            errors.append(f"{policy_id}: accepted must be true")
        artifacts = policy.get("artifacts", {})
        missing = REQUIRED_ARTIFACTS - artifacts.keys()
        errors.extend(f"{policy_id}: missing artifact {name}" for name in sorted(missing))
        hashes = policy.get("sha256", {})
        for name in sorted(REQUIRED_ARTIFACTS & artifacts.keys()):
            raw_path = artifacts[name]
            if not isinstance(raw_path, str) or not raw_path:
                errors.append(f"{policy_id}: artifact path for {name} must be non-empty")
                continue
            path = Path(raw_path)
            if not path.exists():
                errors.append(f"{policy_id}: artifact does not exist: {path}")
                continue
            expected = hashes.get(name)
            if not isinstance(expected, str) or len(expected) != 64:
                errors.append(f"{policy_id}: missing sha256 for {name}")
            elif sha256(path) != expected:
                errors.append(f"{policy_id}: sha256 mismatch for {name}")
        if not missing:
            _validate_evidence(policy_id, artifacts, errors)

    transitions = scenario.get("transitions", [])
    for index, transition in enumerate(transitions):
        if transition.get("from") not in policy_ids or transition.get("to") not in policy_ids:
            errors.append(f"scenario.transitions[{index}] references unknown policy")
    for index, transition in enumerate(scenario.get("unsupported_transitions", [])):
        if transition.get("from") not in policy_ids or transition.get("to") not in policy_ids:
            errors.append(f"scenario.unsupported_transitions[{index}] references unknown policy")
    try:
        frames = compile_scenario(scenario)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError("\n".join(errors))
    return {"manifest": str(manifest_path), "scenario": str(scenario_path),
            "policies": len(policies), "transitions": len(transitions),
            "unsupported_transitions": len(scenario.get("unsupported_transitions", [])),
            "frames": len(frames), "valid": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.manifest, args.scenario), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
