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
REQUIRED_ARTIFACTS = {"checkpoint", "onnx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        artifacts = policy.get("artifacts", {})
        missing = REQUIRED_ARTIFACTS - artifacts.keys()
        errors.extend(f"{policy_id}: missing artifact {name}" for name in sorted(missing))
        for name, raw_path in artifacts.items():
            path = Path(raw_path)
            if not path.exists():
                errors.append(f"{policy_id}: artifact does not exist: {path}")
                continue
            expected = policy.get("sha256", {}).get(name)
            if expected and sha256(path) != expected:
                errors.append(f"{policy_id}: sha256 mismatch for {name}")

    if scenario.get("command_rate_hz") != 50:
        errors.append("scenario.command_rate_hz must be 50")
    transitions = scenario.get("transitions", [])
    for index, transition in enumerate(transitions):
        if transition.get("from") not in policy_ids or transition.get("to") not in policy_ids:
            if not transition.get("unsupported_reason"):
                errors.append(f"scenario.transitions[{index}] references unknown policy")
    try:
        frames = compile_scenario(scenario)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError("\n".join(errors))
    return {"manifest": str(manifest_path), "scenario": str(scenario_path),
            "policies": len(policies), "transitions": len(transitions),
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
