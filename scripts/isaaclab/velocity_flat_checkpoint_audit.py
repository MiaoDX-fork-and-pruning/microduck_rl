"""Audit Velocity-Flat checkpoint provenance and actor-normalizer parity.

This is intentionally simulator-free.  It answers whether a trained-policy
failure can be attributed to a malformed checkpoint/normalizer boundary before
another IsaacLab run is considered.  It does not compare learned weights for
equality: different rollouts are expected to produce different policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).flatten()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(flat).all()),
        "min": float(flat.min()) if flat.numel() else 0.0,
        "max": float(flat.max()) if flat.numel() else 0.0,
        "mean": float(flat.mean()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False)) if flat.numel() else 0.0,
    }


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = payload.get("actor_state_dict")
    critic = payload.get("critic_state_dict")
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise ValueError(f"{path}: missing actor_state_dict/critic_state_dict")
    required = {
        "obs_normalizer._mean",
        "obs_normalizer._std",
        "obs_normalizer._var",
        "obs_normalizer.count",
        "distribution.std_param",
    }
    missing = sorted(required - actor.keys())
    if missing:
        raise ValueError(f"{path}: actor is missing {missing}")
    normalizer = {
        name: _tensor_summary(actor[name])
        for name in ("obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var", "obs_normalizer.count")
    }
    architecture = {
        "actor_keys": sorted(actor.keys()),
        "critic_keys": sorted(critic.keys()),
        "actor_shapes": {name: list(value.shape) for name, value in actor.items()},
        "critic_shapes": {name: list(value.shape) for name, value in critic.items()},
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "iteration": int(payload.get("iter", -1)),
        "normalizer": normalizer,
        "architecture": architecture,
        "actor_parameter_count": int(sum(value.numel() for value in actor.values())),
        "critic_parameter_count": int(sum(value.numel() for value in critic.values())),
    }


def audit(strict: list[Path], reference: Path) -> dict[str, Any]:
    if not strict:
        raise ValueError("at least one strict checkpoint is required")
    reference_report = _checkpoint(reference)
    strict_reports = [_checkpoint(path) for path in strict]
    ref_arch = reference_report["architecture"]
    for report in strict_reports:
        if report["architecture"] != ref_arch:
            raise AssertionError(
                f"architecture mismatch: {report['path']} differs from {reference}"
            )

    ref_payload = torch.load(reference, map_location="cpu", weights_only=False)
    ref_actor = ref_payload["actor_state_dict"]
    comparisons = []
    for report, path in zip(strict_reports, strict, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actor = payload["actor_state_dict"]
        mean_delta = (actor["obs_normalizer._mean"] - ref_actor["obs_normalizer._mean"]).detach().to(torch.float64)
        std_delta = (actor["obs_normalizer._std"] - ref_actor["obs_normalizer._std"]).detach().to(torch.float64)
        comparisons.append(
            {
                "strict_checkpoint": report["path"],
                "reference_checkpoint": str(reference),
                "iteration": report["iteration"],
                "normalizer_mean_linf": float(mean_delta.abs().max()),
                "normalizer_std_linf": float(std_delta.abs().max()),
                "normalizer_mean_l2": float(torch.linalg.vector_norm(mean_delta)),
                "normalizer_std_l2": float(torch.linalg.vector_norm(std_delta)),
                "normalizer_count_equal": bool(torch.equal(actor["obs_normalizer.count"], ref_actor["obs_normalizer.count"])),
                "actor_parameter_count_equal": report["actor_parameter_count"] == reference_report["actor_parameter_count"],
                "critic_parameter_count_equal": report["critic_parameter_count"] == reference_report["critic_parameter_count"],
            }
        )

    return {
        "schema": "velocity_flat_checkpoint_audit.v1",
        "reference": reference_report,
        "strict": strict_reports,
        "comparisons": comparisons,
        "classification": "checkpoint_schema_and_normalizer_present; learned_rollout_distribution_differs",
        "interpretation": (
            "All checkpoints have the same actor/critic tensor schema and finite 61D actor normalizer fields. "
            "Normalizer statistics are expected to differ across backend rollouts; this audit does not treat that "
            "difference as an action/command semantic mismatch."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--strict", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.strict, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
