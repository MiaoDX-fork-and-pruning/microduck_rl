#!/usr/bin/env python3
"""Validate a G0 BC artifact and emit reproducible PPO comparison metadata.

This is deliberately a preparation step: the BC ``state_dict`` is not an
rsl_rl checkpoint and must not be passed to ``--agent.load-checkpoint``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

from mjlab_microduck.generalist_model import build_actor
from mjlab_microduck.generalist_schema import ACTION_DIM, OBS_DIM, SCHEMA, SCHEMA_VERSION


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bc_artifact(root: Path) -> dict:
    manifest_path, model_path = root / "manifest.json", root / "model.pt"
    if not manifest_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"BC artifact requires {manifest_path} and {model_path}")
    metadata = json.loads(manifest_path.read_text())
    if metadata.get("schema") != SCHEMA or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("BC artifact schema does not match the G0 contract")
    if metadata.get("input_dim") not in (None, OBS_DIM) or metadata.get("action_dim") not in (None, ACTION_DIM):
        raise ValueError("BC artifact dimensions do not match 71D/14D G0 contract")
    import torch
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("BC checkpoint is missing state_dict")
    actor = build_actor(metadata)
    try:
        actor.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"BC checkpoint architecture does not match manifest: {exc}") from exc
    return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "model_kind": metadata.get("model_kind", "dense"),
            "architecture": metadata.get("architecture"),
            "bounded_actions": bool(metadata.get("bounded_actions", False)),
            "model": str(model_path.resolve()), "model_sha256": sha256(model_path)}


def prepare(bc_root: Path, output: Path, num_envs: int = 4096, iterations: int = 20000) -> dict:
    bc = validate_bc_artifact(bc_root)
    base = ["uv", "run", "train", "Mjlab-GeneralistG0-Flat-MicroDuck",
            "--env.scene.num-envs", str(num_envs), "--agent.max_iterations", str(iterations)]
    return {"schema": "generalist-g0-ppo-bridge", "version": 1, "bc": bc,
            "direct": {"experiment_name": "generalist_g0_direct_ppo", "initialization": "random",
                        "command": shlex.join(base + ["--agent.run_name", "direct_ppo"])},
            "hybrid": {"experiment_name": "generalist_g0_hybrid_ppo", "initialization": "bc_actor",
                        "bc_actor": bc["model"],
                        "command": shlex.join(base + ["--agent.run_name", "hybrid_ppo"])},
            "note": "Hybrid BC actor injection is performed by the PPO runner integration; do not pass model.pt to --agent.load-checkpoint."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()
    result = prepare(args.bc, args.output, args.num_envs, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
