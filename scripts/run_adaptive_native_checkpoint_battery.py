#!/usr/bin/env python3
"""Expose the native schema-v2 evaluator to the runner command interface."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from run_adaptive_native_battery import run_native


def run_checkpoint(checkpoint: Path, output: Path, *, task_id: str, axis_mode: str,
                   evaluation_seed: int, seed_set_id: str, source_sha: str = "unknown",
                   steps: int = 300, zero_mode: str = "nominal") -> dict:
    output = output.resolve()
    payload = run_native(checkpoint, output.parent / f"{output.stem}.native",
        task=task_id, seed=evaluation_seed, steps=steps,
        device=os.environ.get("MICRODUCK_NATIVE_DEVICE", "cuda:0"),
        seed_set_id=seed_set_id, axis_mode=axis_mode, source_sha=source_sha,
        zero_mode=zero_mode)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--axis-mode", choices=("all_static", "com", "head_com", "composed"), required=True)
    p.add_argument("--evaluation-seed", type=int, required=True)
    p.add_argument("--seed-set-id", required=True)
    p.add_argument("--source-sha", default=os.environ.get("MICRODUCK_SOURCE_SHA", "unknown"))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--zero-mode", choices=("nominal", "push"), default="nominal")
    payload = run_checkpoint(**vars(p.parse_args()))
    print(json.dumps(payload["aggregate"], indent=2))
    return 0  # Negative capability is data; nonzero exits are execution errors.


if __name__ == "__main__":
    raise SystemExit(main())
