#!/usr/bin/env python3
"""Verify a captured fall's closed-loop suffix before interpreting recovery."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from capture_specialist_recovery_traces import (
    STATE_SPEC, advance, load_teacher, model_sha256, sha256, trunk_metrics,
)


def replay(trace_path: Path, ticks: int | None = None,
           onnx: Path | None = None, scene: Path | None = None) -> dict:
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    if "metadata" not in trace:
        raise ValueError("legacy trace lacks integration state and replay provenance; recapture it")
    metadata = json.loads(trace["metadata"].item())
    if metadata.get("version") != 2 or metadata.get("state_spec") != int(STATE_SPEC):
        raise ValueError("unsupported recovery trace contract")
    model, data, policy = load_teacher(onnx or Path(metadata["onnx"]), scene or Path(metadata["scene"]))
    if metadata["model_sha256"] != model_sha256(model) or metadata["mujoco_version"] != mujoco.__version__:
        raise ValueError("replay model or MuJoCo version differs from capture")
    chosen_onnx = onnx or Path(metadata["onnx"])
    if metadata["onnx_sha256"] != sha256(chosen_onnx):
        raise ValueError("replay teacher differs from capture")
    falls = np.flatnonzero(trace["tilt_rad"] > math.radians(40))
    if not len(falls):
        raise ValueError("trace contains no fallen state")
    start = int(falls[0]); available = len(trace["tick"]) - start
    count = available if ticks is None else ticks
    if not 1 <= count <= available:
        raise ValueError(f"replay needs 1..{available} ticks covered by the source suffix")
    mujoco.mj_setState(model, data, trace["integration_state"][start], STATE_SPEC)
    policy.last_action = trace["previous_action"][start].copy()
    policy.command = trace["command"][start].copy()
    mujoco.mj_forward(model, data)
    errors = dict.fromkeys(("observation", "raw_action", "next_qpos", "next_qvel"), 0.0)
    finite, max_action, recovered = True, 0.0, False
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    for index in range(start, start + count):
        policy.command = trace["command"][index].copy()
        observation = policy.get_observations(); action = policy.infer(); advance(model, data, policy, action)
        actual = {"observation": observation, "raw_action": action, "next_qpos": data.qpos, "next_qvel": data.qvel}
        for key, value in actual.items():
            if not np.isfinite(value).all() or not np.isfinite(trace[key][index]).all(): finite = False; break
            errors[key] = max(errors[key], float(np.abs(value - trace[key][index]).max()))
        if not finite: break
        max_action = max(max_action, float(np.abs(action).max()))
        metrics = trunk_metrics(model, data, body_id)
        recovered |= metrics["tilt_rad"] < math.radians(25) and metrics["height_m"] > 0.09
    verified = finite and all(value <= 1e-6 for value in errors.values())
    final = trunk_metrics(model, data, body_id)
    return {
        "schema": "specialist-recovery-replay", "version": 2,
        "evidence_class": metadata["evidence_class"], "accepted_reset_equivalence": False,
        "trace": str(trace_path), "trace_sha256": sha256(trace_path), "capture": metadata,
        "source_tick": start, "ticks": count, "finite": finite, "parity_max_abs": errors,
        "replay_verified": verified, "max_abs_raw_action": max_action,
        "raw_action_within_unit_range": max_action <= 1.00001,
        "final_tilt_rad": final["tilt_rad"], "final_height_m": final["height_m"],
        "recovered": bool(recovered) if verified else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--onnx", type=Path, default=None); parser.add_argument("--scene", type=Path, default=None)
    args = parser.parse_args(); report = replay(args.trace, args.ticks, args.onnx, args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["replay_verified"]: raise SystemExit("closed-loop suffix differs from captured trajectory")


if __name__ == "__main__": main()
