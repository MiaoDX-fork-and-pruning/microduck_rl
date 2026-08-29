#!/usr/bin/env python3
"""Compare an exported ONNX policy with deterministic PyTorch golden actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

OBS_DIM = 61
ACTION_DIM = 14
REQUIRED_CASE_PREFIXES = ("zero_command", "command_extreme")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_actions(
    expected: np.ndarray,
    actual: np.ndarray,
    case_ids: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    case_ids = np.asarray(case_ids)
    if expected.ndim != 2 or expected.shape[1] != ACTION_DIM:
        raise ValueError(f"pt_actions must have shape [N, {ACTION_DIM}]")
    if actual.shape != expected.shape:
        raise ValueError(f"onnx_actions shape {actual.shape} != pt_actions {expected.shape}")
    if case_ids.ndim != 1 or len(case_ids) != len(expected):
        raise ValueError("case_ids must have shape [N]")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("policy actions must be finite")
    labels = [str(value) for value in case_ids.tolist()]
    for prefix in REQUIRED_CASE_PREFIXES:
        if not any(label == prefix or label.startswith(prefix + ":") for label in labels):
            raise ValueError(f"case_ids must include {prefix!r}")

    abs_error = np.abs(actual - expected)
    tolerance = atol + rtol * np.abs(expected)
    passing = abs_error <= tolerance
    cases = []
    for index, label in enumerate(labels):
        cases.append({
            "id": label,
            "max_abs_error": float(abs_error[index].max()),
            "mean_abs_error": float(abs_error[index].mean()),
            "passed": bool(passing[index].all()),
        })
    return {
        "samples": len(expected),
        "action_dim": ACTION_DIM,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(abs_error.max(initial=0.0)),
        "mean_abs_error": float(abs_error.mean()) if abs_error.size else 0.0,
        "failed_values": int((~passing).sum()),
        "passed": bool(passing.all()),
        "cases": cases,
    }


def run_comparison(
    onnx_path: Path,
    golden_path: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import onnxruntime as ort

    with np.load(golden_path, allow_pickle=False) as golden:
        required = {"observations", "pt_actions", "case_ids"}
        missing = required - set(golden.files)
        if missing:
            raise ValueError(f"golden dataset missing: {', '.join(sorted(missing))}")
        observations = np.asarray(golden["observations"], dtype=np.float32)
        expected = np.asarray(golden["pt_actions"], dtype=np.float32)
        case_ids = golden["case_ids"]
    if observations.ndim != 2 or observations.shape[1] != OBS_DIM:
        raise ValueError(f"observations must have shape [N, {OBS_DIM}]")
    if len(observations) != len(expected):
        raise ValueError("observations and pt_actions must have the same sample count")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("ONNX policy must have exactly one input and one output")
    actual = np.concatenate([
        session.run([outputs[0].name], {inputs[0].name: observation[None, :]})[0]
        for observation in observations
    ])
    report = compare_actions(expected, actual, case_ids, atol=atol, rtol=rtol)
    report.update({
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "golden": str(golden_path),
        "golden_sha256": sha256(golden_path),
        "observation_dim": OBS_DIM,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("golden", type=Path, help="NPZ with observations, pt_actions, case_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    try:
        report = run_comparison(args.onnx, args.golden, atol=args.atol, rtol=args.rtol)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
