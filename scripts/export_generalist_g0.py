#!/usr/bin/env python3
"""Export a trained G0 BC actor and produce a deterministic ONNX parity report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.generalist_model import build_actor
from mjlab_microduck.generalist_temporal import H4Actor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(run: Path):
    import torch
    manifest = json.loads((run / "manifest.json").read_text())
    metrics = manifest.get("metrics", manifest)
    model = H4Actor(bounded=metrics.get("bounded_actions", True)) if manifest.get("schema") == "generalist-g0-h4" else build_actor(metrics)
    bundle = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    state = bundle.get("state_dict", bundle)
    model.load_state_dict(state, strict=True)
    model.eval()
    metrics = dict(metrics)
    metrics.setdefault("input_dim", manifest.get("input_dim", 71))
    metrics.setdefault("schema", manifest.get("schema"))
    return model, metrics


def export_g0(run: Path, onnx_path: Path, golden_path: Path, parity_path: Path,
              *, seed: int = 20260829, samples: int = 32, atol: float = 1e-5,
              rtol: float = 1e-4) -> dict:
    import torch
    if samples < 2:
        raise ValueError("samples must be at least 2")
    model, metrics = _load(run)
    rng = np.random.default_rng(seed)
    input_dim = int(metrics.get("input_dim", 71))
    inputs = rng.standard_normal((samples, input_dim), dtype=np.float32)
    # Include explicit one-hot behavior cases and zero command fields.
    if input_dim == 71:
        inputs[:, 48:54] = 0
        for i in range(min(3, samples)):
            inputs[i, 48 + i] = 1
    with torch.inference_mode():
        expected = model(torch.from_numpy(inputs)).numpy().astype(np.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.from_numpy(inputs[:1]), str(onnx_path),
                      input_names=["observations"], output_names=["actions"],
                      dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
                      opset_version=17)
    if metrics.get("schema") == "generalist-g0-h4":
        import onnx
        exported = onnx.load(str(onnx_path))
        onnx.helper.set_model_props(exported, {
            "schema": "generalist-g0-h4", "schema_version": "1",
            "input_dim": "215", "action_dim": "14", "history_frames": "4",
            "proprioception_dim": "48", "condition_dim": "23",
            "frame_order": "oldest_to_newest", "padding": "repeat_first_frame",
            "segment_reset": "true", "control_hz": "50",
        })
        onnx.save(exported, str(onnx_path))
    np.savez_compressed(golden_path, observations=inputs, pt_actions=expected,
                        case_ids=np.asarray([f"sample:{i}" for i in range(samples)]))
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = np.asarray(session.run(["actions"], {"observations": inputs})[0], dtype=np.float32)
    error = np.abs(actual - expected)
    passed = bool(np.all(error <= atol + rtol * np.abs(expected)))
    report = {"schema_version": 1, "passed": passed, "input_dim": input_dim, "action_dim": 14,
              "samples": samples, "seed": seed, "atol": atol, "rtol": rtol,
              "max_abs_error": float(error.max(initial=0.0)),
              "onnx_sha256": _sha256(onnx_path), "golden": str(golden_path),
              "model_kind": metrics.get("model_kind", "dense"),
              "architecture": metrics.get("architecture")}
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"ONNX parity failed (max_abs_error={report['max_abs_error']})")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(export_g0(args.run, args.onnx, args.golden, args.parity_report,
                               seed=args.seed, samples=args.samples), indent=2))


if __name__ == "__main__":
    main()
