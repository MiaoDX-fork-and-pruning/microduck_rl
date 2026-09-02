#!/usr/bin/env python3
"""Benchmark G0 actor inference against the 50 Hz control-period budget."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


CONTROL_PERIOD_MS = 20.0
DEFAULT_MARGIN_MS = 5.0


def _summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {"p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(values.min()), "max_ms": float(values.max())}


def benchmark(model, onnx_path: Path | None = None, *, seed: int = 42,
              warmup: int = 20, iterations: int = 100,
              control_period_ms: float = CONTROL_PERIOD_MS,
              margin_ms: float = DEFAULT_MARGIN_MS) -> dict:
    """Measure single-sample CPU inference and return an auditable report."""
    if iterations < 2 or warmup < 0:
        raise ValueError("iterations must be >= 2 and warmup must be >= 0")
    if margin_ms < 0 or margin_ms >= control_period_ms:
        raise ValueError("margin_ms must be non-negative and below control period")
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((1, 71), dtype=np.float32)
    import torch
    tensor = torch.from_numpy(inputs)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        pt_samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(tensor)
            if hasattr(output, "item"):
                output.reshape(-1)[0].item()
            pt_samples.append((time.perf_counter() - start) * 1000.0)
    result = {"schema_version": 1, "input_dim": 71, "action_dim": 14,
              "seed": seed, "warmup": warmup, "iterations": iterations,
              "control_period_ms": control_period_ms, "margin_ms": margin_ms,
              "budget_ms": control_period_ms - margin_ms,
              "pytorch": _summary(pt_samples)}
    if onnx_path is not None:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        for _ in range(warmup):
            session.run(None, {input_name: inputs})
        ort_samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = session.run(None, {input_name: inputs})[0]
            np.asarray(output).reshape(-1)[0].item()
            ort_samples.append((time.perf_counter() - start) * 1000.0)
        result["onnx"] = _summary(ort_samples)
    result["passed"] = all(result[name]["p95_ms"] <= result["budget_ms"]
                            for name in ("pytorch", "onnx") if name in result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--margin-ms", type=float, default=DEFAULT_MARGIN_MS)
    args = parser.parse_args()
    import torch
    from mjlab_microduck.generalist_model import G0MultiHeadActor
    report = benchmark(G0MultiHeadActor(), args.onnx, warmup=args.warmup,
                       iterations=args.iterations, margin_ms=args.margin_ms)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
