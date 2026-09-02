import importlib.util
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location(
    "benchmark_generalist_g0", Path(__file__).parents[1] / "scripts/benchmark_generalist_g0.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_benchmark_reports_budget_and_pytorch_metrics():
    model = torch.nn.Sequential(torch.nn.Linear(71, 14), torch.nn.Tanh())
    report = MODULE.benchmark(model, warmup=1, iterations=3)
    assert report["input_dim"] == 71
    assert report["action_dim"] == 14
    assert report["budget_ms"] == 15.0
    assert set(("p50_ms", "p95_ms", "min_ms", "max_ms")) <= report["pytorch"].keys()
    assert report["passed"] is True


def test_benchmark_can_fail_explicit_budget(monkeypatch):
    model = torch.nn.Identity()
    monkeypatch.setattr(MODULE.time, "perf_counter", iter([0.0, 1.0, 2.0, 3.0]).__next__)
    report = MODULE.benchmark(model, warmup=0, iterations=2, control_period_ms=2.0, margin_ms=1.0)
    assert report["passed"] is False
