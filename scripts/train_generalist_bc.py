#!/usr/bin/env python3
"""Collect immutable specialist traces and train the first walk BC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.generalist_schema import SCHEMA, SCHEMA_VERSION, make_conditioned_observation, validate_batch


def collect(trace_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    sources = (("stand", "velstand_flat"), ("locomotion", "velocity_flat"))
    xs, ys, manifest_sources = [], [], []
    for behavior, name in sources:
        paths = sorted((trace_root / name).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no traces for {name} under {trace_root}")
        for path in paths:
            data = np.load(path)
            x = make_conditioned_observation(data["observation"], data["requested_command"], behavior)
            y = np.asarray(data["raw_action"], dtype=np.float32)
            validate_batch(x, y)
            xs.append(x)
            ys.append(y)
            manifest_sources.append(str(path))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return x, y, {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "sources": manifest_sources}


def train(x: np.ndarray, y: np.ndarray, out: Path, epochs: int, seed: int) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    order = torch.randperm(len(x))
    split = max(1, int(len(x) * 0.9))
    train_idx, val_idx = order[:split], order[split:]
    model = nn.Sequential(nn.Linear(71, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 14))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        pred = model(tx[train_idx])
        loss = ((pred - ty[train_idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        val = ((model(tx[val_idx]) - ty[val_idx]) ** 2).mean().item() if len(val_idx) else float("nan")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "state_dict": model.state_dict()}, out / "model.pt")
    return {"train_mse": float(loss.item()), "validation_mse": val, "samples": len(x), "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", type=Path, default=Path("artifacts/generalist-v0/p0-action-battery-deadband-final"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/generalist-v0/p2-walk-bc-smoke"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, manifest = collect(args.trace_root)
    np.savez_compressed(args.output / "dataset.npz", inputs=x, actions=y)
    manifest.update({"samples": len(x), "input_dim": 71, "action_dim": 14, "seed": args.seed})
    metrics = train(x, y, args.output, args.epochs, args.seed)
    manifest["metrics"] = metrics
    (args.output / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
