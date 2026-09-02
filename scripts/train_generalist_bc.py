#!/usr/bin/env python3
"""Collect immutable specialist traces and train the first walk BC baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.generalist_schema import SCHEMA, SCHEMA_VERSION, make_conditioned_observation, validate_batch
from mjlab_microduck.generalist_model import G0MultiHeadActor


def collect(trace_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    sources = (("stand", "velstand_flat"), ("locomotion", "velocity_flat"))
    xs, ys, manifest_sources = [], [], []
    for behavior, name in sources:
        paths = sorted((trace_root / name).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no traces for {name} under {trace_root}")
        for path in paths:
            data = np.load(path)
            # Traces and exported ONNX use raw v1 observations; normalization is
            # already baked into each teacher ONNX graph.
            x = make_conditioned_observation(data["observation"], data["requested_command"], behavior)
            y = np.asarray(data["raw_action"], dtype=np.float32)
            validate_batch(x, y)
            xs.append(x)
            ys.append(y)
            manifest_sources.append(str(path))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return x, y, {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "sources": manifest_sources}


def train(x: np.ndarray, y: np.ndarray, out: Path, epochs: int, seed: int, balance: bool = True, init_checkpoint: Path | None = None, small: bool = False, bounded: bool = False, multihead: bool = False) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    labels = x[:, 48:54].argmax(axis=1)
    if balance:
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(labels == i) for i in (0, 1)]
        target = max(len(g) for g in groups)
        balanced = np.concatenate([rng.choice(g, target, replace=len(g) < target) for g in groups])
        order = torch.from_numpy(balanced[rng.permutation(len(balanced))].astype(np.int64))
    else:
        order = torch.randperm(len(x))
    split = max(1, int(len(x) * 0.9))
    train_idx, val_idx = order[:split], order[split:]
    model = G0MultiHeadActor(bounded=bounded) if multihead else (nn.Sequential(nn.Linear(71, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 14)) if small else nn.Sequential(nn.Linear(71, 512), nn.Tanh(), nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 14)))
    if bounded and not multihead:
        model.add_module("output_tanh", nn.Tanh())
    if init_checkpoint:
        source = torch.load(init_checkpoint, weights_only=False)["actor_state_dict"]
        with torch.no_grad():
            model[0].weight[:, :48] = source["mlp.0.weight"][:, :48]
            model[0].weight[:, 54:67] = source["mlp.0.weight"][:, 48:61]
            model[0].bias.copy_(source["mlp.0.bias"])
            for dst, key in ((2, "mlp.2"), (4, "mlp.4"), (6, "mlp.6")):
                model[dst].weight.copy_(source[key + ".weight"]); model[dst].bias.copy_(source[key + ".bias"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        pred = model(tx[train_idx])
        loss = ((pred - ty[train_idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        val_pred = model(tx[val_idx]) if len(val_idx) else torch.empty((0, 14))
        val = ((val_pred - ty[val_idx]) ** 2).mean().item() if len(val_idx) else float("nan")
        labels = x[:, 48:54].argmax(axis=1)
        per_behavior = {}
        for index, name in enumerate(("stand", "locomotion")):
            selected = val_idx.numpy()[labels[val_idx.numpy()] == index]
            if len(selected):
                per_behavior[name] = float(((model(tx[selected]) - ty[selected]) ** 2).mean().item())
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "state_dict": model.state_dict()}, out / "model.pt")
    model_hash = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    return {"train_mse": float(loss.item()), "validation_mse": val,
            "validation_mse_by_behavior": per_behavior, "samples": len(x),
            "seed": seed, "model_sha256": model_hash,
            "architecture": [71, 256, 256, 14] if small else [71, 512, 256, 128, 14], "model_kind": "g0_multihead" if multihead else "dense", "bounded_actions": bounded, "init_checkpoint": str(init_checkpoint) if init_checkpoint else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", type=Path, default=Path("artifacts/generalist-v0/p0-action-battery-deadband-final"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/generalist-v0/p2-walk-bc-smoke"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extra-data", type=Path, default=None)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--init-checkpoint", type=Path, default=None)
    ap.add_argument("--small-model", action="store_true")
    ap.add_argument("--bounded-actions", action="store_true")
    ap.add_argument("--multihead", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, manifest = collect(args.trace_root)
    if args.extra_data:
        with np.load(args.extra_data, allow_pickle=False) as extra:
            validate_batch(extra["inputs"], extra["actions"])
            x = np.concatenate((x, extra["inputs"])); y = np.concatenate((y, extra["actions"]))
            manifest["extra_data"] = str(args.extra_data)
    np.savez_compressed(args.output / "dataset.npz", inputs=x, actions=y)
    manifest.update({"samples": len(x), "input_dim": 71, "action_dim": 14, "seed": args.seed})
    metrics = train(x, y, args.output, args.epochs, args.seed, balance=not args.no_balance, init_checkpoint=args.init_checkpoint, small=args.small_model, bounded=args.bounded_actions, multihead=args.multihead)
    manifest["metrics"] = metrics
    (args.output / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
