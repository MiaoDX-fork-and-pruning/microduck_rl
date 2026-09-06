#!/usr/bin/env python3
"""Collect immutable specialist traces and train the first walk BC baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.generalist_schema import SCHEMA, SCHEMA_VERSION, make_conditioned_observation, validate_batch
from mjlab_microduck.generalist_model import G0MultiHeadActor, GatedAdapterG0Actor


def collect(trace_root: Path, behaviors: tuple[str, ...] = ("stand", "locomotion", "sit_stand")) -> tuple[np.ndarray, np.ndarray, dict]:
    sources = (("stand", "velstand_flat"), ("locomotion", "velocity_flat"), ("sit_stand", "sitstand_flat"))
    xs, ys, manifest_sources = [], [], []
    for behavior, name in sources:
        if behavior not in behaviors:
            continue
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
    validate_dataset(x, y)
    return x, y, {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "sources": manifest_sources,
                  "behavior_counts": behavior_counts(x)}


def behavior_counts(x: np.ndarray) -> dict[str, int]:
    """Return counts for the three G0 labels, including explicit zeroes."""
    labels = np.asarray(x)[:, 48:54].argmax(axis=1)
    return {name: int(np.sum(labels == i)) for i, name in enumerate(("stand", "locomotion", "sit_stand"))}


def dense_architecture(*, small: bool = False, capacity_2x: bool = False,
                       capacity_4x: bool = False) -> list[int]:
    """Return the explicit shared dense capacity arm and reject ambiguity."""
    if sum((small, capacity_2x, capacity_4x)) > 1:
        raise ValueError("choose exactly one dense architecture arm")
    if capacity_4x:
        return [71, 1088, 544, 272, 14]
    if capacity_2x:
        return [71, 768, 384, 192, 14]
    if small:
        return [71, 256, 256, 14]
    return [71, 512, 256, 128, 14]


def validate_dataset(x: np.ndarray, y: np.ndarray) -> None:
    """Validate the immutable 71D/14D dataset and its active G0 labels."""
    validate_batch(np.asarray(x), np.asarray(y))
    labels = np.asarray(x)[:, 48:54]
    if not np.allclose(labels.sum(axis=1), 1.0) or not np.isin(labels.argmax(axis=1), (0, 1, 2)).all():
        raise ValueError("dataset contains invalid or out-of-scope G0 behavior labels")


def balanced_indices(labels: np.ndarray, seed: int = 0, bucket_labels: np.ndarray | None = None) -> np.ndarray:
    """Deterministically oversample each behavior (and optional bucket) equally."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty vector")
    keys = labels if bucket_labels is None else np.stack((labels, np.asarray(bucket_labels)), axis=1)
    groups = [np.flatnonzero(np.all(keys == key, axis=1)) for key in np.unique(keys, axis=0)] if keys.ndim == 2 else [np.flatnonzero(keys == key) for key in np.unique(keys)]
    target = max(len(group) for group in groups)
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(group, target, replace=len(group) < target) for group in groups])
    return selected[rng.permutation(len(selected))]


def train(x: np.ndarray, y: np.ndarray, out: Path, epochs: int, seed: int, balance: bool = True, init_checkpoint: Path | None = None, small: bool = False, bounded: bool = False, multihead: bool = False, gated_adapter: bool = False, capacity_2x: bool = False, capacity_4x: bool = False, trajectory_ids: np.ndarray | None = None, bucket_labels: np.ndarray | None = None) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    validate_dataset(x, y)
    labels = x[:, 48:54].argmax(axis=1)
    # Split whole trajectories first. Balancing before this point leaks near-
    # duplicate frames across train/validation and invalidates closed-loop
    # generalization evidence.
    if trajectory_ids is None:
        trajectory_ids = np.arange(len(x), dtype=np.int64)
    trajectory_ids = np.asarray(trajectory_ids)
    if trajectory_ids.shape != (len(x),):
        raise ValueError("trajectory_ids must align with samples")
    unique = np.unique(trajectory_ids)
    rng = np.random.default_rng(seed)
    train_trajectories = set(rng.choice(unique, max(1, int(len(unique) * 0.9)), replace=False).tolist())
    train_mask = np.isin(trajectory_ids, list(train_trajectories))
    val_mask = ~train_mask
    if not val_mask.any() and len(unique) > 1:
        moved = next(iter(train_trajectories - {min(train_trajectories)}))
        train_mask[trajectory_ids == moved] = False
        val_mask = ~train_mask
    train_base = np.flatnonzero(train_mask)
    val_idx = torch.from_numpy(np.flatnonzero(val_mask).astype(np.int64))
    if balance:
        train_bucket = None if bucket_labels is None else np.asarray(bucket_labels)[train_base]
        local = balanced_indices(labels[train_base], seed=seed, bucket_labels=train_bucket)
        train_idx = torch.from_numpy(train_base[local].astype(np.int64))
    else:
        train_idx = torch.from_numpy(train_base[rng.permutation(len(train_base))].astype(np.int64))
    if (multihead or gated_adapter) and (small or capacity_2x or capacity_4x):
        raise ValueError("capacity flags apply only to the shared dense actor")
    if multihead and gated_adapter:
        raise ValueError("choose one conditioned actor variant")
    architecture = dense_architecture(small=small, capacity_2x=capacity_2x, capacity_4x=capacity_4x)
    if init_checkpoint and (capacity_2x or capacity_4x):
        raise ValueError("capacity ablations do not support specialist checkpoint initialization")
    if init_checkpoint and gated_adapter:
        raise ValueError("gated-adapter actor does not support specialist checkpoint initialization")
    model = (
        G0MultiHeadActor(bounded=bounded) if multihead else
        GatedAdapterG0Actor(bounded=bounded) if gated_adapter else
        nn.Sequential(*[layer for index, (source, target) in enumerate(zip(architecture, architecture[1:])) for layer in ((nn.Linear(source, target),) if index == len(architecture) - 2 else (nn.Linear(source, target), nn.Tanh()))])
    )
    if bounded and not multihead and not gated_adapter:
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
        for index, name in enumerate(("stand", "locomotion", "sit_stand")):
            selected = val_idx.numpy()[labels[val_idx.numpy()] == index]
            if len(selected):
                per_behavior[name] = float(((model(tx[selected]) - ty[selected]) ** 2).mean().item())
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "state_dict": model.state_dict()}, out / "model.pt")
    model_hash = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    return {"train_mse": float(loss.item()), "validation_mse": val,
            "validation_mse_by_behavior": per_behavior, "samples": len(x),
            "seed": seed, "model_sha256": model_hash,
            "architecture": [71, 256, 256, 14] if gated_adapter else architecture, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "model_kind": "g0_multihead" if multihead else "gated_adapter" if gated_adapter else "dense", "bounded_actions": bounded, "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "hidden_dim": 256 if gated_adapter else None, "adapter_dim": 32 if gated_adapter else None,
            "behavior_count": 3 if gated_adapter else None,
            "trajectory_split": True, "train_trajectories": len(train_trajectories), "validation_trajectories": len(unique) - len(train_trajectories)}


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
    ap.add_argument("--gated-adapter", action="store_true", help="use the shared trunk/gated residual adapter actor")
    ap.add_argument("--capacity-4x", action="store_true", help="use the explicit larger shared dense actor ablation")
    ap.add_argument("--capacity-2x", action="store_true", help="use the approximately 2x shared dense actor ablation")
    ap.add_argument("--behavior", choices=("stand", "locomotion", "sit_stand"), default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, manifest = collect(args.trace_root, (args.behavior,) if args.behavior else ("stand", "locomotion", "sit_stand"))
    if args.extra_data:
        with np.load(args.extra_data, allow_pickle=False) as extra:
            validate_batch(extra["inputs"], extra["actions"])
            x = np.concatenate((x, extra["inputs"])); y = np.concatenate((y, extra["actions"]))
            manifest["extra_data"] = str(args.extra_data)
    np.savez_compressed(args.output / "dataset.npz", inputs=x, actions=y)
    manifest.update({"samples": len(x), "input_dim": 71, "action_dim": 14, "seed": args.seed})
    metrics = train(x, y, args.output, args.epochs, args.seed, balance=not args.no_balance, init_checkpoint=args.init_checkpoint, small=args.small_model, bounded=args.bounded_actions, multihead=args.multihead, gated_adapter=args.gated_adapter, capacity_2x=args.capacity_2x, capacity_4x=args.capacity_4x)
    manifest["metrics"] = metrics
    (args.output / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
