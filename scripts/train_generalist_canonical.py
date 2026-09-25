#!/usr/bin/env python3
"""Train a single shared G0 actor on exact canonical teacher rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_generalist_bc as bc
from mjlab_microduck.generalist_schema import validate_batch
from mjlab_microduck.generalist_temporal import build_windows, validate_batch as validate_temporal_batch, H4Actor, SCHEMA as H4_SCHEMA, SCHEMA_VERSION as H4_SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gated-adapter", action="store_true")
    parser.add_argument("--film", action="store_true")
    parser.add_argument("--action-adapter", action="store_true")
    parser.add_argument("--onehot-action-adapter", action="store_true",
                        help="use frozen one-hot low-rank action residuals")
    parser.add_argument("--unbounded", action="store_true",
                        help="leave the shared action head unbounded for diagnostics")
    parser.add_argument("--fit-all", action="store_true", help="fit every canonical trajectory; diagnostic only")
    parser.add_argument("--input-noise-std", type=float, default=0.0,
                        help="augment proprioception with deterministic Gaussian noise")
    parser.add_argument("--extra-data", type=Path, action="append", default=[],
                        help="cumulative teacher-labeled student-state shard")
    parser.add_argument("--init-run", type=Path, default=None,
                        help="initialize from a compatible shared actor run")
    parser.add_argument("--critical-ticks", type=int, default=0,
                        help="oversample the first N ticks of every canonical segment")
    parser.add_argument("--temporal-window", action="store_true", help="train the H4 temporal candidate")
    args = parser.parse_args()
    if args.temporal_window and any((args.gated_adapter, args.film, args.action_adapter, args.onehot_action_adapter)):
        raise SystemExit("H4 uses one shared trunk/head; do not select a conditioned adapter")
    if not args.temporal_window and sum((args.gated_adapter, args.film, args.action_adapter, args.onehot_action_adapter)) != 1:
        raise SystemExit("choose exactly one conditioned actor variant")
    if args.critical_ticks < 0:
        raise SystemExit("--critical-ticks must be non-negative")
    if args.input_noise_std < 0 or not np.isfinite(args.input_noise_std):
        raise SystemExit("--input-noise-std must be finite and non-negative")
    with np.load(args.data, allow_pickle=False) as payload:
        x = np.asarray(payload["inputs"], dtype=np.float32)
        y = np.asarray(payload["actions"], dtype=np.float32)
        trajectory_ids = np.asarray(payload.get("trajectory_ids", np.arange(len(x))), dtype=np.int64)
        segment_ids = np.asarray(payload.get("segment_ids", np.arange(len(x))))
    if args.temporal_window:
        x = build_windows(x, segment_ids)
        validate_temporal_batch(x, y)
    else:
        validate_batch(x, y)
    if trajectory_ids.shape != (len(x),) or segment_ids.shape != (len(x),):
        raise ValueError("canonical data metadata must align with samples")
    if args.input_noise_std:
        rng = np.random.default_rng(args.seed)
        noisy = x.copy()
        proprio_width = 192 if args.temporal_window else 48
        noisy[:, :proprio_width] += rng.normal(0.0, args.input_noise_std, noisy[:, :proprio_width].shape).astype(np.float32)
        x = np.concatenate((x, noisy))
        y = np.concatenate((y, y.copy()))
        trajectory_ids = np.concatenate((trajectory_ids, trajectory_ids + trajectory_ids.max() + 1))
        segment_ids = np.concatenate((segment_ids, segment_ids))
    for shard_path in args.extra_data:
        with np.load(shard_path, allow_pickle=False) as shard:
            extra_x = np.asarray(shard["inputs"], dtype=np.float32)
            extra_y = np.asarray(shard["actions"], dtype=np.float32)
            extra_segments = np.asarray(shard.get("segment_ids", np.arange(len(extra_x))))
        if args.temporal_window:
            if extra_x.shape[1] == 71:
                extra_x = build_windows(extra_x, extra_segments)
            validate_temporal_batch(extra_x, extra_y)
        else:
            validate_batch(extra_x, extra_y)
        if extra_segments.shape != (len(extra_x),):
            raise ValueError("extra segment_ids must align with shard samples")
        offset = int(trajectory_ids.max()) + 1 if len(trajectory_ids) else 0
        _, extra_ids = np.unique(extra_segments.astype(str), return_inverse=True)
        extra_ids = extra_ids.astype(np.int64) + offset
        x = np.concatenate((x, extra_x)); y = np.concatenate((y, extra_y))
        trajectory_ids = np.concatenate((trajectory_ids, extra_ids))
        segment_ids = np.concatenate((segment_ids, extra_segments))
    bucket_labels = None
    if args.critical_ticks:
        offsets = np.zeros(len(x), dtype=np.int64)
        for trajectory in np.unique(trajectory_ids):
            selected = np.flatnonzero(trajectory_ids == trajectory)
            offsets[selected] = np.arange(len(selected))
        bucket_labels = (offsets < args.critical_ticks).astype(np.int64)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "dataset.npz", inputs=x, actions=y,
                        trajectory_ids=trajectory_ids, segment_ids=segment_ids)
    if args.temporal_window:
        import torch
        torch.manual_seed(args.seed)
        model = H4Actor(bounded=not args.unbounded)
        tx, ty = torch.from_numpy(x), torch.from_numpy(y)
        unique = np.unique(trajectory_ids)
        rng = np.random.default_rng(args.seed)
        held_out = rng.choice(unique) if len(unique) > 1 and not args.fit_all else None
        train_mask = np.ones(len(x), dtype=bool) if held_out is None else trajectory_ids != held_out
        val_mask = ~train_mask
        train_idx = torch.from_numpy(np.flatnonzero(train_mask))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for _ in range(args.epochs):
            loss = ((model(tx[train_idx]) - ty[train_idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.inference_mode():
            prediction = model(tx)
            squared = (prediction - ty) ** 2
        condition = x[:, 192:198].argmax(axis=1)
        per_behavior = {name: float(squared[condition == index].mean())
                        for index, name in enumerate(("stand", "locomotion", "sit_stand"))
                        if np.any(condition == index)}
        metrics = {"model_kind": "h4", "input_dim": 215, "action_dim": 14,
                   "bounded_actions": not args.unbounded, "train_mse": float(loss.item()),
                   "validation_mse": float(squared[val_mask].mean()) if val_mask.any() else None,
                   "mse_by_behavior": per_behavior,
                   "mse_by_joint": [float(value) for value in squared.mean(dim=0)],
                   "behavior_samples": {name: int(np.sum(condition == index)) for index, name in enumerate(("stand", "locomotion", "sit_stand"))},
                   "held_out_trajectory": int(held_out) if held_out is not None else None,
                   "architecture": [215, 256, 256, 14], "parameter_count": sum(p.numel() for p in model.parameters())}
        torch.save({"state_dict": model.state_dict()}, args.output / "model.pt")
    else:
        metrics = bc.train(
        x, y, args.output, args.epochs, args.seed,
        balance=True, bounded=not args.unbounded, gated_adapter=args.gated_adapter,
        film=args.film, action_adapter=args.action_adapter,
        onehot_action_adapter=args.onehot_action_adapter,
        trajectory_ids=trajectory_ids, bucket_labels=bucket_labels,
        fit_all=args.fit_all, init_model=(args.init_run / "model.pt") if args.init_run else None,
        )
    manifest = {
        "schema": H4_SCHEMA if args.temporal_window else "generalist-v0-canonical-teacher",
        "schema_version": H4_SCHEMA_VERSION if args.temporal_window else 1,
        "input_dim": int(x.shape[1]),
        "data": str(args.data),
        "samples": int(len(x)),
        "trajectories": int(len(np.unique(trajectory_ids))),
        "segments": sorted({str(item) for item in segment_ids}),
        "critical_ticks": args.critical_ticks,
        "input_noise_std": args.input_noise_std,
        "extra_data": [str(path) for path in args.extra_data],
        "init_run": str(args.init_run) if args.init_run else None,
        "model_kind": "h4" if args.temporal_window else "onehot_action_adapter" if args.onehot_action_adapter else "action_adapter" if args.action_adapter else "film" if args.film else "gated_adapter",
        "metrics": metrics,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
