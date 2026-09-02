#!/usr/bin/env python3
"""Independent offline evaluator for a generalist-v0 BC checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    import torch

    data = np.load(args.run / "dataset.npz", allow_pickle=False)
    x, y = data["inputs"].astype(np.float32), data["actions"].astype(np.float32)
    bundle = torch.load(args.run / "model.pt", weights_only=False)
    model = torch.nn.Sequential(torch.nn.Linear(71, 512), torch.nn.Tanh(), torch.nn.Linear(512, 256), torch.nn.Tanh(), torch.nn.Linear(256, 128), torch.nn.Tanh(), torch.nn.Linear(128, 14))
    model.load_state_dict(bundle["state_dict"]); model.eval()
    with torch.no_grad(): pred = model(torch.from_numpy(x)).numpy()
    err = (pred - y) ** 2
    labels = x[:, 48:54].argmax(axis=1)
    report = {"schema": bundle.get("schema"), "samples": len(x), "finite": bool(np.isfinite(pred).all()),
              "max_abs_action": float(np.abs(pred).max()), "outside_unit_range": int((np.abs(pred) > 1).sum()),
              "mse": float(err.mean()), "mse_by_behavior": {}}
    for i, name in enumerate(("stand", "locomotion")):
        mask = labels == i
        report["mse_by_behavior"][name] = float(err[mask].mean())
    print(json.dumps(report, indent=2))
    if not report["finite"] or report["outside_unit_range"]:
        raise SystemExit("invalid BC output")


if __name__ == "__main__":
    main()
