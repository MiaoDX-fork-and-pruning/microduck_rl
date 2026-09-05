#!/usr/bin/env python3
"""Export one merged graph from accepted frozen stand and locomotion teachers."""
import argparse, json
from pathlib import Path
import torch
from torch import nn
from mjlab_microduck.generalist_model import RoutedG0TeacherActor, build_actor

def load_teacher(root, name):
    payload = torch.load(root / name / "checkpoint.pt", map_location="cpu", weights_only=False)["actor_state_dict"]
    layers = nn.Sequential(nn.Linear(61, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 14))
    with torch.no_grad():
        for i in (0, 2, 4, 6): layers[i].weight.copy_(payload[f"mlp.{i}.weight"]); layers[i].bias.copy_(payload[f"mlp.{i}.bias"])
    mean, std = payload["obs_normalizer._mean"].reshape(1, -1), payload["obs_normalizer._std"].reshape(1, -1) + .01
    class Normalized(nn.Module):
        def forward(self, x): return layers((x - mean) / std)
    return Normalized().eval()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--specialists", type=Path, default=Path("artifacts/specialists")); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    names = ("velstand_flat", "velocity_flat", "sitstand_flat", "ground_pick_flat", "ball_kick_flat", "roulade_flat")
    model = RoutedG0TeacherActor(*(load_teacher(args.specialists, name) for name in names)); args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 71), str(args.output), input_names=["observations"], output_names=["actions"], dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17)
    print(json.dumps({"output": str(args.output), "input_dim": 71, "action_dim": 14, "routing": {name: 48 + i for i, name in enumerate(("stand", "locomotion", "sit_stand", "ground_pick", "kick", "roulade"))}}, indent=2))
if __name__ == "__main__": main()
