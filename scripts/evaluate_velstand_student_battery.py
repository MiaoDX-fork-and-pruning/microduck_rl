#!/usr/bin/env python3
"""Run the manifest-sized VELSTAND student battery and fail closed on missing metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import mujoco, numpy as np, torch
from mjlab_microduck.generalist_model import build_actor
from mjlab_microduck.generalist_schema import make_conditioned_observation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_generalist_bc import run_case

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--teacher-onnx', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ticks', type=int, default=120)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    bundle = torch.load(args.run / 'model.pt', weights_only=False)
    meta = json.loads((args.run / 'manifest.json').read_text()).get('metrics', {})
    student = build_actor(meta); student.load_state_dict(bundle['state_dict']); student.eval()
    model = mujoco.MjModel.from_xml_path('src/mjlab_microduck/robot/microduck/scene.xml'); model.opt.timestep = .005
    episodes = []
    for index in range(args.episodes):
        np.random.seed(args.seed + index)
        result = run_case(model, student, args.teacher_onnx, 'stand', 0.0, args.ticks)
        result['episode'] = index; result['seed'] = args.seed + index
        episodes.append(result)
    success_rate = sum(bool(e['passed']) for e in episodes) / len(episodes)
    report = {'schema': 'generalist-g0-velstand-student-battery', 'seed': args.seed,
              'episodes': episodes, 'success_rate': success_rate,
              'main_task_metric': None,
              'acceptance': {'success_rate': success_rate >= .8,
                             'main_task_metric': False,
                             'finite': all(e['finite'] for e in episodes),
                             'accepted': False},
              'note': 'Student harness does not expose specialist reward terms; main-task gate is intentionally fail-closed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0)
if __name__ == '__main__': main()
