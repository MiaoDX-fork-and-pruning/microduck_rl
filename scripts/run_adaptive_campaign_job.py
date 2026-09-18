#!/usr/bin/env python3
"""Run one matched-budget branch/seed, export its final state, and evaluate held-out."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TASKS = {
    "fixed": ("Mjlab-Velocity-Flat-MicroDuck", "all_static"),
    "all-static": ("Mjlab-Velocity-Flat-Adaptive-Static-MicroDuck", "all_static"),
    "com": ("Mjlab-Velocity-Flat-Adaptive-CoM-MicroDuck", "com"),
    "head-com": ("Mjlab-Velocity-Flat-Adaptive-HeadCoM-MicroDuck", "head_com"),
    "composed": ("Mjlab-Velocity-Flat-Adaptive-MicroDuck", "composed"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--gate-interval", type=int, default=250)
    parser.add_argument("--gate-seed", type=int, default=20260916)
    parser.add_argument("--heldout-seed", type=int, default=20260915)
    args = parser.parse_args()
    if set(range(args.gate_seed, args.gate_seed + 6)) & set(range(args.heldout_seed, args.heldout_seed + 6)):
        parser.error("gate and held-out six-bucket seed sets overlap")
    source_sha = os.environ["MICRODUCK_SOURCE_SHA"]
    source = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    training = output / "training"
    if training.exists():
        raise FileExistsError(f"refusing to mix runs in {training}")
    training.mkdir(parents=True)
    task_id, axis_mode = TASKS[args.branch]
    adaptive = axis_mode != "all_static"
    environment = dict(os.environ)
    environment.pop("MICRODUCK_ADAPTIVE_STAGE_FILE", None)
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL": str(args.gate_interval if adaptive else 0),
        "MICRODUCK_ADAPTIVE_EVALUATION_SEED": str(args.gate_seed),
        "MICRODUCK_ADAPTIVE_SEED_SET_ID": f"adaptive-gate-{args.gate_seed}",
        "MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND": (
            f"{sys.executable} {source}/scripts/run_adaptive_checkpoint_battery.py "
            "--checkpoint {checkpoint} --task-id {task_id} --axis-mode {axis_mode} "
            "--evaluation-seed {evaluation_seed} --seed-set-id {seed_set_id} --output {output}"
        ),
    })
    config = {**vars(args), "output": str(output), "task_id": task_id, "axis_mode": axis_mode, "source_sha": source_sha}
    (output / "campaign-config.json").write_text(json.dumps(config, indent=2) + "\n")
    train = str(Path(sys.executable).parent / "train")
    for phase, envs, iterations in (("smoke", 64, 5), ("training", args.num_envs, args.iterations)):
        env = dict(environment)
        if phase == "smoke":
            env["MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL"] = "0"
        command = [train, task_id, "--env.scene.num-envs", str(envs),
                   "--agent.max-iterations", str(iterations), "--agent.seed", str(args.seed),
                   "--agent.logger", "tensorboard", "--agent.experiment-name", f"matched_{args.branch}",
                   "--agent.run-name", f"matched-{args.branch}-s{args.seed}"]
        print(f"Starting {phase}: {command}", flush=True)
        with (output / f"{phase}.log").open("w") as log:
            subprocess.run(command, env=env, cwd=(training if phase == "training" else output / "smoke"), check=True, stdout=log, stderr=subprocess.STDOUT)
    # Artifact collection only: no glob determines curriculum state or resume.
    checkpoints = [p for p in training.rglob(f"model_{args.iterations - 1}.pt") if ".adaptive" not in p.name]
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one explicit final checkpoint, found {checkpoints}")
    checkpoint = checkpoints[0]
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["iter"] != args.iterations - 1:
        raise ValueError("final checkpoint iteration does not match budget")
    adaptive_state = (state.get("infos") or {}).get("adaptive_curriculum")
    if adaptive:
        if not adaptive_state or adaptive_state["completed_iterations"] != args.iterations:
            raise ValueError("missing adaptive final state or budget mismatch")
        if not adaptive_state["evaluation_events"]:
            raise ValueError("adaptive run produced no evaluator events")
    report_path = output / "heldout" / "capability.json"
    with (output / "heldout.log").open("w") as log:
        subprocess.run([
            sys.executable, str(source / "scripts/run_adaptive_checkpoint_battery.py"),
            "--checkpoint", str(checkpoint), "--task-id", task_id, "--axis-mode", axis_mode,
            "--evaluation-seed", str(args.heldout_seed),
            "--seed-set-id", f"adaptive-default-{args.heldout_seed}", "--output", str(report_path),
        ], env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads(report_path.read_text())
    result = {
        **config,
        "status": "evaluated", "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "total_training_transitions": args.iterations * 24 * args.num_envs,
        "adaptive_state": adaptive_state,
        "heldout_report": str(report_path),
        "heldout_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "heldout_aggregate": report["aggregate"],
    }
    (output / "campaign-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": report["aggregate"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
