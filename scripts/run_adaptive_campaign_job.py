#!/usr/bin/env python3
"""Run one matched-budget branch/seed, export its final state, and evaluate held-out."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

TASKS = {
    "feedback": ("Mjlab-Velocity-Flat-Adaptive-Feedback-MicroDuck", "composed"),
    "acquisition-feedback": ("Mjlab-Velocity-Flat-Adaptive-AcquisitionFeedback-MicroDuck", "composed"),
    "lateral-drive": ("Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck", "composed"),
    "fixed": ("Mjlab-Velocity-Flat-MicroDuck", "all_static"),
    "all-static": ("Mjlab-Velocity-Flat-Adaptive-Static-MicroDuck", "all_static"),
    "com": ("Mjlab-Velocity-Flat-Adaptive-CoM-MicroDuck", "com"),
    "head-com": ("Mjlab-Velocity-Flat-Adaptive-HeadCoM-MicroDuck", "head_com"),
    "composed": ("Mjlab-Velocity-Flat-Adaptive-MicroDuck", "composed"),
}


def read_training_result(path: Path, *, task_id: str, completed_iterations: int,
                         start_iterations: int, num_envs: int) -> tuple[Path, dict]:
    """Read an explicit completion manifest; never guess the last file label."""
    result = json.loads(path.read_text())
    if (result.get("version") != 1 or result.get("status") != "completed"
            or result.get("task_id") != task_id
            or result.get("completed_iterations") != completed_iterations
            or result.get("start_completed_iterations") != start_iterations
            or result.get("num_envs") != num_envs):
        raise ValueError("training result task or budget mismatch")
    checkpoint = Path(result["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result["checkpoint_sha256"]:
        raise ValueError("training result checkpoint hash mismatch")
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = (saved.get("infos") or {}).get("adaptive_curriculum")
    if (saved["iter"] != completed_iterations - 1 or not state
            or state != result["adaptive_state"]
            or state["completed_iterations"] != completed_iterations):
        raise ValueError("training result does not match checkpoint state")
    return checkpoint, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4000, help="Total completed PPO updates, including a resumed prefix")
    parser.add_argument("--resume", type=Path, help="Exact checkpoint path; supported by adaptive/static experiment runners")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--gate-interval", type=int, default=250)
    parser.add_argument(
        "--gate-cohort-size", type=int, default=3,
        help="Fixed native reset/DR seeds per adaptive gate evaluation (default: 3)",
    )
    parser.add_argument("--gate-distribution", choices=("final", "stage"), default=None,
                        help="Inherit on resume; fresh adaptive runs use checkpointed CoM stages")
    parser.add_argument("--rebaseline-gate", action="store_true",
                        help="Explicitly discard old gate evidence when changing distribution on resume")
    parser.add_argument("--final-com-fraction", type=float, default=None,
                        help="Final-CoM rehearsal fraction (0..0.20); defaults to checkpoint value or zero")
    parser.add_argument("--transition-probability", type=float, default=None,
                        help="Initial forward-to-yaw transition probability (0..0.40); defaults to checkpoint value or zero")
    parser.add_argument("--transition-mode", choices=("forward", "zero"), default=None,
                        help="Transition bootstrap mode; defaults to checkpoint mode or forward")
    parser.add_argument("--gate-seed", type=int, default=20260815)
    parser.add_argument("--heldout-seed", type=int, default=20260915)
    args = parser.parse_args()
    gate_consumed_seeds = set(range(args.gate_seed, args.gate_seed + args.gate_cohort_size + 5))
    heldout_consumed_seeds = set(range(args.heldout_seed, args.heldout_seed + 6))
    if gate_consumed_seeds & heldout_consumed_seeds:
        parser.error("gate cohort and held-out six-bucket seed sets overlap")
    if args.iterations < 1 or args.num_envs < 1 or args.gate_interval < 1 or args.gate_cohort_size < 1:
        parser.error("iterations, num-envs, gate-interval and gate-cohort-size must be positive")
    if args.resume and args.branch == "fixed":
        parser.error("the canonical fixed runner has no audited resume budget; use an adaptive/static branch")
    task_id, axis_mode = TASKS[args.branch]
    saved_gate_distribution = "final" if axis_mode == "all_static" else "stage"
    transition_override_requested = args.transition_probability is not None
    start_iterations = 0
    saved_final_com_fraction = 0.0
    saved_sensor_reset_fraction = 0.0
    saved_action_rate_relief_scope = "all"
    saved_transition_probability = 0.0
    saved_transition_mode = "forward"
    transition_mode_override_requested = args.transition_mode is not None
    if args.resume:
        import torch

        args.resume = args.resume.resolve(strict=True)
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = (resume_state.get("infos") or {}).get("adaptive_curriculum")
        if not state or state.get("task_id") != task_id:
            parser.error("resume requires a runner checkpoint with a matching task and completed-update budget")
        start_iterations = int(state["completed_iterations"])
        saved_gate_distribution = state.get("evaluation_distribution", "final")
        saved_fraction = state.get("final_com_fraction", 0.0)
        saved_final_com_fraction = 0.0 if saved_fraction is None else float(saved_fraction)
        saved_sensor_fraction = state.get("sensor_reset_fraction", 0.0)
        saved_relief = state.get("action_rate_relief")
        if isinstance(saved_relief, dict):
            saved_action_rate_relief_scope = saved_relief.get("scope", "all")
        saved_sensor_reset_fraction = 0.0 if saved_sensor_fraction is None else float(saved_sensor_fraction)
        if not 0.0 <= saved_sensor_reset_fraction <= 1.0:
            parser.error("checkpoint sensor reset fraction must be in [0, 1]")
        transition_state = state.get("transition_exposure") or {}
        saved_transition_probability = float(transition_state.get("probability", 0.0))
        saved_transition_mode = str(state.get("transition_bootstrap_mode", "forward"))
        if state.get("num_envs") != args.num_envs:
            parser.error("resume must retain num-envs for comparable cumulative transition accounting")
        if state.get("evaluation_seed") is not None and state["evaluation_seed"] != args.gate_seed:
            parser.error("resume must retain the checkpoint's gate seed")
        if args.iterations <= start_iterations:
            parser.error("iterations must exceed the checkpoint's completed update count")
    if args.gate_distribution is None:
        args.gate_distribution = saved_gate_distribution
    if args.gate_distribution not in ("final", "stage"):
        parser.error("checkpoint evaluation distribution must be final or stage")
    if args.resume and args.gate_distribution != saved_gate_distribution and not args.rebaseline_gate:
        parser.error("changing gate distribution requires --rebaseline-gate")
    if axis_mode == "all_static" and args.gate_distribution != "final":
        parser.error("stage gate requires an adaptive axis")
    if args.final_com_fraction is None:
        args.final_com_fraction = saved_final_com_fraction
    if not 0.0 <= args.final_com_fraction <= 0.20:
        parser.error("final CoM rehearsal fraction must be in [0, 0.20]")
    if args.transition_probability is None:
        args.transition_probability = saved_transition_probability
    if args.transition_mode is None:
        args.transition_mode = saved_transition_mode
    if args.transition_mode not in ("forward", "zero"):
        parser.error("transition mode must be forward or zero")
    if not 0.0 <= args.transition_probability <= 0.40:
        parser.error("transition acquisition probability must be in [0, 0.40]")
    if args.transition_probability and args.branch not in (
        "feedback", "acquisition-feedback", "lateral-drive"
    ):
        parser.error("transition acquisition requires a command-exposure branch")
    if args.final_com_fraction and axis_mode == "all_static":
        parser.error("final CoM rehearsal requires an adaptive CoM axis")
    source_sha = os.environ["MICRODUCK_SOURCE_SHA"]
    source = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    smoke_dir = output / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    training = output / "training"
    if training.exists():
        raise FileExistsError(f"refusing to mix runs in {training}")
    training.mkdir(parents=True)
    adaptive = axis_mode != "all_static"
    environment = dict(os.environ)
    environment.pop("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", None)
    environment.pop("MICRODUCK_ADAPTIVE_RESULT_FILE", None)
    environment.pop("MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE", None)
    environment.pop("MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE", None)
    environment.setdefault("MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE", saved_action_rate_relief_scope)
    if environment["MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE"] not in ("all", "pure_yaw"):
        parser.error("action-rate relief scope must be all or pure_yaw")
    if args.resume:
        # The checkpoint owns this training-distribution setting. Recreate it
        # before the environment is constructed, even when the shell that
        # launched the continuation has no sensor-reset variable.
        environment.pop("MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION", None)
        environment["MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION"] = str(
            saved_sensor_reset_fraction
        )
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL": str(args.gate_interval if adaptive else 0),
        "MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION": str(args.final_com_fraction),
        "MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY": str(args.transition_probability),
        "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE": args.transition_mode,
        "MICRODUCK_ADAPTIVE_EVALUATION_SEED": str(args.gate_seed),
        "MICRODUCK_ADAPTIVE_EVALUATION_COHORT_SIZE": str(args.gate_cohort_size),
        "MICRODUCK_ADAPTIVE_EVALUATION_DISTRIBUTION": args.gate_distribution,
        "MICRODUCK_ADAPTIVE_ALLOW_DISTRIBUTION_MIGRATION": "1" if args.rebaseline_gate else "0",
        "MICRODUCK_ADAPTIVE_ALLOW_LEGACY_COHORT_MIGRATION": (
            "1" if args.resume and args.gate_cohort_size > 1 else "0"
        ),
        "MICRODUCK_ADAPTIVE_SEED_SET_ID": f"adaptive-gate-{args.gate_seed}",
        "MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND": (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(source / 'scripts/run_adaptive_native_cohort_battery.py'))} "
            "--checkpoint {checkpoint} --task {task_id} --axis-mode {axis_mode} "
            "--evaluation-seed {evaluation_seed} --cohort-size {cohort_size} "
            "--distribution {distribution} "
            "--seed-set-id {seed_set_id} --output {output}"
        ),
    })
    if transition_override_requested:
        # The initial value lets a fresh run construct the command term.  On a
        # resume, the runner first restores the checkpoint's exact exposure
        # state, then applies this explicit CLI override.
        environment["MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE"] = str(
            args.transition_probability
        )
    if transition_mode_override_requested:
        environment["MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE"] = "1"
    config = {
        **vars(args),
        "resume": str(args.resume) if args.resume else None,
        "start_completed_iterations": start_iterations,
        "output": str(output),
        "task_id": task_id,
        "axis_mode": axis_mode,
        "source_sha": source_sha,
        "action_rate_relief_scope": environment["MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE"],
        "sensor_reset_fraction": environment.get(
            "MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION",
            str(saved_sensor_reset_fraction),
        ),
    }
    (output / "campaign-config.json").write_text(json.dumps(config, indent=2) + "\n")
    train = str(Path(sys.executable).parent / "train")
    for phase, envs, iterations in (("smoke", 64, 5), ("training", args.num_envs, args.iterations - start_iterations)):
        env = dict(environment)
        env["MICRODUCK_ADAPTIVE_RESULT_FILE"] = str(output / f"{phase}-result.json")
        if phase == "smoke":
            env["MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL"] = "0"
        elif args.resume:
            env["MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT"] = str(args.resume)
        command = [train, task_id, "--env.scene.num-envs", str(envs),
                   "--agent.max-iterations", str(iterations), "--agent.seed", str(args.seed),
                   "--agent.logger", "tensorboard", "--agent.experiment-name", f"matched_{args.branch}",
                   "--agent.run-name", f"matched-{args.branch}-s{args.seed}"]
        print(f"Starting {phase}: {command}", flush=True)
        with (output / f"{phase}.log").open("w") as log:
            subprocess.run(command, env=env, cwd=(training if phase == "training" else output / "smoke"), check=True, stdout=log, stderr=subprocess.STDOUT)
    if args.branch == "fixed":
        # Canonical baseline has no adaptive runner/result contract and cannot
        # resume here. This is artifact collection inside a fresh, unique run.
        checkpoints = list(training.rglob(f"model_{args.iterations - 1}.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"expected one fixed baseline checkpoint, found {checkpoints}")
        checkpoint, adaptive_state = checkpoints[0], None
    else:
        checkpoint, adaptive_state = read_training_result(
            output / "training-result.json", task_id=task_id,
            completed_iterations=args.iterations, start_iterations=start_iterations,
            num_envs=args.num_envs,
        )
        if float(adaptive_state.get("final_com_fraction", 0.0)) != args.final_com_fraction:
            raise ValueError("training result lost the configured final CoM rehearsal fraction")
        if adaptive_state.get("evaluation_distribution", "final") != args.gate_distribution:
            raise ValueError("training result lost the configured gate distribution")
        transition_state = adaptive_state.get("transition_exposure")
        if args.transition_probability > 0.0:
            if not isinstance(transition_state, dict):
                raise ValueError("training result lost transition exposure state")
            final_transition_probability = float(transition_state.get("probability", -1.0))
            if not 0.0 <= final_transition_probability <= 0.40:
                raise ValueError("training result contains invalid transition probability")
        if adaptive and not any(event["kind"] in ("hold", "advance", "regress", "preservation_failure")
                                for event in adaptive_state["evaluation_events"]):
            raise ValueError("adaptive run produced no valid evaluation windows")
    # Training may use launch-only transition overrides, but native and CPU
    # evaluators explicitly disable transition acquisition and must not inherit
    # those training-only settings. Passing the override through makes the
    # evaluator construct a non-exposure command term and then reject the
    # override before it can produce a behavioral report.
    evaluation_environment = dict(environment)
    for name in (
        "MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY",
        "MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE",
        "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE",
        "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE",
        "MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT",
        # Sensor-corner coverage is a training-only augmentation.  Frozen
        # native and CPU reports must use the product DR distribution.
        "MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION",
        "MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION",
        "MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE",
        "MICRODUCK_ADAPTIVE_EVALUATION_DISTRIBUTION",
        "MICRODUCK_ADAPTIVE_ALLOW_DISTRIBUTION_MIGRATION",
    ):
        evaluation_environment.pop(name, None)
    report_path = output / "heldout" / "capability.json"
    with (output / "heldout.log").open("w") as log:
        subprocess.run([
            sys.executable, str(source / "scripts/run_adaptive_native_checkpoint_battery.py"),
            "--checkpoint", str(checkpoint), "--task-id", task_id, "--axis-mode", axis_mode,
            "--evaluation-seed", str(args.heldout_seed),
            "--seed-set-id", f"adaptive-default-{args.heldout_seed}", "--output", str(report_path),
        ], env=evaluation_environment, check=True, stdout=log, stderr=subprocess.STDOUT)
    report = json.loads(report_path.read_text())
    transfer_path = output / "cpu-transfer" / "capability.json"
    with (output / "cpu-transfer.log").open("w") as log:
        subprocess.run([
            sys.executable, str(source / "scripts/run_adaptive_checkpoint_battery.py"),
            "--checkpoint", str(checkpoint), "--task-id", task_id, "--axis-mode", axis_mode,
            "--evaluation-seed", str(args.heldout_seed),
            "--seed-set-id", f"adaptive-default-{args.heldout_seed}", "--output", str(transfer_path),
        ], env=evaluation_environment, check=True, stdout=log, stderr=subprocess.STDOUT)
    transfer = json.loads(transfer_path.read_text())
    result = {
        **config,
        "status": "evaluated", "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "segment_training_transitions": (args.iterations - start_iterations) * 24 * args.num_envs,
        "total_training_transitions": args.iterations * 24 * args.num_envs,
        "adaptive_state": adaptive_state,
        "heldout_report": str(report_path),
        "heldout_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "heldout_aggregate": report["aggregate"],
        "cpu_transfer_report": str(transfer_path),
        "cpu_transfer_report_sha256": hashlib.sha256(transfer_path.read_bytes()).hexdigest(),
        "cpu_transfer_aggregate": transfer["aggregate"],
    }
    (output / "campaign-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": report["aggregate"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
