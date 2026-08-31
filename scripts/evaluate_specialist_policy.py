#!/usr/bin/env python3
"""Evaluate one specialist checkpoint in a deterministic MuJoCo battery."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


_PENALTY_MARKERS = (
    "collision",
    "fallen",
    "impact",
    "limit",
    "overshoot",
    "penalty",
    "tax",
)


def is_penalty_term(name: str, weight: float, explicit: Iterable[str] = ()) -> bool:
    """Recognize stock costs and Microduck's positive-weight self-negating costs."""
    lowered = name.lower()
    return (
        name in explicit
        or weight < 0.0
        or lowered.endswith(("_l1", "_l2"))
        or any(marker in lowered for marker in _PENALTY_MARKERS)
    )


def diagnostic_video_path(report_path: Path, requested: Path | None) -> Path:
    return requested if requested is not None else report_path.with_suffix(".mp4")


def episode_is_finite(step_finite: bool, termination_reasons: Iterable[str]) -> bool:
    return step_finite and "nan_state" not in termination_reasons


def build_evaluation_report(
    *,
    task: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    seed: int,
    episodes: list[dict[str, Any]],
    main_task_term: str,
    success_threshold: float,
    minimum_success_rate: float,
    minimum_main_task_metric: float,
    penalty_names: Iterable[str],
    video_path: Path,
    video_review: str,
) -> dict[str, Any]:
    """Build the stable validator-facing report from raw per-episode evidence."""
    if not episodes:
        raise ValueError("episodes must not be empty")

    penalty_names = sorted(set(penalty_names))
    main_values = [float(ep["reward_terms"].get(main_task_term, 0.0)) for ep in episodes]
    successes = [bool(ep.get("success", False)) for ep in episodes]
    finite = all(bool(ep.get("finite", False)) for ep in episodes)
    success_rate = sum(successes) / len(successes)
    main_task_metric = sum(main_values) / len(main_values)
    penalty_terms = {
        name: sum(float(ep["reward_terms"].get(name, 0.0)) for ep in episodes)
        / len(episodes)
        for name in penalty_names
    }
    positive_penalties = {}
    for name in penalty_names:
        positive_values = [
            float(ep["reward_terms"].get(name, 0.0))
            for ep in episodes
            if float(ep["reward_terms"].get(name, 0.0)) > 1.0e-8
        ]
        if positive_values:
            positive_penalties[name] = max(positive_values)
    termination_counts: dict[str, int] = {}
    for episode in episodes:
        for reason in episode.get("termination_reasons", []):
            termination_counts[reason] = termination_counts.get(reason, 0) + 1

    acceptance_checks = {
        "finite": finite,
        "main_task_metric": main_task_metric >= minimum_main_task_metric,
        "penalties_non_positive": not positive_penalties,
        "success_rate": success_rate >= minimum_success_rate,
        "video_reviewed": bool(video_review.strip()),
    }
    return {
        "schema_version": 1,
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "battery": {
            "episodes": len(episodes),
            "main_task_term": main_task_term,
            "success_threshold": success_threshold,
            "minimum_success_rate": minimum_success_rate,
            "minimum_main_task_metric": minimum_main_task_metric,
        },
        "accepted": all(acceptance_checks.values()),
        "finite": finite,
        "success_rate": success_rate,
        "main_task_metric": main_task_metric,
        "penalty_terms": penalty_terms,
        "positive_penalty_terms": positive_penalties,
        "termination_counts": dict(sorted(termination_counts.items())),
        "episode_length_steps": {
            "mean": sum(int(ep["length_steps"]) for ep in episodes) / len(episodes),
            "min": min(int(ep["length_steps"]) for ep in episodes),
            "max": max(int(ep["length_steps"]) for ep in episodes),
        },
        "total_reward_mean": sum(float(ep["total_reward"]) for ep in episodes)
        / len(episodes),
        "diagnostic_video": str(video_path),
        "video_review": video_review,
        "acceptance_checks": acceptance_checks,
        "episodes": episodes,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensors_finite(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if hasattr(value, "values"):
        values = value.values() if callable(value.values) else value.values
        return all(_tensors_finite(item) for item in values)
    return True


def _sim_state_finite(env: Any) -> bool:
    import numpy as np

    data = env.sim.data
    for name in ("qpos", "qvel", "ctrl"):
        value = getattr(data, name, None)
        if value is not None and not np.isfinite(value.numpy()).all():
            return False
    return True


def _write_video(path: Path, frames: list[Any], fps: float) -> None:
    if not frames:
        raise RuntimeError("simulator produced no diagnostic video frames")
    import mediapy

    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    import mjlab.tasks  # noqa: F401

    configure_torch_backends()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        raise ValueError("--minimum-success-rate must be in [0, 1]")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.episodes
    env_cfg.viewer.width = args.video_width
    env_cfg.viewer.height = args.video_height
    env_cfg.viewer.env_idx = 0
    env_cfg.viewer.max_extra_envs = 0

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    reward_names = list(raw_env.reward_manager.active_terms)
    reward_weights = {
        name: float(raw_env.reward_manager.get_term_cfg(name).weight)
        for name in reward_names
    }
    if args.main_task_term not in reward_names:
        raise ValueError(
            f"--main-task-term {args.main_task_term!r} is not active; "
            f"choose one of: {', '.join(reward_names)}"
        )
    unknown_penalties = sorted(set(args.penalty_term) - set(reward_names))
    if unknown_penalties:
        raise ValueError(f"unknown --penalty-term values: {', '.join(unknown_penalties)}")
    penalty_names = [
        name
        for name in reward_names
        if is_penalty_term(name, reward_weights[name], args.penalty_term)
    ]

    observation = env.get_observations()
    active = torch.ones(args.episodes, dtype=torch.bool, device=raw_env.device)
    lengths = [0] * args.episodes
    totals = [0.0] * args.episodes
    term_sums = [{name: 0.0 for name in reward_names} for _ in range(args.episodes)]
    completed: dict[int, dict[str, Any]] = {}
    frames: list[Any] = []
    video_steps = round(args.video_seconds / raw_env.step_dt)
    fatal_reason: str | None = None

    try:
        while active.any() and max(lengths) < raw_env.max_episode_length + 1:
            if not _tensors_finite(observation) or not _sim_state_finite(raw_env):
                fatal_reason = "nonfinite_state"
                break
            with torch.inference_mode():
                action = policy(observation)
            if not _tensors_finite(action):
                fatal_reason = "nonfinite_action"
                break

            observation, reward, dones, _ = env.step(action)
            step_finite = (
                _tensors_finite(observation)
                and _tensors_finite(reward)
                and _sim_state_finite(raw_env)
            )
            if len(frames) < video_steps:
                frame = raw_env.render()
                if frame is not None:
                    frames.append(np.asarray(frame[0] if frame.ndim == 4 else frame))

            scale = raw_env.step_dt if raw_env.cfg.scale_rewards_by_dt else 1.0
            for env_id in active.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
                lengths[env_id] += 1
                totals[env_id] += float(reward[env_id].item())
                step_terms = dict(raw_env.reward_manager.get_active_iterable_terms(env_id))
                for name in reward_names:
                    term_sums[env_id][name] += float(step_terms[name][0]) * scale
                if bool(dones[env_id].item()):
                    reasons = [
                        name
                        for name in raw_env.termination_manager.active_terms
                        if bool(raw_env.termination_manager.get_term(name)[env_id].item())
                    ]
                    episode_finite = episode_is_finite(step_finite, reasons)
                    main_value = term_sums[env_id][args.main_task_term]
                    completed[env_id] = {
                        "id": env_id,
                        "length_steps": lengths[env_id],
                        "length_seconds": lengths[env_id] * raw_env.step_dt,
                        "total_reward": totals[env_id],
                        "reward_terms": term_sums[env_id],
                        "termination_reasons": sorted(reasons),
                        "success": main_value >= args.success_threshold,
                        "finite": episode_finite,
                    }
                    active[env_id] = False
            if not step_finite:
                fatal_reason = "nonfinite_state"
                break
    finally:
        env.close()

    if fatal_reason is not None or active.any():
        reason = fatal_reason or "incomplete_episode"
        for env_id in active.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
            completed[env_id] = {
                "id": env_id,
                "length_steps": lengths[env_id],
                "length_seconds": lengths[env_id] * raw_env.step_dt,
                "total_reward": totals[env_id],
                "reward_terms": term_sums[env_id],
                "termination_reasons": [reason],
                "success": False,
                "finite": False,
            }

    video_path = diagnostic_video_path(args.output, args.video_output).resolve()
    _write_video(video_path, frames, fps=1.0 / raw_env.step_dt)
    minimum_main = (
        args.success_threshold
        if args.minimum_main_task_metric is None
        else args.minimum_main_task_metric
    )
    report = build_evaluation_report(
        task=args.task,
        checkpoint=checkpoint,
        checkpoint_sha256=_sha256(checkpoint),
        seed=args.seed,
        episodes=[completed[index] for index in sorted(completed)],
        main_task_term=args.main_task_term,
        success_threshold=args.success_threshold,
        minimum_success_rate=args.minimum_success_rate,
        minimum_main_task_metric=minimum_main,
        penalty_names=penalty_names,
        video_path=video_path,
        video_review=args.video_review,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="registered mjlab task id")
    parser.add_argument("--checkpoint", type=Path, required=True, help="final model_*.pt")
    parser.add_argument("--output", type=Path, required=True, help="evaluation JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--device", help="torch device (default: cuda:0 when available)")
    parser.add_argument("--main-task-term", required=True, help="active reward term used as the task metric")
    parser.add_argument("--success-threshold", type=float, required=True, help="per-episode main-term threshold")
    parser.add_argument("--minimum-success-rate", type=float, required=True)
    parser.add_argument("--minimum-main-task-metric", type=float, help="mean main-term gate (default: success threshold)")
    parser.add_argument("--penalty-term", action="append", default=[], help="additional self-negating penalty term; repeatable")
    parser.add_argument("--video-output", type=Path, help="default: OUTPUT with .mp4 suffix")
    parser.add_argument("--video-seconds", type=float, default=15.0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-review", default="", help="human clip review; acceptance stays false when empty")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="exit successfully after writing evidence even when acceptance is false",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    finite_thresholds = {
        "--success-threshold": args.success_threshold,
        "--minimum-success-rate": args.minimum_success_rate,
        "--minimum-main-task-metric": args.minimum_main_task_metric,
    }
    for name, value in finite_thresholds.items():
        if value is not None and not math.isfinite(value):
            raise SystemExit(f"{name} must be finite")
    if args.video_seconds <= 0:
        raise SystemExit("--video-seconds must be positive")
    if args.video_width <= 0 or args.video_height <= 0:
        raise SystemExit("--video-width and --video-height must be positive")
    report = run_evaluation(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] or args.report_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
