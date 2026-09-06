#!/usr/bin/env python3
"""Capture the exact reset state and manager contract used by a specialist task."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.episodes
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset(seed=args.seed)
        robot = env.scene["robot"]
        root_pose = robot.data.root_link_pose_w.detach().cpu().numpy().astype(np.float32)
        root_vel = robot.data.root_link_vel_w.detach().cpu().numpy().astype(np.float32)
        joint_pos = robot.data.joint_pos.detach().cpu().numpy().astype(np.float32)
        joint_vel = robot.data.joint_vel.detach().cpu().numpy().astype(np.float32)
        twist = env.command_manager.get_term("twist")
        command = twist.vel_command_b.detach().cpu().numpy().astype(np.float32)
        def json_value(value):
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, dict):
                return {str(k): json_value(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [json_value(v) for v in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        term_cfg = {
            name: {
                "time_out": bool(env.termination_manager.get_term_cfg(name).time_out),
                "params": json_value(env.termination_manager.get_term_cfg(name).params),
            }
            for name in env.termination_manager.active_terms
        }
        payload = {
            "schema": "specialist-reset-contract",
            "version": 1,
            "task": args.task,
            "seed": args.seed,
            "episodes": args.episodes,
            "step_dt": float(env.step_dt),
            "max_episode_length": int(env.max_episode_length),
            "active_termination_terms": term_cfg,
            "reset_state": {
                "root_link_pose_w": root_pose.tolist(),
                "root_link_vel_w": root_vel.tolist(),
                "joint_pos": joint_pos.tolist(),
                "joint_vel": joint_vel.tolist(),
                "twist_command_b": command.tolist(),
            },
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": payload["sha256"], "episodes": args.episodes}))


if __name__ == "__main__":
    main()
