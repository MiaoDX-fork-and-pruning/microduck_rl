#!/usr/bin/env python3
"""Capture the exact reset state and manager contract used by a specialist task."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

_TRACE_FIELDS = frozenset({
    "observation", "qpos", "qvel", "ctrl", "root_pose",
    "previous_action", "raw_action", "applied_action",
    "first_episode_active", "episode_step", "reward", "done",
    "termination",
})


class EvaluatorTrace:
    """Read-only samples from the accepted evaluator, replayed via its full prefix.

    These tensors are evidence, not a standalone environment checkpoint. The
    prefix replay preserves manager/RNG/delay state without approximating it.
    """

    def __init__(self, contract: dict):
        self.contract = contract
        self.rows = []

    @staticmethod
    def _copy(tensor):
        return tensor.detach().cpu().numpy().copy()

    def before_step(self, env, observation, action, active) -> None:
        import torch

        raw = env.unwrapped
        # Do not recompute observations: that advances noise/history buffers.
        self.rows.append({
            "observation": self._copy(observation["actor"]),
            "qpos": self._copy(raw.sim.data.qpos),
            "qvel": self._copy(raw.sim.data.qvel),
            "ctrl": self._copy(raw.sim.data.ctrl),
            "root_pose": self._copy(raw.scene["robot"].data.root_link_pose_w),
            "previous_action": self._copy(raw.action_manager.action),
            "raw_action": self._copy(action),
            "applied_action": self._copy(
                action if env.clip_actions is None
                else torch.clamp(action, -env.clip_actions, env.clip_actions)
            ),
            "first_episode_active": self._copy(active),
            "episode_step": self._copy(raw.episode_length_buf),
        })

    def after_step(self, env, reward, done) -> None:
        import torch

        self.rows[-1].update({
            "reward": self._copy(reward), "done": self._copy(done),
            "termination": self._copy(torch.stack([
                env.termination_manager.get_term(name)
                for name in env.termination_manager.active_terms
            ], dim=-1)),
        })

    def write(self, path: Path, reference: Path | None = None) -> dict:
        if not self.rows:
            raise ValueError("cannot write an empty evaluator trace")
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays, metadata=json.dumps(self.contract, sort_keys=True))
        result = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contract": self.contract, "control_ticks": len(self.rows),
            "finite": all(np.isfinite(value).all() for value in arrays.values()),
            "snapshot_resume_supported": False,
        }
        if reference is not None:
            result["prefix_replay"] = compare_evaluator_traces(reference, path)
        return result


def compare_evaluator_traces(reference: Path, candidate: Path) -> dict:
    """Compare complete seeded evaluator trajectories without tolerating truncation."""
    with np.load(reference, allow_pickle=False) as left, np.load(candidate, allow_pickle=False) as right:
        metadata_present = "metadata" in left.files and "metadata" in right.files
        contract_matches = (
            metadata_present
            and json.loads(left["metadata"].item()) == json.loads(right["metadata"].item())
        )
        fields_match = (
            set(left.files) == set(right.files)
            and _TRACE_FIELDS.issubset(set(left.files) - {"metadata"})
            and _TRACE_FIELDS.issubset(set(right.files) - {"metadata"})
        )
        errors, mismatches = {}, []
        for field in sorted((set(left.files) & set(right.files)) - {"metadata"}):
            a, b = left[field], right[field]
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                mismatches.append(field)
            elif a.dtype.kind in "biu":
                if not np.array_equal(a, b):
                    mismatches.append(field)
                errors[field] = float(np.max(a != b, initial=False))
            else:
                errors[field] = float(np.max(np.abs(a - b), initial=0))
                if errors[field] > 1e-6:
                    mismatches.append(field)
    return {
        "reference": str(reference),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "contract_matches": contract_matches, "fields_match": fields_match,
        "required_fields": sorted(_TRACE_FIELDS),
        "max_abs": errors, "mismatched_fields": mismatches,
        "passed": contract_matches and fields_match and not mismatches,
    }


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
            "accepted_evaluator_equivalence": False,
            "capture_route": "CPU explicit reset(seed); not RslRlVecEnvWrapper startup",
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
