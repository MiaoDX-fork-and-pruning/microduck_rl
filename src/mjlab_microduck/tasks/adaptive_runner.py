"""Runner adapter for feeding frozen capability-battery results to the gate."""

from __future__ import annotations

import os
import random
import hashlib
from typing import Mapping

import numpy as np
import torch
from pathlib import Path

from .adaptive_curriculum import (
    ADAPTIVE_AXIS_CONFIGS,
    CapabilityGate,
    apply_stage_to_env,
)
from mjlab_microduck.evaluation.capability import resolve_enabled_axes
from . import MicroduckOnPolicyRunner


def _manager_env(env):
    """Resolve the ManagerBasedRlEnv behind VecEnv wrappers."""
    current = env
    seen: set[int] = set()
    while not hasattr(current, "event_manager"):
        if id(current) in seen:
            raise AttributeError("could not resolve adaptive environment event_manager")
        seen.add(id(current))
        current = getattr(current, "unwrapped", None)
        if current is None:
            raise AttributeError("adaptive environment has no event_manager")
    return current


class AdaptiveMicroduckOnPolicyRunner(MicroduckOnPolicyRunner):
    """Canonical PPO runner plus explicit, checkpointed curriculum state."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        mode = getattr(env.cfg, "adaptive_axis_mode", "composed")
        axis_names = resolve_enabled_axes(mode)
        axis_configs = tuple(axis for axis in ADAPTIVE_AXIS_CONFIGS if axis.name in axis_names)
        self.evaluation_interval = int(getattr(env.cfg, "adaptive_evaluation_interval", 0))
        self.evaluator = None
        self.evaluation_seed = int(getattr(env.cfg, "adaptive_evaluation_seed", 0))
        self.evaluation_events: list[dict[str, object]] = []
        self.last_known_good_checkpoint: str | None = None
        if not axis_configs:
            self.capability_gate = None
            return
        self.capability_gate = CapabilityGate(
            axis_configs,
            critical_buckets=("zero", "forward", "lateral", "yaw", "turn-left", "turn-right"),
            axis_mode=mode,
            ema_alpha=0.25,
            preservation_tolerance=0.05,
        )

    def set_evaluator(self, evaluator) -> None:
        """Inject a synchronous evaluator (used by production adapters/tests)."""
        self.evaluator = evaluator

    def _rng_state(self) -> dict[str, object]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self, state: Mapping[str, object]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _evaluate_window(self, checkpoint_path: str) -> None:
        if self.evaluation_interval <= 0 or self.evaluator is None or self.capability_gate is None:
            return
        try:
            report = self.evaluator.evaluate(
                checkpoint_path=Path(checkpoint_path),
                task_id=getattr(self.env.cfg, "task_id", "adaptive_velocity"),
                axis_mode=self.capability_gate.axis_mode,
                curriculum_state=self.capability_gate.state_dict(),
                iteration=self.current_learning_iteration,
                seed_set_id=str(self.evaluation_seed),
            )
            self._validate_report(report, checkpoint_path)
            metrics = report.to_gate_metrics() if hasattr(report, "to_gate_metrics") else report["metrics"]
        except Exception as exc:
            self.evaluation_events.append({"kind": "evaluation_error", "error": repr(exc), "iteration": self.current_learning_iteration})
            return
        self.record_capability_metrics(
            metrics,
            step=self.current_learning_iteration,
            checkpoint=checkpoint_path,
            seed=self.evaluation_seed,
        )

    def _validate_report(self, report, checkpoint_path: str) -> None:
        """Fail closed when an evaluator returns the wrong artifact/report."""
        payload = getattr(report, "payload", report)
        if not isinstance(payload, Mapping):
            raise ValueError("evaluator returned a non-mapping report")
        if payload.get("schema_version") != int(getattr(self.env.cfg, "adaptive_evaluator_schema_version", 2)):
            raise ValueError("capability report schema mismatch")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("capability report metadata missing")
        expected = Path(checkpoint_path)
        if metadata.get("checkpoint") and str(metadata["checkpoint"]) != str(expected):
            raise ValueError("capability report checkpoint mismatch")
        if expected.exists() and metadata.get("checkpoint_sha256"):
            digest = hashlib.sha256(expected.read_bytes()).hexdigest()
            if metadata["checkpoint_sha256"] != digest:
                raise ValueError("capability report checkpoint hash mismatch")
        mode = self.capability_gate.axis_mode
        if payload.get("axis_mode") != mode or tuple(payload.get("enabled_axes", ())) != tuple(self.capability_gate.axis_order):
            raise ValueError("capability report axis contract mismatch")
        if not expected.exists():
            raise ValueError("evaluated checkpoint does not exist")
        if not metadata.get("checkpoint_sha256"):
            raise ValueError("capability report checkpoint hash is missing")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the normal runner in explicit iteration windows when enabled.

        Calling the installed runner once per iteration keeps the stock PPO
        lifecycle intact while giving the adaptive evaluator a deterministic
        checkpoint boundary. The default interval is zero, so canonical-like
        adaptive runs use the inherited implementation unchanged.
        """
        if self.evaluation_interval <= 0 or self.evaluator is None:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        # The installed RSL-RL runner closes its writer at the end of learn(),
        # so windows are implemented here rather than by repeatedly calling
        # super().learn(1).
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        from rsl_rl.utils import check_nan

                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic)
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=0.0,
                learn_time=0.0,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            if self.current_learning_iteration > 0 and self.current_learning_iteration % self.evaluation_interval == 0:
                checkpoint = os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
                # Evaluation always consumes the exact post-update state at this
                # boundary, even when save_interval differs or logging is disabled.
                if not os.path.exists(checkpoint):
                    self.save(checkpoint)
                self._evaluate_window(checkpoint)
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def record_capability_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        checkpoint: str | None = None,
        seed: int = 0,
    ):
        """Consume one frozen battery window and apply at most one transition."""
        if self.capability_gate is None:
            return None
        decision = self.capability_gate.decide(
            self.current_learning_iteration if step is None else step,
            metrics,
            checkpoint=checkpoint,
            seed=seed,
        )
        transition = decision.transition
        self.evaluation_events.append({"kind": decision.outcome.value, "step": self.current_learning_iteration if step is None else step, "checkpoint": checkpoint, "reason": decision.reason})
        if transition is not None:
            apply_stage_to_env(
                _manager_env(self.env),
                transition.axis,
                self.capability_gate.stage_value(transition.axis),
            )
            self.last_known_good_checkpoint = checkpoint
        return transition

    def save(self, path: str, infos: dict | None = None) -> None:
        payload = {} if infos is None else dict(infos)
        payload["adaptive_curriculum"] = (
            None if self.capability_gate is None else self.capability_gate.state_dict()
        )
        if payload["adaptive_curriculum"] is not None:
            payload["adaptive_curriculum"] = dict(payload["adaptive_curriculum"])
            payload["adaptive_curriculum"].update({
                "version": 1,
                "stage_values": {name: self.capability_gate.stage_value(name) for name in self.capability_gate.axis_order},
                "last_known_good_checkpoint": self.last_known_good_checkpoint,
                "evaluation_events": list(self.evaluation_events),
            })
        payload["adaptive_rng_state"] = self._rng_state()
        super().save(path, payload)

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and infos.get("adaptive_curriculum") and self.capability_gate is not None:
            adaptive_state = infos["adaptive_curriculum"]
            if adaptive_state.get("version", 1) != 1:
                raise ValueError("unsupported adaptive checkpoint version")
            self.capability_gate.load_state_dict(adaptive_state)
            self.last_known_good_checkpoint = adaptive_state.get("last_known_good_checkpoint")
            self.evaluation_events = list(adaptive_state.get("evaluation_events", []))
            for axis_name, state in self.capability_gate.states.items():
                apply_stage_to_env(
                    _manager_env(self.env),
                    axis_name,
                    self.capability_gate.stage_value(axis_name),
                )
        if infos and infos.get("adaptive_rng_state"):
            self._restore_rng_state(infos["adaptive_rng_state"])
        return infos

    def rollback(self, checkpoint_path: str):
        """Restore the complete trainer and curriculum state from a known-good checkpoint."""
        if not checkpoint_path:
            raise ValueError("rollback requires an explicit checkpoint path")
        infos = self.load(checkpoint_path)
        self.evaluation_events.append({"kind": "rollback", "checkpoint": checkpoint_path})
        return infos
