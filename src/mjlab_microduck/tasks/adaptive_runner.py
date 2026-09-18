"""Runner adapter for feeding frozen capability-battery results to the gate."""

from __future__ import annotations

import os
import random
import hashlib
import json
import shlex
import subprocess
import time
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


class CommandCapabilityEvaluator:
    """Run the frozen battery through a configured command at runner boundaries."""

    def __init__(self, command: str, timeout_s: int = 900):
        if not command.strip():
            raise ValueError("adaptive evaluator command cannot be empty")
        self.command = command
        self.timeout_s = int(timeout_s)

    def evaluate(self, *, checkpoint_path: Path, task_id: str, axis_mode: str,
                 curriculum_state: Mapping[str, object], iteration: int,
                 seed_set_id: str, evaluation_seed: int):
        output = checkpoint_path.resolve().parent / "adaptive_eval" / checkpoint_path.stem / "capability.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "checkpoint": str(checkpoint_path.resolve()),
            "task_id": task_id,
            "axis_mode": axis_mode,
            "iteration": str(iteration),
            "seed_set_id": seed_set_id,
            "evaluation_seed": str(evaluation_seed),
            "output": str(output),
        }
        # Split the configured argv before substituting paths, so spaces in a
        # checkpoint path remain part of one argument. No shell is involved.
        command = [arg.format(**values) for arg in shlex.split(self.command)]
        with (output.parent / "evaluator.log").open("w") as log:
            subprocess.run(command, check=True, timeout=self.timeout_s, stdout=log, stderr=subprocess.STDOUT)
        if not output.exists():
            raise FileNotFoundError(f"adaptive evaluator did not write {output}")
        from mjlab_microduck.evaluation.capability import CapabilityReport

        return CapabilityReport.from_dict(json.loads(output.read_text(encoding="utf-8")))


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
        self.evaluation_seed_set_id = str(
            getattr(env.cfg, "adaptive_seed_set_id", str(self.evaluation_seed))
        )
        self.evaluation_events: list[dict[str, object]] = []
        self.last_evaluation_provenance: dict[str, object] | None = None
        self.last_known_good_checkpoint: str | None = None
        self.last_gate_outcome: str | None = None
        self.completed_iterations = 0
        evaluator_command = os.environ.get("MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND") or os.environ.get("MICRODUCK_ADAPTIVE_EVALUATOR")
        if evaluator_command and self.evaluation_interval > 0:
            self.evaluator = CommandCapabilityEvaluator(
                evaluator_command,
                int(getattr(env.cfg, "adaptive_evaluation_timeout_s", 900)),
            )
        if not axis_configs:
            self.capability_gate = None
            return
        if self.evaluation_interval > 0 and self.evaluator is None:
            raise ValueError("adaptive evaluation enabled without an evaluator command")
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
        checkpoint_path = str(Path(checkpoint_path).resolve())
        try:
            seed_set_id = getattr(self, "evaluation_seed_set_id", str(self.evaluation_seed))
            report = self.evaluator.evaluate(
                checkpoint_path=Path(checkpoint_path),
                task_id=self.env.cfg.task_id,
                axis_mode=self.capability_gate.axis_mode,
                curriculum_state=self.capability_gate.state_dict(),
                iteration=self.current_learning_iteration,
                seed_set_id=seed_set_id,
                evaluation_seed=self.evaluation_seed,
            )
            self._validate_report(report, checkpoint_path)
            payload = getattr(report, "payload", report)
            metrics = report.to_gate_metrics()
        except Exception as exc:
            self.last_gate_outcome = "evaluation_error"
            self.evaluation_events.append({"kind": "evaluation_error", "error": repr(exc), "iteration": self.current_learning_iteration, "checkpoint": checkpoint_path})
        else:
            self.last_evaluation_provenance = {
                **payload["metadata"],
                "report_sha256": report.sha256(),
                "evaluation_seed": self.evaluation_seed,
                "schema_version": payload["schema_version"],
                "report_path": str(Path(checkpoint_path).parent / "adaptive_eval" / Path(checkpoint_path).stem / "capability.json"),
            }
            previous_best = self.capability_gate.best_metrics.copy()
            self.record_capability_metrics(
                metrics,
                step=getattr(self, "completed_iterations", self.current_learning_iteration) * int(getattr(self, "cfg", {}).get("num_steps_per_env", 24)),
                checkpoint=checkpoint_path,
                seed=self.evaluation_seed,
            )
            preserved = all(
                value >= previous_best.get(name, value) * (1 - self.capability_gate.preservation_tolerance)
                for name, value in metrics.items()
            )
            if payload["aggregate"]["passed"] and preserved and self.last_gate_outcome != "preservation_failure":
                self.last_known_good_checkpoint = str(Path(checkpoint_path).with_suffix(".adaptive.pt"))
        # A hold also changes pass counters/EMA. Persist every boundary without
        # changing the hash of the checkpoint consumed by the evaluator. Direct
        # unit callers may provide a virtual checkpoint; the real learn loop has
        # already written the pre-evaluation file.
        if Path(checkpoint_path).exists():
            self.save(str(Path(checkpoint_path).with_suffix(".adaptive.pt")))

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
        from mjlab_microduck.evaluation.capability import CapabilityReport

        CapabilityReport.from_dict(payload).to_gate_metrics()
        expected_task = getattr(self.env.cfg, "task_id", None)
        if expected_task and metadata["task_id"] != expected_task:
            raise ValueError("capability report task mismatch")
        if hasattr(self, "evaluation_seed_set_id") and metadata["seed_set_id"] != self.evaluation_seed_set_id:
            raise ValueError("capability report seed set mismatch")
        for cfg_key, report_key in (
            ("adaptive_source_sha", "source_sha"),
            ("adaptive_evaluator_config_sha256", "evaluator_config_sha256"),
        ):
            expected_value = getattr(self.env.cfg, cfg_key, "")
            if expected_value and metadata[report_key] != expected_value:
                raise ValueError(f"capability report {report_key} mismatch")


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the normal runner in explicit iteration windows when enabled.

        Preserve stock rollout/update/logging order and evaluate after complete
        PPO updates. A zero interval uses the inherited runner unchanged.
        """
        if self.evaluation_interval <= 0 or self.capability_gate is None:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        # The installed RSL-RL runner closes its writer at the end of learn(),
        # so windows are implemented here rather than by repeatedly calling
        # super().learn(1).
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        if self.is_distributed:
            raise ValueError("adaptive synchronous evaluation currently requires a single training process")
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        start_it = self.completed_iterations
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.perf_counter()
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
                collect_time = time.perf_counter() - start
                start = time.perf_counter()
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            learn_time = time.perf_counter() - start
            self.current_learning_iteration = it
            self.completed_iterations = it + 1
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            if self.completed_iterations % self.evaluation_interval == 0:
                checkpoint = os.path.join(self.logger.log_dir, f"model_{it}.eval.pt")
                self.save(checkpoint)
                self._evaluate_window(checkpoint)
                if self.last_gate_outcome == "preservation_failure":
                    # Restoring PPO does not rewind collected experience or the
                    # campaign budget. Start fresh episodes under restored state.
                    self.current_learning_iteration = it
                    self.completed_iterations = it + 1
                    obs, _ = self.env.reset()
                    obs = obs.to(self.device)
                    self.alg.train_mode()
                    self.save(str(Path(checkpoint).with_suffix(".adaptive.pt")))
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
        self.last_gate_outcome = decision.outcome.value
        event = {"kind": decision.outcome.value, "step": self.current_learning_iteration if step is None else step, "checkpoint": checkpoint, "reason": decision.reason}
        if getattr(self, "last_evaluation_provenance", None) is not None:
            event["provenance"] = dict(self.last_evaluation_provenance)
        self.evaluation_events.append(event)
        if decision.outcome.value == "preservation_failure" and self.last_known_good_checkpoint:
            self.rollback(self.last_known_good_checkpoint)
            return None
        if transition is not None:
            apply_stage_to_env(
                _manager_env(self.env),
                transition.axis,
                self.capability_gate.stage_value(transition.axis),
            )
            if checkpoint is not None:
                # Keep the evaluated checkpoint immutable and persist the
                # applied stage in a separate explicit rollback checkpoint.
                applied_checkpoint = Path(checkpoint).with_name(
                    f"{Path(checkpoint).stem}.adaptive.pt"
                )
                self.save(str(applied_checkpoint))
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
                "evaluation_provenance": getattr(self, "last_evaluation_provenance", None),
                "evaluation_schema_version": int(getattr(self.env.cfg, "adaptive_evaluator_schema_version", 2)),
                "evaluation_iteration": self.current_learning_iteration,
                "completed_iterations": getattr(self, "completed_iterations", self.current_learning_iteration + 1),
                "evaluation_seed_set_id": getattr(self, "evaluation_seed_set_id", None),
                "evaluation_seed": getattr(self, "evaluation_seed", None),
                "env_step": getattr(self, "completed_iterations", self.current_learning_iteration + 1) * int(getattr(self, "cfg", {}).get("num_steps_per_env", 24)),
            })
        payload["adaptive_rng_state"] = self._rng_state()
        super().save(path, payload)

    def adaptive_checkpoint_info(self) -> dict[str, object]:
        """Return the co-located adaptive metadata for audit and checkpoint tests."""
        state = None if self.capability_gate is None else dict(self.capability_gate.state_dict())
        if state is not None:
            state.update({
                "version": 1,
                "stage_values": {name: self.capability_gate.stage_value(name) for name in self.capability_gate.axis_order},
                "last_known_good_checkpoint": self.last_known_good_checkpoint,
                "evaluation_events": list(self.evaluation_events),
                "evaluation_provenance": getattr(self, "last_evaluation_provenance", None),
                "evaluation_schema_version": int(getattr(self.env.cfg, "adaptive_evaluator_schema_version", 2)),
                "evaluation_iteration": self.current_learning_iteration,
                "completed_iterations": getattr(self, "completed_iterations", self.current_learning_iteration + 1),
                "evaluation_seed_set_id": getattr(self, "evaluation_seed_set_id", None),
                "evaluation_seed": getattr(self, "evaluation_seed", None),
                "env_step": getattr(self, "completed_iterations", self.current_learning_iteration + 1) * int(getattr(self, "cfg", {}).get("num_steps_per_env", 24)),
            })
        return {"adaptive_curriculum": state, "adaptive_rng_state": self._rng_state()}

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and infos.get("adaptive_curriculum") and self.capability_gate is not None:
            adaptive_state = infos["adaptive_curriculum"]
            self.completed_iterations = int(adaptive_state.get("completed_iterations", self.current_learning_iteration + 1))
            if adaptive_state.get("version", 1) != 1:
                raise ValueError("unsupported adaptive checkpoint version")
            self.capability_gate.load_state_dict(adaptive_state)
            self.last_known_good_checkpoint = adaptive_state.get("last_known_good_checkpoint")
            self.evaluation_events = list(adaptive_state.get("evaluation_events", []))
            self.last_evaluation_provenance = adaptive_state.get("evaluation_provenance")
            stored_schema = adaptive_state.get("evaluation_schema_version", 2)
            expected_schema = int(getattr(self.env.cfg, "adaptive_evaluator_schema_version", 2))
            if int(stored_schema) != expected_schema:
                raise ValueError("adaptive checkpoint evaluator schema mismatch")
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
        if not self.last_known_good_checkpoint:
            raise ValueError("rollback requires a recorded known-good checkpoint")
        if checkpoint_path != self.last_known_good_checkpoint:
            raise ValueError("rollback checkpoint is not the recorded known-good checkpoint")
        history = list(self.evaluation_events)
        provenance = getattr(self, "last_evaluation_provenance", None)
        consumed_iterations = getattr(self, "completed_iterations", None)
        infos = self.load(checkpoint_path)
        self.evaluation_events = history
        self.last_evaluation_provenance = provenance
        if consumed_iterations is not None:
            self.completed_iterations = consumed_iterations
        # Loading a checkpoint may contain the predecessor's pointer. The
        # explicit rollback target remains the known-good state after restore.
        self.last_known_good_checkpoint = checkpoint_path
        self.evaluation_events.append({"kind": "rollback", "checkpoint": checkpoint_path})
        return infos
