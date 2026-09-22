"""Runner adapter for feeding frozen capability-battery results to the gate."""

from __future__ import annotations

from copy import deepcopy
import os
import random
import hashlib
import json
import shlex
import subprocess
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from pathlib import Path

from .adaptive_curriculum import (
    ADAPTIVE_AXIS_CONFIGS,
    CapabilityGate,
    CommandExposure,
    apply_stage_to_env,
)
from mjlab_microduck.evaluation.capability import BUCKETS, resolve_enabled_axes
from . import MicroduckOnPolicyRunner


# ``AdaptiveVelocityCommand`` samples six explicit capability buckets and a
# seventh residual continuous-command pool. Keep the latter visible in the
# evidence contract so its reward mass is not reported as an unexplained
# invalid sample.
COMMAND_FEEDBACK_BUCKETS = (*BUCKETS, "nominal")


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


class BucketFeedbackTracker:
    """Accumulate command-conditioned samples and weighted reward mass.

    ``RewardManager._step_reward`` contains active terms after weights have
    been applied but before environment ``dt`` scaling. Bucket ids are captured
    before ``env.step`` so a reset-time command resample cannot attribute a
    reward to the next episode's command.
    """

    schema_version = 1

    def __init__(self, term_names: Sequence[str], *, device: torch.device | str, step_dt: float):
        names = tuple(str(name) for name in term_names)
        if not names:
            raise ValueError("bucket feedback requires at least one reward term")
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("bucket feedback step_dt must be finite and positive")
        self.term_names = names
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        # The rollout loop runs under ``torch.inference_mode``.  Keep tracker
        # buffers as ordinary mutable tensors because window snapshots reset
        # them after evaluation in normal mode, and checkpoint restore writes
        # into them outside the rollout context.
        with torch.inference_mode(False):
            self.sample_count = torch.zeros(
                len(COMMAND_FEEDBACK_BUCKETS), dtype=torch.long, device=self.device
            )
            self.unclassified_count = torch.zeros((), dtype=torch.long, device=self.device)
            self.reward_mass = torch.zeros(
                (len(COMMAND_FEEDBACK_BUCKETS), len(names)),
                dtype=torch.float32,
                device=self.device,
            )
            self.reward_abs_mass = torch.zeros_like(self.reward_mass)

    def reset(self) -> None:
        self.sample_count.zero_()
        self.unclassified_count.zero_()
        self.reward_mass.zero_()
        self.reward_abs_mass.zero_()

    def record(self, bucket_ids: torch.Tensor, weighted_step_reward: torch.Tensor) -> None:
        ids = bucket_ids.detach().to(device=self.device, dtype=torch.long).reshape(-1)
        values = weighted_step_reward.detach().to(device=self.device).reshape(ids.shape[0], -1)
        if values.shape[1] != len(self.term_names):
            raise ValueError("bucket feedback reward term shape mismatch")
        valid = (ids >= 0) & (ids < len(COMMAND_FEEDBACK_BUCKETS))
        self.unclassified_count += (~valid).sum()
        if not torch.any(valid):
            return
        valid_ids = ids[valid]
        valid_values = values[valid].to(dtype=self.reward_mass.dtype) * self.step_dt
        self.sample_count.index_add_(0, valid_ids, torch.ones_like(valid_ids))
        self.reward_mass.index_add_(0, valid_ids, valid_values)
        self.reward_abs_mass.index_add_(0, valid_ids, valid_values.abs())

    def _payload(self) -> dict[str, object]:
        counts = self.sample_count.detach().cpu().tolist()
        total = int(sum(counts))
        fractions = {
            name: (float(counts[idx]) / total if total else 0.0)
            for idx, name in enumerate(COMMAND_FEEDBACK_BUCKETS)
        }
        return {
            "schema_version": self.schema_version,
            "bucket_names": list(COMMAND_FEEDBACK_BUCKETS),
            "term_names": list(self.term_names),
            "step_dt": self.step_dt,
            "sample_count": {
                name: int(counts[idx])
                for idx, name in enumerate(COMMAND_FEEDBACK_BUCKETS)
            },
            "sample_fraction": fractions,
            "unclassified_count": int(self.unclassified_count.item()),
            "weighted_reward_mass": {
                name: {
                    term: float(value)
                    for term, value in zip(
                        self.term_names,
                        self.reward_mass[idx].detach().cpu().tolist(),
                        strict=True,
                    )
                }
                for idx, name in enumerate(COMMAND_FEEDBACK_BUCKETS)
            },
            "weighted_reward_abs_mass": {
                name: {
                    term: float(value)
                    for term, value in zip(
                        self.term_names,
                        self.reward_abs_mass[idx].detach().cpu().tolist(),
                        strict=True,
                    )
                }
                for idx, name in enumerate(COMMAND_FEEDBACK_BUCKETS)
            },
        }

    def state_dict(self) -> dict[str, object]:
        return self._payload()

    def snapshot(self, *, reset: bool = True) -> dict[str, object]:
        payload = self._payload()
        if reset:
            self.reset()
        return payload

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("unsupported bucket feedback schema")
        if tuple(payload.get("bucket_names", ())) != tuple(COMMAND_FEEDBACK_BUCKETS):
            raise ValueError("bucket feedback bucket mismatch")
        if tuple(payload.get("term_names", ())) != self.term_names:
            raise ValueError("bucket feedback reward term mismatch")
        step_dt = float(payload.get("step_dt", self.step_dt))
        if not np.isclose(step_dt, self.step_dt):
            raise ValueError("bucket feedback step_dt mismatch")
        counts = payload.get("sample_count")
        signed = payload.get("weighted_reward_mass")
        absolute = payload.get("weighted_reward_abs_mass")
        if not isinstance(counts, Mapping) or not isinstance(signed, Mapping) or not isinstance(absolute, Mapping):
            raise ValueError("bucket feedback payload is incomplete")
        try:
            count_values = [int(counts[name]) for name in COMMAND_FEEDBACK_BUCKETS]
            signed_values = [
                [float(signed[name][term]) for term in self.term_names]
                for name in COMMAND_FEEDBACK_BUCKETS
            ]
            absolute_values = [
                [float(absolute[name][term]) for term in self.term_names]
                for name in COMMAND_FEEDBACK_BUCKETS
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bucket feedback payload has invalid values") from exc
        if any(value < 0 for value in count_values) or not np.isfinite(signed_values).all() or not np.isfinite(absolute_values).all():
            raise ValueError("bucket feedback payload has non-finite values")
        self.sample_count.copy_(torch.tensor(count_values, dtype=torch.long, device=self.device))
        self.unclassified_count.fill_(int(payload.get("unclassified_count", 0)))
        self.reward_mass.copy_(torch.tensor(signed_values, dtype=torch.float32, device=self.device))
        self.reward_abs_mass.copy_(torch.tensor(absolute_values, dtype=torch.float32, device=self.device))


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
        self.last_known_good_buckets: tuple[str, ...] = ()
        self.last_gate_outcome: str | None = None
        self.completed_iterations = 0
        self.resume_checkpoint: str | None = None
        self.bucket_feedback: BucketFeedbackTracker | None = None
        initial_focus = getattr(env.cfg, "adaptive_initial_focus", "forward")
        frontier_order = getattr(env.cfg, "adaptive_frontier_order", ()) or None
        self.command_exposure = (
            CommandExposure(
                initial_focus=initial_focus,
                frontier_order=frontier_order,
                stall_windows=int(getattr(env.cfg, "adaptive_frontier_stall_windows", 0)),
                stall_improvement=float(getattr(env.cfg, "adaptive_frontier_stall_improvement", 0.05)),
            )
            if getattr(env.cfg, "adaptive_command_exposure", False)
            else None
        )
        if self.command_exposure is not None:
            self.command_exposure.apply(_manager_env(env))
        evaluator_command = os.environ.get("MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND") or os.environ.get("MICRODUCK_ADAPTIVE_EVALUATOR")
        if evaluator_command and self.evaluation_interval > 0:
            self.evaluator = CommandCapabilityEvaluator(
                evaluator_command,
                int(getattr(env.cfg, "adaptive_evaluation_timeout_s", 900)),
            )
        if axis_configs and self.evaluation_interval > 0 and self.evaluator is None:
            raise ValueError("adaptive evaluation enabled without an evaluator command")
        self.capability_gate = CapabilityGate(
            axis_configs,
            critical_buckets=("zero", "forward", "lateral", "yaw", "turn-left", "turn-right"),
            axis_mode=mode,
            ema_alpha=0.25,
            preservation_tolerance=0.05,
        ) if axis_configs else None
        # The launcher passes an exact path, without MJLab's regex run lookup.
        # Evaluators/exporters have no training log_dir and never resume here.
        resume = os.environ.get("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT")
        if resume and log_dir is not None:
            self.load(resume, map_location=device)

    def set_evaluator(self, evaluator) -> None:
        """Inject a synchronous evaluator (used by production adapters/tests)."""
        self.evaluator = evaluator

    def _ensure_bucket_feedback(self) -> BucketFeedbackTracker | None:
        """Create the tracker once the adaptive command term has sampled ids."""
        existing = getattr(self, "bucket_feedback", None)
        if existing is not None:
            return existing
        try:
            env = _manager_env(self.env)
            command_term = env.command_manager.get_term("twist")
            bucket_ids = getattr(command_term, "bucket_ids", None)
            reward_manager = env.reward_manager
            term_names = tuple(reward_manager.active_terms)
            step_dt = float(getattr(env, "step_dt", 0.02))
        except (AttributeError, KeyError, ValueError):
            return None
        if bucket_ids is None or not term_names:
            return None
        self.bucket_feedback = BucketFeedbackTracker(
            term_names, device=bucket_ids.device, step_dt=step_dt
        )
        return self.bucket_feedback

    def _capture_command_bucket_ids(self) -> torch.Tensor | None:
        try:
            command_term = _manager_env(self.env).command_manager.get_term("twist")
            bucket_ids = getattr(command_term, "bucket_ids", None)
        except (AttributeError, KeyError):
            return None
        if bucket_ids is None:
            return None
        self._ensure_bucket_feedback()
        return bucket_ids.detach().clone()

    def _record_bucket_feedback(self, bucket_ids: torch.Tensor | None) -> None:
        if bucket_ids is None:
            return
        tracker = self._ensure_bucket_feedback()
        if tracker is None:
            return
        reward_values = getattr(_manager_env(self.env).reward_manager, "_step_reward", None)
        if reward_values is not None:
            tracker.record(bucket_ids, reward_values)

    def _snapshot_bucket_feedback(self, *, reset: bool = True) -> dict[str, object] | None:
        tracker = self._ensure_bucket_feedback()
        if tracker is None:
            return None
        return tracker.snapshot(reset=reset)

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
        # Checkpoints loaded with map_location=cuda move every tensor in the
        # metadata too; the CPU generator only accepts a CPU ByteTensor.
        torch.set_rng_state(state["torch"].detach().cpu())
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [item.detach().cpu() for item in state["torch_cuda"]]
            )

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
            command_feedback = self._snapshot_bucket_feedback(reset=True)
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
                command_feedback=command_feedback,
            )
            threshold = max(axis.upper_threshold for axis in self.capability_gate.axes.values())
            mastered_before = {
                name for name, value in previous_best.items() if value >= threshold
            }
            mastered_now = {name for name, value in metrics.items() if value >= threshold}
            preserved = all(
                metrics[name] >= previous_best[name] * (1 - self.capability_gate.preservation_tolerance)
                for name in mastered_before
            )
            # Acquisition is incremental: a checkpoint becomes rollback-safe
            # as soon as it establishes one capability and preserves every
            # capability mastered before it. Waiting for aggregate ``passed``
            # meant the first useful zero/forward checkpoint was discarded,
            # leaving preservation failures with no rollback target.
            if (mastered_before | mastered_now) and preserved and self.last_gate_outcome != "preservation_failure":
                self.last_known_good_checkpoint = str(Path(checkpoint_path).with_suffix(".adaptive.pt"))
                self.last_known_good_buckets = tuple(sorted(mastered_before | mastered_now))
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
        """Count completed PPO updates identically with and without evaluation."""
        if num_learning_iterations < 1:
            raise ValueError("num_learning_iterations must be positive")
        if getattr(self, "_needs_reset", False):
            self.env.reset()
            self._needs_reset = False
        # The installed RSL-RL runner closes its writer at the end of learn(),
        # so windows are implemented here rather than by repeatedly calling
        # super().learn(1).
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        if self.is_distributed:
            if self.evaluation_interval > 0 and self.capability_gate is not None:
                raise ValueError("adaptive synchronous evaluation currently requires a single training process")
            self.alg.broadcast_parameters()
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        start_it = self.completed_iterations
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    command_bucket_ids = self._capture_command_bucket_ids()
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
                    self._record_bucket_feedback(command_bucket_ids)
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
            if self.capability_gate is not None and self.evaluation_interval > 0 and self.completed_iterations % self.evaluation_interval == 0:
                checkpoint = os.path.join(self.logger.log_dir, f"model_{it}.eval.pt")
                self.save(checkpoint)
                self._evaluate_window(checkpoint)
                if getattr(self, "_needs_reset", False):
                    # Restoring PPO does not rewind collected experience or the
                    # campaign budget. Start fresh episodes under restored state.
                    # The preceding rollout ran in inference mode and may have
                    # created inference tensors inside BAM/mjlab delay buffers;
                    # reset those buffers in the same mode before collecting the
                    # next rollout.
                    self.current_learning_iteration = it
                    self.completed_iterations = it + 1
                    obs, _ = self._reset_after_rollback()
                    self._needs_reset = False
                    obs = obs.to(self.device)
                    self.alg.train_mode()
                    self.save(str(Path(checkpoint).with_suffix(".adaptive.pt")))
        if self.logger.writer is not None:
            checkpoint = Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"
            self.save(str(checkpoint))
            self._write_training_result(checkpoint, start_it)
            self.logger.stop_logging_writer()

    def _reset_after_rollback(self):
        """Reset stateful simulator buffers in rollout's inference context."""
        with torch.inference_mode():
            return self.env.reset()

    def _write_training_result(self, checkpoint: Path, start_iteration: int) -> None:
        """Publish completion only after the final checkpoint is durably written."""
        checkpoint = checkpoint.resolve()
        result = {
            "version": 1, "status": "completed", "task_id": self.env.cfg.task_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "resume_checkpoint": getattr(self, "resume_checkpoint", None),
            "start_completed_iterations": start_iteration,
            "completed_iterations": self.completed_iterations,
            "segment_iterations": self.completed_iterations - start_iteration,
            "env_step": self.completed_iterations * self.cfg["num_steps_per_env"],
            "num_envs": self.env.num_envs,
            "adaptive_state": self.adaptive_checkpoint_info()["adaptive_curriculum"],
        }
        path = Path(os.environ.get("MICRODUCK_ADAPTIVE_RESULT_FILE", checkpoint.parent / "training-result.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def record_capability_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        checkpoint: str | None = None,
        seed: int = 0,
        command_feedback: Mapping[str, object] | None = None,
    ):
        """Consume one frozen battery window and apply at most one transition."""
        if self.capability_gate is None:
            return None
        if step is None:
            step = getattr(self, "completed_iterations", self.current_learning_iteration + 1) * self.cfg["num_steps_per_env"]
        decision = self.capability_gate.decide(
            step,
            metrics,
            checkpoint=checkpoint,
            seed=seed,
        )
        transition = decision.transition
        self.last_gate_outcome = decision.outcome.value
        event = {"kind": decision.outcome.value, "step": step, "checkpoint": checkpoint, "reason": decision.reason}
        if command_feedback is not None:
            event["command_feedback"] = dict(command_feedback)
        if getattr(self, "last_evaluation_provenance", None) is not None:
            event["provenance"] = dict(self.last_evaluation_provenance)
        self.evaluation_events.append(event)
        if decision.outcome.value == "preservation_failure" and self.last_known_good_checkpoint:
            self.rollback(self.last_known_good_checkpoint)
            return None
        exposure = getattr(self, "command_exposure", None)
        if exposure is not None:
            exposure.update(metrics)
            exposure.apply(_manager_env(self.env))
            event["command_exposure"] = exposure.state_dict()
        if transition is not None:
            apply_stage_to_env(
                _manager_env(self.env),
                transition.axis,
                self.capability_gate.stage_value(transition.axis),
            )
        return transition

    def save(self, path: str, infos: dict | None = None) -> None:
        payload = {} if infos is None else dict(infos)
        payload.update(self.adaptive_checkpoint_info())
        super().save(path, payload)

    def adaptive_checkpoint_info(self) -> dict[str, object]:
        """One metadata contract for adaptive and static runs, save and audit."""
        gate = self.capability_gate
        completed = getattr(self, "completed_iterations", self.current_learning_iteration + 1)
        exposure = getattr(self, "command_exposure", None)
        state = gate.state_dict() if gate is not None else {"axis_mode": "all_static", "enabled_axes": []}
        state.update({
            "version": 1,
            "task_id": getattr(self.env.cfg, "task_id", None),
            "num_envs": getattr(self.env, "num_envs", None),
            "stage_values": {} if gate is None else {name: gate.stage_value(name) for name in gate.axis_order},
            "command_exposure": None if exposure is None else exposure.state_dict(),
            "command_feedback": None
            if getattr(self, "bucket_feedback", None) is None
            else self.bucket_feedback.state_dict(),
            "last_known_good_checkpoint": self.last_known_good_checkpoint,
            "last_known_good_buckets": list(getattr(self, "last_known_good_buckets", ())),
            "evaluation_events": list(self.evaluation_events),
            "evaluation_provenance": getattr(self, "last_evaluation_provenance", None),
            "evaluation_schema_version": int(getattr(self.env.cfg, "adaptive_evaluator_schema_version", 2)),
            "evaluation_iteration": self.current_learning_iteration,
            "completed_iterations": completed,
            "evaluation_seed_set_id": getattr(self, "evaluation_seed_set_id", None),
            "evaluation_seed": getattr(self, "evaluation_seed", None),
            "env_step": completed * int(getattr(self, "cfg", {}).get("num_steps_per_env", 24)),
        })
        return {"adaptive_curriculum": state, "adaptive_rng_state": self._rng_state()}

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        # PPO keeps its scheduler scalar separately from Adam's param groups;
        # restore the scalar after the optimizer state is loaded so a rollback
        # resumes with the exact learning-rate state of the checkpoint.
        if load_cfg is None or load_cfg.get("optimizer", False):
            optimizer = getattr(self.alg, "optimizer", None)
            groups = getattr(optimizer, "param_groups", ())
            if groups and hasattr(self.alg, "learning_rate"):
                self.alg.learning_rate = groups[0]["lr"]
        # Match RSL-RL's iteration flag: actor-only loading is inference or
        # fine-tuning, never a restoration of trainer RNG/curriculum/progress.
        if load_cfg is not None and not load_cfg.get("iteration", False):
            return infos
        state = (infos or {}).get("adaptive_curriculum")
        gate = deepcopy(self.capability_gate)
        exposure = deepcopy(getattr(self, "command_exposure", None))
        completed = self.current_learning_iteration + 1
        if state:
            if state.get("version", 1) != 1:
                raise ValueError("unsupported adaptive checkpoint version")
            if state.get("task_id") and state["task_id"] != getattr(self.env.cfg, "task_id", None):
                raise ValueError("adaptive checkpoint task mismatch")
            if int(state.get("evaluation_schema_version", 2)) != int(getattr(self.env.cfg, "adaptive_evaluator_schema_version", 2)):
                raise ValueError("adaptive checkpoint evaluator schema mismatch")
            if gate is not None:
                gate.load_state_dict(state)
            elif state.get("enabled_axes"):
                raise ValueError("adaptive state axis mode/allowlist mismatch")
            if bool(state.get("command_exposure")) != (exposure is not None):
                raise ValueError("adaptive checkpoint command exposure mismatch")
            if exposure is not None:
                exposure.load_state_dict(state["command_exposure"])
            feedback_state = state.get("command_feedback")
            if feedback_state is not None:
                tracker = self._ensure_bucket_feedback()
                if tracker is None:
                    raise ValueError("adaptive checkpoint command feedback mismatch")
                tracker.load_state_dict(feedback_state)
            else:
                # Checkpoints written before command-conditioned feedback was
                # introduced carry no samples. Drop any live tracker before a
                # resume or rollback so post-checkpoint evidence cannot leak
                # into the restored state.
                self.bucket_feedback = None
            completed = int(state.get("completed_iterations", completed))
            if completed < 0 or state.get("env_step", completed * self.cfg["num_steps_per_env"]) != completed * self.cfg["num_steps_per_env"]:
                raise ValueError("adaptive checkpoint step budget mismatch")
            self.last_known_good_checkpoint = state.get("last_known_good_checkpoint")
            buckets = state.get("last_known_good_buckets", ())
            allowed_buckets = set(self.capability_gate.critical_buckets) if self.capability_gate is not None else set()
            if not isinstance(buckets, (list, tuple)) or any(str(name) not in allowed_buckets for name in buckets):
                raise ValueError("adaptive checkpoint known-good bucket mismatch")
            self.last_known_good_buckets = tuple(str(name) for name in buckets)
            self.evaluation_events = list(state.get("evaluation_events", []))
            self.last_evaluation_provenance = state.get("evaluation_provenance")
            # Continue the same evaluation stream after a training restart.
            if state.get("evaluation_seed") is not None:
                self.evaluation_seed = int(state["evaluation_seed"])
            if state.get("evaluation_seed_set_id") is not None:
                self.evaluation_seed_set_id = str(state["evaluation_seed_set_id"])
        elif gate is not None or exposure is not None:
            raise ValueError("resume requires adaptive state; use actor-only load for a warm start")
        self.capability_gate = gate
        self.command_exposure = exposure
        self.completed_iterations = completed
        manager_env = _manager_env(self.env)
        manager_env.common_step_counter = completed * self.cfg["num_steps_per_env"]
        if gate is not None:
            for name in gate.axis_order:
                apply_stage_to_env(manager_env, name, gate.stage_value(name))
        if exposure is not None:
            exposure.apply(manager_env)
        if infos and infos.get("adaptive_rng_state"):
            self._restore_rng_state(infos["adaptive_rng_state"])
        self.resume_checkpoint = str(Path(path).resolve())
        self._needs_reset = True
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
        current_iteration = self.current_learning_iteration
        resume_checkpoint = getattr(self, "resume_checkpoint", None)
        # RSL-RL creates normalizer buffers during inference-mode rollouts.
        # Replace those buffers before load_state_dict copies into them. Keep
        # deserialization outside inference mode so Adam's restored moments
        # remain writable by the next gradient update.
        with torch.inference_mode(False):
            for model in (getattr(self.alg, "actor", None), getattr(self.alg, "critic", None)):
                if model is None:
                    continue
                for name, buffer in model.named_buffers():
                    if buffer.is_inference():
                        parent, _, leaf = name.rpartition(".")
                        setattr(model.get_submodule(parent), leaf, buffer.clone())
            infos = self.load(checkpoint_path)
        self.resume_checkpoint = resume_checkpoint
        self.evaluation_events = history
        self.last_evaluation_provenance = provenance
        if consumed_iterations is not None:
            self.completed_iterations = consumed_iterations
            self.current_learning_iteration = current_iteration
            _manager_env(self.env).common_step_counter = consumed_iterations * self.cfg["num_steps_per_env"]
        # Loading a checkpoint may contain the predecessor's pointer. The
        # explicit rollback target remains the known-good state after restore.
        self.last_known_good_checkpoint = checkpoint_path
        self.evaluation_events.append({"kind": "rollback", "checkpoint": checkpoint_path})
        return infos
