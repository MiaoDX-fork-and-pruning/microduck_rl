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
    AdaptiveActionRateRelief,
    CapabilityGate,
    CommandExposure,
    EntropyConsolidation,
    TransitionExposure,
    apply_stage_to_env,
    evaluation_com_widths,
)
from mjlab_microduck.evaluation.capability import (
    BUCKETS,
    CapabilityReport,
    canonical_sha256,
    resolve_enabled_axes,
)
from . import MicroduckOnPolicyRunner


# ``AdaptiveVelocityCommand`` samples six explicit capability buckets and a
# seventh residual continuous-command pool. Keep the latter visible in the
# evidence contract so its reward mass is not reported as an unexplained
# invalid sample.
COMMAND_FEEDBACK_BUCKETS = (*BUCKETS, "nominal")


def _entropy_coefficient(value: object) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not np.isfinite(value) or value < 0.0
    ):
        raise ValueError("adaptive entropy coefficient must be a finite nonnegative number")
    return float(value)


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


def _validate_cohort_envelope(
    payload: Mapping[str, object],
    *,
    expected_size: int,
    evaluation_seed: int,
    task_id: str,
    axis_mode: str,
    seed_set_id: str,
    checkpoint: Path,
) -> None:
    """Verify the runner received the configured worst-member cohort artifact."""
    metadata = payload.get("metadata")
    cohort = payload.get("cohort_manifest")
    metadata_cohort = metadata.get("cohort") if isinstance(metadata, Mapping) else None
    seed_manifest = payload.get("seed_manifest")
    if not isinstance(metadata_cohort, Mapping) or not isinstance(cohort, Mapping) or not isinstance(seed_manifest, Mapping):
        raise ValueError("cohort report manifest is missing")
    if metadata_cohort != cohort or metadata_cohort.get("version") != "native-reset-dr-cohort-v1":
        raise ValueError("cohort report metadata manifest mismatch")
    expected_seeds = tuple(evaluation_seed + offset for offset in range(expected_size))
    if tuple(metadata_cohort.get("seeds", ())) != expected_seeds:
        raise ValueError("cohort seed manifest does not match configured size")
    if tuple(seed_manifest.get("cohort_seeds", ())) != expected_seeds:
        raise ValueError("cohort seed manifest seed list mismatch")
    members = seed_manifest.get("members")
    if not isinstance(members, list) or len(members) != expected_size:
        raise ValueError("cohort member manifest size mismatch")
    selected = metadata_cohort.get("selected_seed_by_bucket")
    if not isinstance(selected, Mapping) or set(selected) != set(BUCKETS):
        raise ValueError("cohort selected seed map is incomplete")
    member_reports: dict[int, CapabilityReport] = {}
    config_hash: str | None = None
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("invalid cohort member manifest")
        seed_value = member.get("seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, (int, float)) or int(seed_value) != seed_value:
            raise ValueError("invalid cohort member seed")
        seed = int(seed_value)
        if seed not in expected_seeds or seed in member_reports:
            raise ValueError("invalid or duplicate cohort member seed")
        path_value = member.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("cohort member report path is missing")
        path = Path(path_value)
        if not path.is_file():
            raise ValueError("cohort member report is missing")
        member_payload = json.loads(path.read_text(encoding="utf-8"))
        member_hash = member_payload.get("report_sha256")
        if member.get("sha256") != member_hash or member_hash != canonical_sha256({k: v for k, v in member_payload.items() if k != "report_sha256"}):
            raise ValueError("cohort member report hash mismatch")
        member_report = CapabilityReport.from_dict(member_payload)
        member_meta = member_report.payload["metadata"]
        if (
            member_meta.get("task_id") != task_id
            or member_report.payload.get("axis_mode") != axis_mode
            or member_meta.get("seed_set_id") != seed_set_id
            or member_meta.get("checkpoint") != str(checkpoint)
            or member_meta.get("checkpoint_sha256") != metadata.get("checkpoint_sha256")
            or member_meta.get("source_sha") != metadata.get("source_sha")
            or member_meta.get("evaluation_seed") != seed
        ):
            raise ValueError("cohort member provenance mismatch")
        member_config_hash = canonical_sha256(member_report.payload.get("evaluator_config", {}))
        if member_meta.get("evaluator_config_sha256") != member_config_hash:
            raise ValueError("cohort member evaluator config hash mismatch")
        for key in ("distribution", "stage_values", "com_widths", "reference_env_step"):
            if member_report.payload.get("evaluator_config", {}).get(key) != payload.get("evaluator_config", {}).get(key):
                raise ValueError("cohort member evaluation distribution mismatch")
        if config_hash is None:
            config_hash = member_config_hash
        elif member_config_hash != config_hash:
            raise ValueError("cohort member evaluator config mismatch")
        cases = member_payload.get("cases")
        if not isinstance(cases, list) or {case.get("bucket") for case in cases if isinstance(case, Mapping)} != set(BUCKETS):
            raise ValueError("cohort member trace cases are incomplete")
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("invalid cohort member trace case")
            trace_value = case.get("trace")
            trace_hash = case.get("trace_sha256")
            trace = Path(trace_value) if isinstance(trace_value, str) else None
            if trace is None or not isinstance(trace_hash, str) or not trace.is_file():
                raise ValueError("cohort member trace is missing")
            if hashlib.sha256(trace.read_bytes()).hexdigest() != trace_hash:
                raise ValueError("cohort member trace hash mismatch")
        member_reports[seed] = member_report
    for bucket in BUCKETS:
        worst = min(
            ((member_reports[seed].metrics[bucket], seed) for seed in expected_seeds),
            key=lambda item: (item[0], item[1]),
        )
        if int(selected[bucket]) != worst[1]:
            raise ValueError(f"cohort selected seed for {bucket} is not the worst member")
        if canonical_sha256(payload["buckets"][bucket]["raw"]) != canonical_sha256(
            member_reports[worst[1]].payload["buckets"][bucket]["raw"]
        ):
            raise ValueError(f"cohort raw evidence for {bucket} is not from selected member")


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
        saved_terms = tuple(str(name) for name in payload.get("term_names", ()))
        if not saved_terms or len(set(saved_terms)) != len(saved_terms):
            raise ValueError("bucket feedback reward term mismatch")
        # Adaptive-only diagnostics may add a reward term while resuming a
        # checkpoint created by the base recipe.  Preserve the accumulated
        # mass for terms that still exist and initialize the new term at zero.
        # Removing a persisted term is unsafe because its evidence cannot be
        # represented by the destination tracker, so reject that direction.
        if any(name not in self.term_names for name in saved_terms):
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
            signed_values = []
            absolute_values = []
            for name in COMMAND_FEEDBACK_BUCKETS:
                signed_row = [0.0] * len(self.term_names)
                absolute_row = [0.0] * len(self.term_names)
                for saved_index, term in enumerate(saved_terms):
                    current_index = self.term_names.index(term)
                    signed_row[current_index] = float(signed[name][term])
                    absolute_row[current_index] = float(absolute[name][term])
                signed_values.append(signed_row)
                absolute_values.append(absolute_row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bucket feedback payload has invalid values") from exc
        if set(saved_terms) != set(self.term_names) and (
            sum(count_values) > 0 or int(payload.get("unclassified_count", 0)) > 0
        ):
            raise ValueError("empty feedback window required when reward terms change")
        if any(value < 0 for value in count_values) or not np.isfinite(signed_values).all() or not np.isfinite(absolute_values).all():
            raise ValueError("bucket feedback payload has non-finite values")
        self.sample_count.copy_(torch.tensor(count_values, dtype=torch.long, device=self.device))
        self.unclassified_count.fill_(int(payload.get("unclassified_count", 0)))
        self.reward_mass.copy_(torch.tensor(signed_values, dtype=torch.float32, device=self.device))
        self.reward_abs_mass.copy_(torch.tensor(absolute_values, dtype=torch.float32, device=self.device))


class CommandCapabilityEvaluator:
    """Run the frozen battery through a configured command at runner boundaries."""

    def __init__(self, command: str, timeout_s: int = 900, cohort_size: int = 1):
        if not command.strip():
            raise ValueError("adaptive evaluator command cannot be empty")
        self.command = command
        self.timeout_s = int(timeout_s)
        self.cohort_size = int(cohort_size)
        if self.cohort_size < 1:
            raise ValueError("adaptive evaluator cohort size must be positive")

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
            "cohort_size": str(self.cohort_size),
            "distribution": str(curriculum_state.get("evaluation_distribution", "final")),
            "output": str(output),
        }
        # Split the configured argv before substituting paths, so spaces in a
        # checkpoint path remain part of one argument. No shell is involved.
        command = [arg.format(**values) for arg in shlex.split(self.command)]
        # Transition acquisition belongs to training command exposure. The
        # frozen evaluator disables it explicitly; remove launch-only
        # transition settings before spawning the evaluator so a training
        # override cannot make the actor-only evaluation config inconsistent.
        evaluator_env = dict(os.environ)
        for name in (
            "MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY",
            "MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE",
            "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE",
            "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE",
            "MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT",
            "MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION",
            "MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION",
            "MICRODUCK_ADAPTIVE_FINAL_RANGE_FINETUNE",
        ):
            evaluator_env.pop(name, None)
        with (output.parent / "evaluator.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                timeout=self.timeout_s,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=evaluator_env,
            )
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
        self.final_range_finetune = bool(
            getattr(env.cfg, "adaptive_final_range_finetune", False)
        )
        self.final_range_axes = tuple(
            getattr(env.cfg, "adaptive_final_range_axes", ())
        ) if self.final_range_finetune else ()
        if self.final_range_finetune and (
            not self.final_range_axes
            or any(name not in axis_names for name in self.final_range_axes)
            or len(set(self.final_range_axes)) != len(self.final_range_axes)
        ):
            raise ValueError("final-range axes must be a nonempty owned subset")
        if self.final_range_finetune and self.evaluation_interval != 0:
            raise ValueError("final-range fine-tuning requires evaluation_interval=0")
        if self.final_range_finetune and getattr(env.cfg, "adaptive_entropy_consolidation", False):
            raise ValueError("final-range fine-tuning cannot enable entropy consolidation")
        self.evaluator = None
        self.evaluation_seed = int(getattr(env.cfg, "adaptive_evaluation_seed", 0))
        self.evaluation_seed_set_id = str(
            getattr(env.cfg, "adaptive_seed_set_id", str(self.evaluation_seed))
        )
        self.evaluation_cohort_size = int(
            getattr(env.cfg, "adaptive_evaluation_cohort_size", 1)
        )
        if self.evaluation_cohort_size < 1:
            raise ValueError("adaptive evaluation cohort size must be positive")
        self.allow_legacy_cohort_migration = bool(
            getattr(env.cfg, "adaptive_allow_legacy_cohort_migration", False)
        )
        self.evaluation_distribution = getattr(env.cfg, "adaptive_evaluation_distribution", None) or "final"
        self.evaluation_events: list[dict[str, object]] = []
        self.last_evaluation_provenance: dict[str, object] | None = None
        self.last_known_good_checkpoint: str | None = None
        self.last_known_good_buckets: tuple[str, ...] = ()
        self.last_gate_outcome: str | None = None
        self.completed_iterations = 0
        self.resume_checkpoint: str | None = None
        self.bucket_feedback: BucketFeedbackTracker | None = None
        self.entropy_consolidation = (
            EntropyConsolidation()
            if getattr(env.cfg, "adaptive_entropy_consolidation", False) else None
        )
        self._consolidation_needs_baseline = False
        if self.entropy_consolidation is not None and getattr(env.cfg, "adaptive_entropy_coef_override", None) is not None:
            raise ValueError("automatic consolidation cannot be combined with an entropy override")
        if self.entropy_consolidation is not None and not axis_configs:
            raise ValueError("automatic consolidation requires the native adaptive gate")
        self.final_com_fraction = 0.0
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
        self.transition_exposure = (
            TransitionExposure(
                initial_probability=float(
                    getattr(env.cfg, "adaptive_transition_probability", 0.0)
                )
            )
            if getattr(env.cfg, "adaptive_transition_acquisition", False)
            else None
        )
        self.action_rate_relief = (
            AdaptiveActionRateRelief(
                relief_weight=float(getattr(env.cfg, "adaptive_action_rate_relief_weight", -0.2)),
                trigger_threshold=float(getattr(env.cfg, "adaptive_action_rate_relief_trigger", 0.55)),
                release_threshold=float(getattr(env.cfg, "adaptive_action_rate_relief_release", 0.80)),
                active_windows=int(getattr(env.cfg, "adaptive_action_rate_relief_windows", 4)),
                cooldown_windows=int(getattr(env.cfg, "adaptive_action_rate_relief_cooldown_windows", 1)),
                scope=getattr(env.cfg, "adaptive_action_rate_relief_scope", "all"),
            )
            if getattr(env.cfg, "adaptive_action_rate_relief", False)
            else None
        )
        if self.command_exposure is not None:
            self.command_exposure.apply(_manager_env(env))
        if self.transition_exposure is not None:
            self.transition_exposure.apply(_manager_env(env))
        if self.action_rate_relief is not None:
            self.action_rate_relief.apply(_manager_env(env))
        evaluator_command = os.environ.get("MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND") or os.environ.get("MICRODUCK_ADAPTIVE_EVALUATOR")
        if evaluator_command and self.evaluation_interval > 0:
            self.evaluator = CommandCapabilityEvaluator(
                evaluator_command,
                int(getattr(env.cfg, "adaptive_evaluation_timeout_s", 900)),
                self.evaluation_cohort_size,
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
        requested_fraction = getattr(env.cfg, "adaptive_final_com_fraction", None)
        if self.final_range_finetune and (not resume or log_dir is None):
            raise ValueError("final-range fine-tuning requires an explicit resume checkpoint")
        if self.final_range_finetune and requested_fraction not in (None, 0.0):
            raise ValueError("final-range fine-tuning cannot enable final-CoM rehearsal")
        if resume and log_dir is not None:
            self.load(resume, map_location=device)
        if requested_fraction is not None:
            previous_fraction = self.final_com_fraction
            self._set_final_com_fraction(requested_fraction)
            if previous_fraction != self.final_com_fraction:
                self.evaluation_events.append({
                    "kind": "com_rehearsal_override",
                    "previous_fraction": previous_fraction,
                    "final_com_fraction": self.final_com_fraction,
                    "completed_iterations": self.completed_iterations,
                })
                self._needs_reset = True
        requested_transition = getattr(
            env.cfg, "adaptive_transition_probability_override", None
        )
        if requested_transition is not None:
            previous_probability = (
                None
                if self.transition_exposure is None
                else self.transition_exposure.probability
            )
            self._set_transition_probability(float(requested_transition))
            if previous_probability != float(requested_transition):
                self.evaluation_events.append({
                    "kind": "transition_acquisition_override",
                    "previous_probability": previous_probability,
                    "transition_probability": float(requested_transition),
                    "completed_iterations": self.completed_iterations,
                })
                self._needs_reset = True

        if getattr(env.cfg, "adaptive_entropy_coef_override", None) is not None:
            self._apply_entropy_override()

    def _set_entropy_coefficient(self, value: object) -> None:
        coefficient = _entropy_coefficient(value)
        self.alg.entropy_coef = coefficient
        self.cfg.setdefault("algorithm", {})["entropy_coef"] = coefficient

    def _apply_entropy_override(self) -> None:
        """Apply a deliberate launch treatment after restoring checkpoint state."""
        requested = getattr(self.env.cfg, "adaptive_entropy_coef_override", None)
        if requested is None:
            return
        previous = self.alg.entropy_coef
        self._set_entropy_coefficient(requested)
        if previous != self.alg.entropy_coef:
            self.evaluation_events.append({
                "kind": "entropy_override",
                "previous_entropy_coef": previous,
                "entropy_coef": self.alg.entropy_coef,
                "completed_iterations": self.completed_iterations,
            })

    def _set_final_com_fraction(self, fraction: float) -> None:
        """Install rehearsal on live, adaptive-owned axes using stock DR fields."""
        from mjlab.envs.mdp import dr

        from .mdp import randomize_com_with_rehearsal

        if not 0.0 <= fraction <= 0.20:
            raise ValueError("final CoM rehearsal fraction must be in [0, 0.20]")
        gate = self.capability_gate
        axes = gate.axis_order if gate is not None else ()
        if fraction and not axes:
            raise ValueError("final CoM rehearsal requires an adaptive CoM axis")
        # Stock events have already expanded these same fields at construction.
        # Only their reset function and cohort params change; live stage ranges
        # remain owned by apply_stage_to_env.
        for axis in ADAPTIVE_AXIS_CONFIGS:
            if axis.name not in axes:
                continue
            name = {"com_range": "randomize_com", "head_com_range": "randomize_head_com"}[axis.name]
            term = _manager_env(self.env).event_manager.get_term_cfg(name)
            if fraction:
                if term.func not in (dr.body_ipos, randomize_com_with_rehearsal):
                    raise ValueError(f"unsupported CoM rehearsal event: {name}")
                term.func = randomize_com_with_rehearsal
                width = axis.stages[-1]
                term.params.update(final_fraction=fraction, final_ranges=(-width, width))
            elif getattr(term, "func", None) is randomize_com_with_rehearsal:
                term.func = dr.body_ipos
                term.params.pop("final_fraction", None)
                term.params.pop("final_ranges", None)
        self.final_com_fraction = float(fraction)

    def _activate_final_range_finetune(self) -> None:
        """Install canonical final ranges after an explicit full resume."""
        if not getattr(self, "final_range_finetune", False):
            return
        if self.capability_gate is None:
            raise ValueError("final-range fine-tuning requires an adaptive gate")
        step = int(self.completed_iterations) * int(self.cfg["num_steps_per_env"])
        final_axes = tuple(
            getattr(self, "final_range_axes", self.capability_gate.axis_order)
        )
        self.capability_gate.freeze_at_final(step=step, axis_names=final_axes)
        manager_env = _manager_env(self.env)
        for axis_name in self.capability_gate.axis_order:
            apply_stage_to_env(
                manager_env,
                axis_name,
                self.capability_gate.stage_value(axis_name),
            )
        if self.final_com_fraction:
            self._set_final_com_fraction(0.0)
        self.evaluation_distribution = "final"

    def _set_transition_probability(self, probability: float) -> None:
        """Apply a launch override while retaining controller accounting."""
        transition_exposure = getattr(self, "transition_exposure", None)
        if transition_exposure is None:
            if probability == 0.0:
                return
            raise ValueError("transition acquisition override requires command exposure")
        value = float(probability)
        if not np.isfinite(value) or not 0.0 <= value <= transition_exposure.maximum_probability:
            raise ValueError("transition acquisition probability must be finite and in [0, 0.40]")
        transition_exposure.probability = value
        transition_exposure.last_reason = "launch_override"
        transition_exposure.apply(_manager_env(self.env))

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

    def _frozen_capability_report(self, checkpoint_path: str):
        checkpoint_path = str(Path(checkpoint_path).resolve())
        report = self.evaluator.evaluate(
            checkpoint_path=Path(checkpoint_path), task_id=self.env.cfg.task_id,
            axis_mode=self.capability_gate.axis_mode,
            curriculum_state={
                **self.capability_gate.state_dict(),
                "evaluation_distribution": getattr(self, "evaluation_distribution", "final"),
            },
            iteration=self.current_learning_iteration,
            seed_set_id=getattr(self, "evaluation_seed_set_id", str(self.evaluation_seed)),
            evaluation_seed=self.evaluation_seed,
        )
        self._validate_report(report, checkpoint_path)
        return report

    def _begin_entropy_consolidation(self, metrics, checkpoint_path: str) -> None:
        controller = getattr(self, "entropy_consolidation", None)
        if controller is None or not controller.ready(metrics, self.alg.entropy_coef):
            return
        baseline = str(Path(checkpoint_path).with_suffix(".consolidation.pt").resolve())
        # Protect the full trainer before changing entropy; the evaluated file
        # remains immutable and command/curriculum state is not advanced twice.
        self.save(baseline)
        focus_bucket = None
        exposure = getattr(self, "command_exposure", None)
        if exposure is not None:
            focus_bucket = min(
                exposure.frontier_order,
                key=lambda name: (float(metrics[name]), exposure.frontier_order.index(name)),
            )
            exposure.consolidation_focus(focus_bucket, metrics)
            exposure.apply(_manager_env(self.env))
        controller.begin(metrics, entropy_coef=self.alg.entropy_coef, checkpoint=baseline,
                         completed_iterations=self.completed_iterations,
                         window_updates=self.evaluation_interval, focus_bucket=focus_bucket)
        self._set_entropy_coefficient(0.0)
        self.evaluation_events.append({
            "kind": "entropy_consolidation_started", "state": controller.state_dict(),
            "evaluation_checkpoint": checkpoint_path, "focus_bucket": focus_bucket,
        })

    def _finish_entropy_consolidation(self, controller, metrics, *, gate_retained: bool) -> None:
        if controller is None or controller.phase != "active" or self.completed_iterations < controller.deadline:
            return
        controller.finish(metrics, gate_retained=gate_retained,
                          completed_iterations=self.completed_iterations)
        if controller.phase == "rejected":
            self.last_known_good_checkpoint = controller.baseline_checkpoint
            self.rollback(controller.baseline_checkpoint)
            self._set_entropy_coefficient(controller.original_entropy_coef)
        self.entropy_consolidation = controller
        self._consolidation_needs_baseline = False
        self.evaluation_events.append({
            "kind": "entropy_consolidation_stopped", "state": controller.state_dict(),
            "completed_iterations": self.completed_iterations,
        })

    def _bootstrap_entropy_consolidation(self) -> None:
        if not getattr(self, "_consolidation_needs_baseline", False):
            return
        if self.evaluator is None or self.evaluation_interval <= 0:
            raise ValueError("resumed automatic consolidation requires a native evaluator")
        checkpoint = str((Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.consolidation-bootstrap.pt").resolve())
        self.save(checkpoint)
        report = self._frozen_capability_report(checkpoint)
        metrics = report.to_gate_metrics()
        # Historical best_metrics may combine different actors. Re-evaluate the
        # actual retained snapshot without consuming another teacher/gate window.
        if not self.capability_gate.preservation_failures(metrics):
            self._begin_entropy_consolidation(metrics, checkpoint)
        self._consolidation_needs_baseline = False

    def _evaluate_window(self, checkpoint_path: str) -> None:
        if self.evaluation_interval <= 0 or self.evaluator is None or self.capability_gate is None:
            return
        checkpoint_path = str(Path(checkpoint_path).resolve())
        consolidation = getattr(self, "entropy_consolidation", None)
        try:
            report = self._frozen_capability_report(checkpoint_path)
            payload = getattr(report, "payload", report)
            metrics = report.to_gate_metrics()
            command_feedback = self._snapshot_bucket_feedback(reset=True)
        except Exception as exc:
            self.last_gate_outcome = "evaluation_error"
            self.evaluation_events.append({"kind": "evaluation_error", "error": repr(exc), "iteration": self.current_learning_iteration, "checkpoint": checkpoint_path})
            self._finish_entropy_consolidation(consolidation, None, gate_retained=False)
        else:
            self.last_evaluation_provenance = {
                **payload["metadata"],
                "report_sha256": report.sha256(),
                "evaluation_seed": self.evaluation_seed,
                "schema_version": payload["schema_version"],
                "report_path": str(Path(checkpoint_path).parent / "adaptive_eval" / Path(checkpoint_path).stem / "capability.json"),
            }
            previous_best = self.capability_gate.best_metrics.copy()
            transition = self.record_capability_metrics(
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
            stage_changed = transition is not None and getattr(self, "evaluation_distribution", "final") == "stage"
            if stage_changed:
                # The report proves the old stage. Do not label the newly
                # applied difficulty mastered or compare its first score with
                # an easier distribution's best score/EMA/rollback target.
                self.evaluation_events.append({
                    "kind": "stage_evidence_rebaseline",
                    "transition": transition.as_dict(),
                    "previous_best_metrics": dict(self.capability_gate.best_metrics),
                    "previous_known_good_checkpoint": self.last_known_good_checkpoint,
                    "previous_known_good_buckets": list(self.last_known_good_buckets),
                })
                self.capability_gate.reset_evidence(step=transition.step)
                self.last_known_good_checkpoint = None
                self.last_known_good_buckets = ()
                exposure = getattr(self, "command_exposure", None)
                if exposure is not None:
                    exposure.focus_best_score = None
                    exposure.focus_stall_count = 0
            elif (mastered_before | mastered_now) and preserved and self.last_gate_outcome != "preservation_failure":
                self.last_known_good_checkpoint = str(Path(checkpoint_path).with_suffix(".adaptive.pt"))
                self.last_known_good_buckets = tuple(sorted(mastered_before | mastered_now))
            # Policy rollback must not rewind the live consolidation budget.
            self.entropy_consolidation = consolidation
            retained = preserved and self.last_gate_outcome != "preservation_failure"
            if consolidation is not None and consolidation.phase == "active":
                self._set_entropy_coefficient(0.0)
                self._finish_entropy_consolidation(consolidation, metrics, gate_retained=retained)
            elif retained and not stage_changed:
                self._begin_entropy_consolidation(metrics, checkpoint_path)
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

        expected_task = getattr(self.env.cfg, "task_id", None)
        expected_seed = int(self.evaluation_seed)
        expected_seed_set = getattr(self, "evaluation_seed_set_id", None)
        distribution = getattr(self, "evaluation_distribution", "final")
        config = payload.get("evaluator_config", {})
        if config.get("distribution", distribution) != distribution:
            raise ValueError("capability report evaluation distribution mismatch")
        if distribution == "stage":
            stage_values = {name: self.capability_gate.stage_value(name) for name in self.capability_gate.axis_order}
            if (
                config.get("distribution") != "stage"
                or config.get("stage_values") != stage_values
                or config.get("com_widths") != evaluation_com_widths("stage", mode, stage_values)
                or config.get("reference_env_step") != 96000
            ):
                raise ValueError("capability report stage distribution mismatch")
            if metadata.get("evaluator_config_sha256") != canonical_sha256(config):
                raise ValueError("capability report evaluator config hash mismatch")
        if int(getattr(self, "evaluation_cohort_size", 1)) > 1:
            _validate_cohort_envelope(
                payload,
                expected_size=int(self.evaluation_cohort_size),
                evaluation_seed=expected_seed,
                task_id=expected_task,
                axis_mode=mode,
                seed_set_id=expected_seed_set,
                checkpoint=expected,
            )
            CapabilityReport.from_dict(payload).to_gate_metrics()
        else:
            CapabilityReport.from_dict(payload).to_gate_metrics()
        if expected_task and metadata["task_id"] != expected_task:
            raise ValueError("capability report task mismatch")
        if hasattr(self, "evaluation_seed_set_id") and metadata["seed_set_id"] != self.evaluation_seed_set_id:
            raise ValueError("capability report seed set mismatch")
        if "evaluation_seed" not in metadata and int(getattr(self, "evaluation_cohort_size", 1)) > 1:
            raise ValueError("capability report evaluation seed is missing")
        if "evaluation_seed" in metadata and metadata.get("evaluation_seed") != expected_seed:
            raise ValueError("capability report evaluation seed mismatch")
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
        consolidation = getattr(self, "entropy_consolidation", None)
        if consolidation is not None and consolidation.terminal:
            raise ValueError("consolidation attempt is terminal; do not extend it unchanged")
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
        self._bootstrap_entropy_consolidation()
        start_it = self.completed_iterations
        total_it = start_it + num_learning_iterations
        consolidation = getattr(self, "entropy_consolidation", None)
        if consolidation is not None and consolidation.phase == "active" and start_it == consolidation.deadline:
            # A crash can leave a pre-evaluation candidate at the deadline.
            # Re-evaluate it before collecting anything; spent updates stay spent.
            checkpoint = os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.eval.pt")
            self.save(checkpoint)
            self._evaluate_window(checkpoint)
        for it in range(start_it, total_it):
            consolidation = getattr(self, "entropy_consolidation", None)
            if consolidation is not None and consolidation.terminal:
                break
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
            consolidation = getattr(self, "entropy_consolidation", None)
            consolidation_due = (
                consolidation is not None and consolidation.phase == "active"
                and self.completed_iterations >= consolidation.deadline
            )
            if self.capability_gate is not None and self.evaluation_interval > 0 and (
                self.completed_iterations % self.evaluation_interval == 0 or consolidation_due
            ):
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
                consolidation = getattr(self, "entropy_consolidation", None)
                if consolidation is not None and consolidation.terminal:
                    break
        if self.logger.writer is not None:
            checkpoint = Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"
            self.save(str(checkpoint))
            self._write_training_result(checkpoint, start_it, requested_iterations=total_it)
            self.logger.stop_logging_writer()

    def _reset_after_rollback(self):
        """Reset stateful simulator buffers in rollout's inference context."""
        with torch.inference_mode():
            return self.env.reset()

    def _write_training_result(self, checkpoint: Path, start_iteration: int, *, requested_iterations: int | None = None) -> None:
        """Publish completion only after the final checkpoint is durably written."""
        checkpoint = checkpoint.resolve()
        requested = self.completed_iterations if requested_iterations is None else requested_iterations
        consolidation = getattr(self, "entropy_consolidation", None)
        stop_reason = "entropy_consolidation_terminal" if consolidation is not None and consolidation.terminal else None
        result = {
            "version": 1, "status": "stopped" if self.completed_iterations < requested else "completed", "task_id": self.env.cfg.task_id,
            "requested_completed_iterations": requested, "stop_reason": stop_reason,
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
        regressed_buckets = self.capability_gate.preservation_failures(metrics)
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
        if regressed_buckets:
            event["regressed_buckets"] = list(regressed_buckets)
        if command_feedback is not None:
            event["command_feedback"] = dict(command_feedback)
        if getattr(self, "last_evaluation_provenance", None) is not None:
            event["provenance"] = dict(self.last_evaluation_provenance)
        self.evaluation_events.append(event)
        if decision.outcome.value == "preservation_failure":
            # Exposure is the adaptive teacher's live state, rather than part
            # of the policy snapshot being protected by rollback.  Preserve it
            # across the policy/gate restore so repeated retention failures
            # accumulate bounded repair slices instead of replaying the stale
            # exposure from the old known-good checkpoint.
            exposure = getattr(self, "command_exposure", None)
            exposure_state = None if exposure is None else exposure.state_dict()
            transition_exposure = getattr(self, "transition_exposure", None)
            transition_state = (
                None if transition_exposure is None else transition_exposure.state_dict()
            )
            action_rate_relief = getattr(self, "action_rate_relief", None)
            action_rate_relief_state = (
                None if action_rate_relief is None else action_rate_relief.state_dict()
            )
            final_com_fraction = getattr(self, "final_com_fraction", 0.0)
            if self.last_known_good_checkpoint:
                self.rollback(self.last_known_good_checkpoint)
            # This is the live training distribution, like exposure. Old
            # rollback targets must not silently switch off a new experiment.
            self._set_final_com_fraction(final_com_fraction)
            event["final_com_fraction"] = final_com_fraction
            exposure = getattr(self, "command_exposure", None)
            if exposure is not None and exposure_state is not None:
                exposure.load_state_dict(exposure_state)
                exposure.apply(_manager_env(self.env))
            transition_exposure = getattr(self, "transition_exposure", None)
            if transition_exposure is not None and transition_state is not None:
                transition_exposure.load_state_dict(transition_state)
                transition_exposure.apply(_manager_env(self.env))
            repair_buckets = tuple(name for name in regressed_buckets if name != "zero")
            if exposure is not None and repair_buckets:
                repaired = exposure.repair(
                    repair_buckets,
                    metrics,
                    feedback=command_feedback,
                )
                exposure.apply(_manager_env(self.env))
                event["retention_repair"] = list(repaired)
                event["command_exposure"] = exposure.state_dict()
            if transition_exposure is not None:
                if transition_exposure.repair(repair_buckets):
                    transition_exposure.apply(_manager_env(self.env))
                event["transition_exposure"] = transition_exposure.state_dict()
            action_rate_relief = getattr(self, "action_rate_relief", None)
            if action_rate_relief is not None and action_rate_relief_state is not None:
                action_rate_relief.load_state_dict(action_rate_relief_state)
                # Successful buckets from a rejected candidate do not prove
                # that the restored actor has acquired them. Account for the
                # consumed window without releasing relief on that evidence.
                action_rate_relief.update(metrics, accepted=False)
                action_rate_relief.apply(_manager_env(self.env))
                event["action_rate_relief"] = action_rate_relief.state_dict()
            return None
        exposure = getattr(self, "command_exposure", None)
        if exposure is not None:
            exposure.update(metrics)
            exposure.apply(_manager_env(self.env))
            event["command_exposure"] = exposure.state_dict()
        transition_exposure = getattr(self, "transition_exposure", None)
        if transition_exposure is not None:
            transition_exposure.update(metrics)
            transition_exposure.apply(_manager_env(self.env))
            event["transition_exposure"] = transition_exposure.state_dict()
        action_rate_relief = getattr(self, "action_rate_relief", None)
        if action_rate_relief is not None:
            action_rate_relief.update(metrics)
            action_rate_relief.apply(_manager_env(self.env))
            event["action_rate_relief"] = action_rate_relief.state_dict()
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
        sensor_reset_fraction = float(
            getattr(self.env.cfg, "adaptive_sensor_reset_fraction", 0.0)
        )
        if not np.isfinite(sensor_reset_fraction) or not 0.0 <= sensor_reset_fraction <= 1.0:
            raise ValueError("adaptive sensor reset fraction must be finite and in [0, 1]")
        state = gate.state_dict() if gate is not None else {"axis_mode": "all_static", "enabled_axes": []}
        state.update({
            "version": 1,
            "task_id": getattr(self.env.cfg, "task_id", None),
            "num_envs": getattr(self.env, "num_envs", None),
            "stage_values": {} if gate is None else {name: gate.stage_value(name) for name in gate.axis_order},
            "command_exposure": None if exposure is None else exposure.state_dict(),
            "transition_exposure": None
            if getattr(self, "transition_exposure", None) is None
            else self.transition_exposure.state_dict(),
            "action_rate_relief": None
            if getattr(self, "action_rate_relief", None) is None
            else self.action_rate_relief.state_dict(),
            "entropy_consolidation": None
            if getattr(self, "entropy_consolidation", None) is None
            else self.entropy_consolidation.state_dict(),
            "transition_bootstrap_mode": getattr(
                self.env.cfg, "adaptive_transition_bootstrap_mode", "forward"
            ),
            # Sensor reset coverage changes the consumed training distribution.
            # Persist it beside the controller state so a full resume cannot
            # silently fall back to the ordinary fixed realization.
            "sensor_reset_fraction": sensor_reset_fraction,
            "final_com_fraction": getattr(self, "final_com_fraction", 0.0),
            "final_range_finetune": bool(getattr(self, "final_range_finetune", False)),
            "final_range_axes": list(getattr(self, "final_range_axes", ())),
            "training_mode": (
                "final_range_finetune"
                if getattr(self, "final_range_finetune", False)
                else "adaptive"
            ),
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
            "evaluation_cohort_size": getattr(self, "evaluation_cohort_size", 1),
            "evaluation_distribution": getattr(self, "evaluation_distribution", "final"),
            "env_step": completed * int(getattr(self, "cfg", {}).get("num_steps_per_env", 24)),
        })
        # RSL-RL saves model/optimizer tensors but not this PPO loss coefficient.
        # Save the live value so a restarted consolidation run stays identical.
        if hasattr(getattr(self, "alg", None), "entropy_coef"):
            state["entropy_coef"] = _entropy_coefficient(self.alg.entropy_coef)
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
        transition_exposure = deepcopy(getattr(self, "transition_exposure", None))
        action_rate_relief = deepcopy(getattr(self, "action_rate_relief", None))
        consolidation = deepcopy(getattr(self, "entropy_consolidation", None))
        consolidation_needs_baseline = False
        final_range_finetune = getattr(self, "final_range_finetune", False)
        completed = self.current_learning_iteration + 1
        distribution = getattr(self, "evaluation_distribution", "final")
        entropy_coef = None
        override = getattr(self.env.cfg, "adaptive_entropy_coef_override", None)
        if override is not None:
            _entropy_coefficient(override)
        if state:
            saved_final_range_finetune = bool(state.get("final_range_finetune", False))
            if saved_final_range_finetune and not final_range_finetune:
                raise ValueError(
                    "final-range fine-tuning checkpoint requires explicit fine-tuning mode"
                )
            saved_mode = state.get(
                "training_mode",
                "final_range_finetune" if saved_final_range_finetune else "adaptive",
            )
            expected_mode = (
                "final_range_finetune" if saved_final_range_finetune else "adaptive"
            )
            if saved_mode != expected_mode:
                raise ValueError("adaptive checkpoint training mode is invalid")
            saved_final_axes = tuple(state.get("final_range_axes", ()))
            if saved_final_range_finetune and saved_final_axes and (
                tuple(self.final_range_axes) != saved_final_axes
            ):
                raise ValueError("adaptive checkpoint final-range axes mismatch")
            if "entropy_coef" in state:
                entropy_coef = _entropy_coefficient(state["entropy_coef"])
            saved_consolidation = state.get("entropy_consolidation")
            if saved_consolidation is not None:
                if final_range_finetune:
                    # This is a new fixed-range window, not an extension of a
                    # terminal consolidation attempt. Drop its live controller
                    # while retaining the source path in resume metadata.
                    consolidation = None
                elif consolidation is None:
                    raise ValueError("adaptive checkpoint entropy consolidation mismatch")
                else:
                    consolidation.load_state_dict(saved_consolidation)
                    if consolidation.phase == "active" and (
                        entropy_coef != 0.0 or self.evaluation_interval != consolidation.window_updates
                    ):
                        raise ValueError("active consolidation entropy or window mismatch")
            elif consolidation is not None and not final_range_finetune:
                consolidation = EntropyConsolidation()
                consolidation_needs_baseline = True
            elif final_range_finetune:
                consolidation = None
            if state.get("version", 1) != 1:
                raise ValueError("unsupported adaptive checkpoint version")
            if state.get("task_id") and state["task_id"] != getattr(self.env.cfg, "task_id", None):
                raise ValueError("adaptive checkpoint task mismatch")
            if int(state.get("evaluation_schema_version", 2)) != int(getattr(self.env.cfg, "adaptive_evaluator_schema_version", 2)):
                raise ValueError("adaptive checkpoint evaluator schema mismatch")
            saved_distribution = state.get("evaluation_distribution", "final")
            distribution = getattr(self.env.cfg, "adaptive_evaluation_distribution", None) or saved_distribution
            if saved_distribution not in ("final", "stage") or distribution not in ("final", "stage"):
                raise ValueError("invalid adaptive checkpoint evaluation distribution")
            distribution_migration = saved_distribution != distribution
            if distribution_migration and not getattr(self.env.cfg, "adaptive_allow_distribution_migration", False):
                raise ValueError("adaptive checkpoint evaluation distribution mismatch; explicit migration required")
            saved_sensor_fraction = float(state.get("sensor_reset_fraction", 0.0))
            current_sensor_fraction = float(
                getattr(self.env.cfg, "adaptive_sensor_reset_fraction", 0.0)
            )
            if (
                not np.isfinite(saved_sensor_fraction)
                or not 0.0 <= saved_sensor_fraction <= 1.0
                or not np.isfinite(current_sensor_fraction)
                or not 0.0 <= current_sensor_fraction <= 1.0
            ):
                raise ValueError("adaptive checkpoint sensor reset fraction is invalid")
            if saved_sensor_fraction != current_sensor_fraction:
                raise ValueError(
                    "adaptive checkpoint sensor reset fraction mismatch; "
                    "recreate the environment with the checkpoint setting"
                )
            if saved_sensor_fraction > 0.0:
                try:
                    sensor_term = _manager_env(self.env).event_manager.get_term_cfg(
                        "adaptive_sensor_resample"
                    )
                except (AttributeError, KeyError, AssertionError) as exc:
                    raise ValueError(
                        "adaptive checkpoint sensor reset event is missing"
                    ) from exc
                live_fraction = float(sensor_term.params.get("fraction", 0.0))
                if live_fraction != saved_sensor_fraction:
                    raise ValueError(
                        "adaptive checkpoint sensor reset event fraction mismatch"
                    )
            saved_cohort_size = state.get("evaluation_cohort_size")
            legacy_cohort_migration = (
                saved_cohort_size is None and self.evaluation_cohort_size > 1
            )
            # Checkpoints written before cohort gates existed have no field and
            # may be upgraded explicitly by a campaign launch.  Once a cohort
            # size is persisted, changing it would make the gate history
            # incomparable and is rejected.
            if legacy_cohort_migration and not self.allow_legacy_cohort_migration:
                raise ValueError(
                    "legacy adaptive checkpoint requires explicit cohort migration"
                )
            if (
                saved_cohort_size is not None
                and int(saved_cohort_size) != getattr(self, "evaluation_cohort_size", 1)
            ):
                raise ValueError("adaptive checkpoint evaluation cohort size mismatch")
            legacy_gate_audit = None
            legacy_known_good = None
            if gate is not None:
                gate.load_state_dict(state)
                if legacy_cohort_migration or distribution_migration:
                    # Keep the learned PPO policy, optimizer, stage difficulty,
                    # exposure and cumulative step budget. Rebaseline only the
                    # evidence that was measured under the old single-seed
                    # gate: old mastery, dwell counters and rollback targets
                    # must not be allowed to trigger a false cohort rollback.
                    legacy_gate_audit = {
                        "best_metrics": deepcopy(gate.best_metrics),
                        "states": deepcopy(state.get("states", {})),
                        "trace_length": len(gate.trace),
                    }
                    gate.reset_evidence()
                    legacy_known_good = {
                        "checkpoint": state.get("last_known_good_checkpoint"),
                        "buckets": list(state.get("last_known_good_buckets", ())),
                    }
            elif state.get("enabled_axes"):
                raise ValueError("adaptive state axis mode/allowlist mismatch")
            if bool(state.get("command_exposure")) != (exposure is not None):
                raise ValueError("adaptive checkpoint command exposure mismatch")
            if exposure is not None:
                exposure.load_state_dict(state["command_exposure"])
                if distribution_migration:
                    exposure.focus_best_score = None
                    exposure.focus_stall_count = 0
            saved_transition = state.get("transition_exposure")
            saved_bootstrap_mode = str(state.get("transition_bootstrap_mode", "forward"))
            current_bootstrap_mode = str(
                getattr(self.env.cfg, "adaptive_transition_bootstrap_mode", "forward")
            )
            if saved_bootstrap_mode not in ("forward", "zero"):
                raise ValueError("invalid adaptive checkpoint transition bootstrap mode")
            if (
                saved_bootstrap_mode != current_bootstrap_mode
                and not getattr(self.env.cfg, "adaptive_transition_bootstrap_mode_override", False)
            ):
                raise ValueError("adaptive checkpoint transition bootstrap mode mismatch")
            if saved_transition is not None:
                if transition_exposure is None:
                    # A zero-probability state is behaviorally disabled. It
                    # may be loaded by an older command-exposure config that
                    # does not construct the optional controller; preserve
                    # compatibility while still rejecting a live mechanism
                    # that the destination cannot restore.
                    saved_probability = float(saved_transition.get("probability", -1.0))
                    if saved_probability != 0.0:
                        raise ValueError("adaptive checkpoint transition exposure mismatch")
                else:
                    transition_exposure.load_state_dict(saved_transition)
            elif transition_exposure is not None:
                # An explicit load of a legacy checkpoint disables the
                # mechanism. A launch override, if any, is applied by the
                # constructor after this full restore; retaining the live
                # probability here would make load order observable.
                transition_exposure.load_state_dict({
                    "version": transition_exposure.version,
                    "probability": 0.0,
                    "windows": 0,
                    "repairs": 0,
                    "last_reason": "legacy_checkpoint_bootstrap",
                })
            saved_action_rate_relief = state.get("action_rate_relief")
            if saved_action_rate_relief is not None:
                if action_rate_relief is None:
                    raise ValueError("adaptive checkpoint action-rate relief mismatch")
                else:
                    action_rate_relief.load_state_dict(saved_action_rate_relief)
            elif action_rate_relief is not None:
                # Legacy adaptive checkpoints predate this controller. Seed a
                # bounded window from their measured yaw deficit so a resume
                # can repair the missing behavior immediately rather than
                # waiting for the next evaluation interval.
                action_rate_relief.reset()
                if gate is not None and set(action_rate_relief.yaw_buckets) <= set(gate.best_metrics):
                    action_rate_relief.bootstrap(gate.best_metrics)
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
            if consolidation is not None and consolidation.phase != "waiting":
                if not consolidation.start_iterations <= completed <= consolidation.deadline:
                    raise ValueError("consolidation update budget mismatch")
                if consolidation.terminal and completed != consolidation.deadline:
                    raise ValueError("terminal consolidation window is incomplete")
            self.last_known_good_checkpoint = state.get("last_known_good_checkpoint")
            buckets = state.get("last_known_good_buckets", ())
            allowed_buckets = set(self.capability_gate.critical_buckets) if self.capability_gate is not None else set()
            if not isinstance(buckets, (list, tuple)) or any(str(name) not in allowed_buckets for name in buckets):
                raise ValueError("adaptive checkpoint known-good bucket mismatch")
            self.last_known_good_buckets = tuple(str(name) for name in buckets)
            self.evaluation_events = list(state.get("evaluation_events", []))
            if final_range_finetune and not saved_final_range_finetune:
                self.evaluation_events.append({
                    "kind": "final_range_finetune_source",
                    "source_checkpoint": str(Path(path).resolve()),
                    "source_consolidation_phase": (saved_consolidation or {}).get("phase"),
                    "source_consolidation_deadline": (saved_consolidation or {}).get("deadline"),
                    "source_stage_values": state.get("stage_values"),
                    "completed_iterations": completed,
                })
            self.last_evaluation_provenance = state.get("evaluation_provenance")
            if legacy_cohort_migration or distribution_migration:
                self.last_known_good_checkpoint = None
                self.last_known_good_buckets = ()
                self.last_evaluation_provenance = None
                self.evaluation_events.append({
                    "kind": "distribution_rebaseline" if distribution_migration else "cohort_rebaseline",
                    "from_distribution": saved_distribution,
                    "to_distribution": distribution,
                    "from_cohort_size": state.get("evaluation_cohort_size", 1),
                    "to_cohort_size": getattr(self, "evaluation_cohort_size", 1),
                    "completed_iterations": completed,
                    "legacy_gate": legacy_gate_audit,
                    "legacy_known_good": legacy_known_good,
                })
            # Continue the same evaluation stream after a training restart.
            if state.get("evaluation_seed") is not None:
                self.evaluation_seed = int(state["evaluation_seed"])
            if state.get("evaluation_seed_set_id") is not None and not legacy_cohort_migration:
                self.evaluation_seed_set_id = str(state["evaluation_seed_set_id"])
        elif gate is not None or exposure is not None or transition_exposure is not None or action_rate_relief is not None:
            raise ValueError("resume requires adaptive state; use actor-only load for a warm start")
        self.capability_gate = gate
        self.evaluation_distribution = distribution
        self.command_exposure = exposure
        self.transition_exposure = transition_exposure
        self.action_rate_relief = action_rate_relief
        self.entropy_consolidation = consolidation
        self._consolidation_needs_baseline = consolidation_needs_baseline
        self.completed_iterations = completed
        manager_env = _manager_env(self.env)
        manager_env.common_step_counter = completed * self.cfg["num_steps_per_env"]
        if gate is not None:
            for name in gate.axis_order:
                apply_stage_to_env(manager_env, name, gate.stage_value(name))
        if exposure is not None:
            exposure.apply(manager_env)
        if transition_exposure is not None:
            transition_exposure.apply(manager_env)
        if action_rate_relief is not None:
            action_rate_relief.apply(manager_env)
        saved_fraction = (state or {}).get("final_com_fraction", 0.0)
        self._set_final_com_fraction(0.0 if saved_fraction is None else float(saved_fraction))
        if infos and infos.get("adaptive_rng_state"):
            self._restore_rng_state(infos["adaptive_rng_state"])
        self.resume_checkpoint = str(Path(path).resolve())
        self._needs_reset = True
        if entropy_coef is not None:
            self._set_entropy_coefficient(entropy_coef)
        # Legacy checkpoints have no entropy field: keep the launch coefficient.
        # An explicit treatment overrides full loads (including rollback), while
        # ordinary resumes/rollbacks restore the saved trainer coefficient.
        self._apply_entropy_override()
        self._activate_final_range_finetune()
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
