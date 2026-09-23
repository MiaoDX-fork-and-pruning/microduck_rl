"""Capability-gated curriculum state used by the MJLab adaptive experiment.

The gate is deliberately independent of MJLab and torch.  An evaluator feeds it
fixed-seed, bucket-level metrics at evaluation windows; the training adapter can
then apply the returned one-stage transition to live manager term configs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence
from enum import StrEnum

from mjlab_microduck.evaluation.capability import BUCKETS, resolve_enabled_axes


class EntropyConsolidation:
    """One measured consolidation attempt, followed by a campaign checkpoint.

    The .60 acquisition boundary and .80 mastery boundary match the existing
    native gate. Neither is a new acceptance threshold. A full evaluation
    window at zero entropy is retained only if the gate preserves acquired
    skills and the weakest bucket improves by .05 (or all buckets master).
    Every attempt terminates, including failed evaluation, so a large requested
    budget cannot silently extend an ineffective consolidation treatment.
    """

    version = 2

    def __init__(self) -> None:
        self.phase = "waiting"
        self.baseline_metrics: dict[str, float] = {}
        self.baseline_checkpoint: str | None = None
        self.original_entropy_coef: float | None = None
        self.start_iterations: int | None = None
        self.window_updates: int | None = None
        self.focus_bucket: str | None = None
        self.result_metrics: dict[str, float] | None = None
        self.reason: str | None = None

    @staticmethod
    def _metrics(metrics: Mapping[str, float]) -> dict[str, float]:
        values = {name: float(metrics[name]) for name in BUCKETS}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("invalid consolidation capability scores")
        return values

    @property
    def terminal(self) -> bool:
        return self.phase in ("retained", "rejected")

    @property
    def deadline(self) -> int | None:
        if self.start_iterations is None or self.window_updates is None:
            return None
        return self.start_iterations + self.window_updates

    def ready(self, metrics: Mapping[str, float], entropy_coef: float) -> bool:
        values = self._metrics(metrics)
        return (
            self.phase == "waiting" and entropy_coef > 0.0
            and values["zero"] >= 0.80 and 0.60 <= min(values.values()) < 0.80
        )

    def begin(self, metrics: Mapping[str, float], *, entropy_coef: float,
              checkpoint: str, completed_iterations: int, window_updates: int,
              focus_bucket: str | None = None) -> None:
        if not self.ready(metrics, entropy_coef):
            raise ValueError("consolidation requires acquired but unmastered capabilities")
        if (not math.isfinite(entropy_coef) or not checkpoint
                or type(completed_iterations) is not int or completed_iterations < 0
                or type(window_updates) is not int or window_updates < 1):
            raise ValueError("invalid consolidation start contract")
        if focus_bucket is not None and focus_bucket not in BUCKETS[1:]:
            raise ValueError("consolidation focus must be directional")
        self.phase = "active"
        self.baseline_metrics = self._metrics(metrics)
        self.baseline_checkpoint = checkpoint
        self.original_entropy_coef = float(entropy_coef)
        self.start_iterations = completed_iterations
        self.window_updates = window_updates
        self.focus_bucket = focus_bucket

    def finish(self, metrics: Mapping[str, float] | None, *, gate_retained: bool,
               completed_iterations: int) -> None:
        if self.phase != "active" or completed_iterations < self.deadline:
            raise ValueError("consolidation window has not completed")
        values = None if metrics is None else self._metrics(metrics)
        self.result_metrics = values
        improved = values is not None and (
            min(values.values()) >= 0.80
            or min(values.values()) >= min(self.baseline_metrics.values()) + 0.05
        )
        self.phase = "retained" if gate_retained and improved else "rejected"
        self.reason = (
            "evaluation_failed" if values is None
            else "preservation_failed" if not gate_retained
            else "weakest_capability_improved" if improved
            else "insufficient_capability_gain"
        )

    def state_dict(self) -> dict[str, object]:
        return {"version": self.version, **vars(self), "baseline_metrics": dict(self.baseline_metrics),
                "result_metrics": None if self.result_metrics is None else dict(self.result_metrics)}

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("version") == 1:
            payload = {**payload, "version": self.version, "focus_bucket": None}
        if payload.get("version") != self.version or set(payload) != set(self.state_dict()):
            raise ValueError("invalid consolidation state schema")
        phase = payload["phase"]
        if phase not in ("waiting", "active", "retained", "rejected"):
            raise ValueError("invalid consolidation phase")
        restored = EntropyConsolidation()
        if phase == "waiting":
            if dict(payload) != restored.state_dict():
                raise ValueError("waiting consolidation contains an old attempt")
        else:
            for key in ("start_iterations", "window_updates"):
                if type(payload[key]) is not int:
                    raise ValueError("invalid consolidation update budget")
            coefficient = payload["original_entropy_coef"]
            if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
                raise ValueError("invalid consolidation entropy value")
            checkpoint = payload["baseline_checkpoint"]
            if not isinstance(checkpoint, str):
                raise ValueError("invalid consolidation baseline path")
            restored.begin(payload["baseline_metrics"], entropy_coef=coefficient,
                           checkpoint=checkpoint, completed_iterations=payload["start_iterations"],
                           window_updates=payload["window_updates"], focus_bucket=payload["focus_bucket"])
            if phase == "active":
                if payload["reason"] is not None or payload["result_metrics"] is not None:
                    raise ValueError("active consolidation contains a terminal verdict")
            else:
                restored.finish(payload["result_metrics"],
                                gate_retained=payload["reason"] not in ("preservation_failed", "evaluation_failed"),
                                completed_iterations=restored.deadline)
                if restored.phase != phase or restored.reason != payload["reason"]:
                    raise ValueError("consolidation verdict does not match its evidence")
        self.__dict__.update(restored.__dict__)


class CommandExposure:
    """Keep learned commands alive while concentrating on one frontier.

    Twenty percent of resamples retain the nominal continuous command
    distribution.  The zero-command recovery anchor keeps twenty percent;
    each directional bucket keeps eight percent, and the current frontier
    receives the remaining twenty percent.  A window moves one quarter of the
    way to the new target, so a focus switch cannot erase a previously learned
    skill in one update.

    ``zero`` is deliberately an anchor rather than a frontier: the recovery
    and idle behavior must remain present while the controller acquires the
    directional buckets in ``frontier_order``. Configured consolidation dwell
    yields early when another direction has a substantially lower capability.
    """

    version = 2
    nominal_probability = 0.20
    bucket_floor = 0.08
    zero_floor = 0.20
    focus_extra = 0.20
    update_rate = 0.25
    # Match CapabilityGate's upper threshold.  A bucket scoring 0.5 is making
    # measurable progress, but it is not mastered: handing focus away there
    # strands near-pass capabilities below the product acceptance boundary.
    focus_mastery = 0.80
    # Interrupt consolidation only for a large capability gap. This controls
    # sampling urgency, independently of the unchanged mastery/pass threshold.
    focus_preemption_gap = 0.25
    frontier_order = ("forward", "lateral", "yaw", "turn-left", "turn-right")

    def __init__(
        self,
        initial_focus: str | None = None,
        frontier_order: tuple[str, ...] | None = None,
        stall_windows: int = 0,
        stall_improvement: float = 0.05,
    ) -> None:
        if stall_windows < 0:
            raise ValueError("stall_windows cannot be negative")
        if not math.isfinite(stall_improvement) or stall_improvement < 0.0:
            raise ValueError("stall_improvement must be finite and nonnegative")
        self.frontier_order = self._validate_frontier_order(frontier_order)
        self.stall_windows = int(stall_windows)
        self.stall_improvement = float(stall_improvement)
        self.focus_bucket = self.frontier_order[0] if initial_focus is None else initial_focus
        if self.focus_bucket not in self.frontier_order:
            raise ValueError(f"unsupported frontier bucket: {self.focus_bucket}")
        self.probabilities = self._target(self.focus_bucket)
        self.windows = 0
        self.focus_best_score: float | None = None
        self.focus_stall_count = 0
        self.retention_repairs = 0
        self.last_repair_buckets: tuple[str, ...] = ()

    @classmethod
    def _validate_frontier_order(
        cls, frontier_order: tuple[str, ...] | None
    ) -> tuple[str, ...]:
        order = cls.frontier_order if frontier_order is None else tuple(frontier_order)
        if set(order) != set(cls.frontier_order) or len(order) != len(cls.frontier_order):
            raise ValueError("frontier order must contain each directional bucket exactly once")
        return order

    @classmethod
    def _target(cls, focus: str) -> dict[str, float]:
        if focus not in cls.frontier_order:
            raise ValueError(f"unsupported frontier bucket: {focus}")
        target = {name: cls.bucket_floor for name in BUCKETS}
        target["zero"] = cls.zero_floor
        target[focus] += cls.focus_extra
        return target

    def update(self, metrics: Mapping[str, float]) -> None:
        values = {name: float(metrics[name]) for name in BUCKETS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError("exposure scores must be finite and in [0, 1]")
        # Advance only after the first unmastered directional bucket.  This
        # makes a frontier switch deterministic and lets an earlier bucket
        # reclaim focus if a later stage exposes a regression.
        current = self.focus_bucket
        # Once bounded anti-stall is enabled, keep an unmastered focus long
        # enough to learn it. Otherwise the normal frontier scan would reclaim
        # focus for the first unmastered bucket immediately after a stall
        # rotation, giving the newly selected bucket only one window.
        if self.stall_windows and current in self.frontier_order and values[current] < self.focus_mastery:
            focus = current
            weakest = min(self.frontier_order, key=lambda name: values[name])
            # Partial skills can collapse before they ever qualify for the
            # gate's mastered-bucket retention repair. Do not keep decreasing
            # their exposure while waiting for an unrelated focus to stall.
            if values[current] - values[weakest] > self.focus_preemption_gap:
                focus = weakest
        else:
            focus = self.frontier_order[-1]
            for name in self.frontier_order:
                if values[name] < self.focus_mastery:
                    focus = name
                    break
        if self.stall_windows and current in self.frontier_order:
            current_score = values[current]
            if focus != current or current_score >= self.focus_mastery:
                self.focus_best_score = None
                self.focus_stall_count = 0
            elif self.focus_best_score is None or current_score >= self.focus_best_score + self.stall_improvement:
                self.focus_best_score = current_score
                self.focus_stall_count = 0
            else:
                self.focus_stall_count += 1
                if self.focus_stall_count >= self.stall_windows:
                    alternatives = [
                        name for name in self.frontier_order
                        if name != current and values[name] < self.focus_mastery
                    ]
                    if alternatives:
                        focus = min(
                            alternatives,
                            key=lambda name: (values[name], self.frontier_order.index(name)),
                        )
                    self.focus_best_score = values[focus]
                    self.focus_stall_count = 0
        self.focus_bucket = focus
        target = self._target(focus)
        for name in BUCKETS:
            self.probabilities[name] += self.update_rate * (target[name] - self.probabilities[name])
        self.windows += 1

    def repair(
        self,
        buckets: Sequence[str],
        metrics: Mapping[str, float],
        feedback: Mapping[str, object] | None = None,
    ) -> tuple[str, ...]:
        """Reallocate one bounded exposure slice after a retention failure.

        A rollback restores the last policy that mastered the protected
        buckets.  Replaying the same command mixture after that rollback is
        not adaptive: it repeatedly exposes the same failure.  This method
        keeps the zero and nominal anchors and distributes the focus slice
        across the buckets that regressed.  When several buckets regressed,
        command-conditioned reward mass breaks ties toward the bucket with the
        largest normalized tracking burden.
        """
        candidates = tuple(dict.fromkeys(str(name) for name in buckets))
        if not candidates:
            raise ValueError("retention repair requires at least one bucket")
        if any(name not in self.frontier_order for name in candidates):
            raise ValueError("retention repair accepts directional buckets only")
        values = {name: float(metrics[name]) for name in candidates}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("retention repair scores must be finite and in [0, 1]")

        # Score a failed bucket by capability deficit, then by the measured
        # command-aligned penalty mass.  The latter is deliberately bounded so
        # a noisy reward term cannot erase the product gate's direct evidence.
        burden: dict[str, float] = dict.fromkeys(candidates, 0.0)
        if feedback is not None:
            counts = feedback.get("sample_count")
            signed = feedback.get("weighted_reward_mass")
            if isinstance(counts, Mapping) and isinstance(signed, Mapping):
                for name in candidates:
                    count = float(counts.get(name, 0.0))
                    mass = signed.get(name)
                    if count <= 0.0 or not isinstance(mass, Mapping):
                        continue
                    penalty = 0.0
                    positive = 0.0
                    for term in ("linear_velocity_error_l1", "yaw_velocity_error_l1"):
                        value = mass.get(term)
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            penalty += max(0.0, -float(value))
                    for term in ("track_linear_velocity", "track_angular_velocity", "upright"):
                        value = mass.get(term)
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            positive += max(0.0, float(value))
                    # Mass is dt-integrated.  Rates make windows comparable;
                    # cap the ratio to keep this a tie-breaker, not a new gate.
                    burden[name] = min(2.0, (penalty / max(count * 0.02, 1e-6)) / max(positive / max(count * 0.02, 1e-6), 1e-3))

        ordered = tuple(
            sorted(
                candidates,
                key=lambda name: (-(1.0 - values[name]) * (1.0 + burden[name]), self.frontier_order.index(name)),
            )
        )
        target = {name: self.bucket_floor for name in BUCKETS}
        target["zero"] = self.zero_floor
        share = self.focus_extra / len(ordered)
        for name in ordered:
            target[name] += share
        for name in BUCKETS:
            self.probabilities[name] += self.update_rate * (target[name] - self.probabilities[name])
        self.focus_bucket = ordered[0]
        self.focus_best_score = values[self.focus_bucket]
        self.focus_stall_count = 0
        self.windows += 1
        self.retention_repairs += 1
        self.last_repair_buckets = ordered
        return ordered

    def consolidation_focus(self, bucket: str, metrics: Mapping[str, float]) -> str:
        """Move one bounded focus slice to a selected consolidation frontier."""
        if bucket not in self.frontier_order:
            raise ValueError("consolidation focus must be directional")
        value = float(metrics[bucket])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("consolidation focus score must be finite and in [0, 1]")
        target = self._target(bucket)
        for name in BUCKETS:
            self.probabilities[name] += self.update_rate * (target[name] - self.probabilities[name])
        self.focus_bucket = bucket
        self.focus_best_score = value
        self.focus_stall_count = 0
        self.windows += 1
        return bucket

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "probabilities": self.probabilities.copy(),
            "windows": self.windows,
            "focus_bucket": self.focus_bucket,
            "frontier_order": list(self.frontier_order),
            "stall_windows": self.stall_windows,
            "stall_improvement": self.stall_improvement,
            "focus_best_score": self.focus_best_score,
            "focus_stall_count": self.focus_stall_count,
            "retention_repairs": self.retention_repairs,
            "last_repair_buckets": list(self.last_repair_buckets),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("version") != self.version:
            raise ValueError("unsupported command exposure version")
        probabilities = payload.get("probabilities", {})
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(BUCKETS):
            raise ValueError("command exposure bucket mismatch")
        values = {name: float(probabilities[name]) for name in BUCKETS}
        if (any(not math.isfinite(value) or not self.bucket_floor <= value <= self.zero_floor + self.focus_extra
                for value in values.values())
                or not math.isclose(sum(values.values()), 1.0 - self.nominal_probability, abs_tol=1e-9)):
            raise ValueError("invalid command exposure probabilities")
        windows = payload.get("windows")
        if not isinstance(windows, int) or windows < 0:
            raise ValueError("invalid command exposure window count")
        focus = payload.get("focus_bucket")
        if focus not in self.frontier_order:
            raise ValueError("invalid frontier focus bucket")
        saved_order = payload.get("frontier_order")
        if saved_order is not None and tuple(saved_order) != self.frontier_order:
            raise ValueError("adaptive checkpoint frontier order mismatch")
        saved_stall_windows = payload.get("stall_windows", self.stall_windows)
        if not isinstance(saved_stall_windows, int) or saved_stall_windows < 0:
            raise ValueError("invalid stall window count")
        if saved_stall_windows != self.stall_windows:
            raise ValueError("adaptive checkpoint stall window mismatch")
        saved_improvement = float(payload.get("stall_improvement", self.stall_improvement))
        if not math.isfinite(saved_improvement) or saved_improvement < 0.0:
            raise ValueError("invalid stall improvement")
        if not math.isclose(saved_improvement, self.stall_improvement, abs_tol=1e-12):
            raise ValueError("adaptive checkpoint stall improvement mismatch")
        best_score = payload.get("focus_best_score")
        if best_score is not None and (not math.isfinite(float(best_score)) or not 0.0 <= float(best_score) <= 1.0):
            raise ValueError("invalid focus best score")
        stall_count = payload.get("focus_stall_count", 0)
        if not isinstance(stall_count, int) or stall_count < 0:
            raise ValueError("invalid focus stall count")
        self.probabilities = values
        self.windows = windows
        self.focus_bucket = str(focus)
        self.focus_best_score = None if best_score is None else float(best_score)
        self.focus_stall_count = stall_count
        repairs = payload.get("retention_repairs", 0)
        if not isinstance(repairs, int) or repairs < 0:
            raise ValueError("invalid retention repair count")
        repair_buckets = payload.get("last_repair_buckets", ())
        if not isinstance(repair_buckets, (list, tuple)) or any(
            str(name) not in self.frontier_order for name in repair_buckets
        ):
            raise ValueError("invalid retention repair buckets")
        self.retention_repairs = repairs
        self.last_repair_buckets = tuple(str(name) for name in repair_buckets)

    def apply(self, env: object) -> None:
        # CommandManager owns a deepcopy. The live term consumes these values
        # on its next scheduled resample; current episodes are not interrupted.
        term = env.command_manager.get_term("twist")
        term.cfg.bucket_probabilities = tuple(self.probabilities[name] for name in BUCKETS)


class TransitionExposure:
    """Bounded acquisition exposure for commands that need a moving start.

    The native diagnostic showed that a policy can turn after it is already
    walking but often does not initiate a yaw command from rest. This state
    machine adds a small, reversible fraction of forward-to-turn transitions;
    it never changes the six evaluator buckets or their command semantics.
    """

    version = 1
    yaw_buckets = ("yaw", "turn-left", "turn-right")
    mastery_threshold = 0.80
    release_threshold = 0.88
    increase_step = 0.05
    release_step = 0.025
    maximum_probability = 0.40

    def __init__(self, initial_probability: float = 0.0) -> None:
        value = float(initial_probability)
        if not math.isfinite(value) or not 0.0 <= value <= self.maximum_probability:
            raise ValueError("transition probability must be finite and in [0, 0.40]")
        self.probability = value
        self.windows = 0
        self.repairs = 0
        self.last_reason = "initial"

    def update(self, metrics: Mapping[str, float]) -> None:
        values = {name: float(metrics[name]) for name in self.yaw_buckets}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("transition exposure scores must be finite and in [0, 1]")
        frontier = min(values.values())
        if frontier < self.mastery_threshold:
            self.probability = min(self.maximum_probability, self.probability + self.increase_step)
            self.last_reason = "yaw_frontier_below_mastery"
        elif frontier >= self.release_threshold:
            self.probability = max(0.0, self.probability - self.release_step)
            self.last_reason = "yaw_frontier_consolidated"
        else:
            self.last_reason = "yaw_frontier_hold"
        self.windows += 1

    def repair(self, buckets: Sequence[str]) -> bool:
        """Increase transition coverage once after a yaw/turn retention failure."""
        affected = tuple(str(name) for name in buckets if str(name) in self.yaw_buckets)
        if not affected:
            return False
        before = self.probability
        self.probability = min(self.maximum_probability, self.probability + self.increase_step)
        self.repairs += 1
        self.last_reason = "retention_repair:" + ",".join(affected)
        return self.probability > before

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "probability": self.probability,
            "windows": self.windows,
            "repairs": self.repairs,
            "last_reason": self.last_reason,
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("version") != self.version:
            raise ValueError("unsupported transition exposure version")
        probability = float(payload.get("probability", -1.0))
        if not math.isfinite(probability) or not 0.0 <= probability <= self.maximum_probability:
            raise ValueError("invalid transition exposure probability")
        windows = payload.get("windows", 0)
        repairs = payload.get("repairs", 0)
        if not isinstance(windows, int) or windows < 0 or not isinstance(repairs, int) or repairs < 0:
            raise ValueError("invalid transition exposure counters")
        reason = payload.get("last_reason", "initial")
        if not isinstance(reason, str):
            raise ValueError("invalid transition exposure reason")
        self.probability = probability
        self.windows = windows
        self.repairs = repairs
        self.last_reason = reason

    def apply(self, env: object) -> None:
        term = env.command_manager.get_term("twist")
        if not hasattr(term.cfg, "transition_probability"):
            raise ValueError("transition exposure requires AdaptiveVelocityCommandCfg")
        term.cfg.transition_probability = self.probability


class AdaptiveActionRateRelief:
    """Temporarily release action smoothing when yaw acquisition stalls.

    The canonical Velocity curriculum reaches ``action_rate_l2=-1.0`` long
    before the adaptive gate can prove all directional capabilities.  That
    regularizer is useful for a settled gait, but it can tax the first large
    corrective action needed to acquire pure yaw from rest.  This controller
    gives the yaw frontier a bounded, checkpointed relief window and restores
    the canonical curriculum afterward.  It is driven only by frozen gate
    metrics; it never changes evaluator commands or acceptance thresholds.
    The optional pure-yaw scope leaves the other commands at canonical
    smoothing strength and only uses pure-yaw capability to open or close it.
    """

    version = 2
    yaw_buckets = ("yaw", "turn-left", "turn-right")

    def __init__(
        self,
        *,
        relief_weight: float = -0.2,
        trigger_threshold: float = 0.55,
        release_threshold: float = 0.80,
        active_windows: int = 4,
        cooldown_windows: int = 1,
        scope: str = "all",
    ) -> None:
        values = (relief_weight, trigger_threshold, release_threshold)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("action-rate relief parameters must be finite")
        if relief_weight > 0.0:
            raise ValueError("action-rate relief weight must be nonpositive")
        if not 0.0 <= trigger_threshold < release_threshold <= 1.0:
            raise ValueError("action-rate relief thresholds must be ordered in [0, 1]")
        if type(active_windows) is not int or active_windows < 1:
            raise ValueError("action-rate relief active windows must be positive")
        if type(cooldown_windows) is not int or cooldown_windows < 0:
            raise ValueError("action-rate relief cooldown windows must be nonnegative")
        if scope not in ("all", "pure_yaw"):
            raise ValueError("action-rate relief scope must be all or pure_yaw")
        self.scope = scope
        self.yaw_buckets = ("yaw",) if scope == "pure_yaw" else type(self).yaw_buckets
        self.relief_weight = float(relief_weight)
        self.trigger_threshold = float(trigger_threshold)
        self.release_threshold = float(release_threshold)
        self.active_windows = active_windows
        self.cooldown_windows = cooldown_windows
        self.reset()

    def reset(self) -> None:
        """Clear live state before loading a checkpoint without this controller."""
        self.active = False
        self.remaining_windows = 0
        self.cooldown_remaining = 0
        self.triggers = 0
        self.last_frontier: float | None = None
        self.last_reason = "initial"

    def _frontier(self, metrics: Mapping[str, float]) -> float:
        values = []
        for name in self.yaw_buckets:
            value = float(metrics[name])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("action-rate relief metrics must be finite and in [0, 1]")
            values.append(value)
        frontier = min(values)
        self.last_frontier = frontier
        return frontier

    def bootstrap(self, metrics: Mapping[str, float]) -> None:
        """Restore a useful relief window when resuming a legacy checkpoint."""
        self.reset()
        frontier = self._frontier(metrics)
        if frontier < self.trigger_threshold:
            self.active = True
            self.remaining_windows = self.active_windows
            self.cooldown_remaining = 0
            self.triggers += 1
            self.last_reason = "legacy_checkpoint_deficit"

    def update(self, metrics: Mapping[str, float], *, accepted: bool = True) -> None:
        """Consume one capability window and update the bounded relief state."""
        frontier = self._frontier(metrics)
        if self.active:
            if accepted and frontier >= self.release_threshold:
                self.active = False
                self.remaining_windows = 0
                self.last_reason = "yaw_frontier_released"
            else:
                self.remaining_windows -= 1
                if self.remaining_windows <= 0:
                    self.active = False
                    self.cooldown_remaining = self.cooldown_windows
                    self.last_reason = "bounded_relief_expired"
                else:
                    self.last_reason = "yaw_frontier_relief_hold"
            return
        if self.cooldown_remaining:
            self.cooldown_remaining -= 1
            self.last_reason = "relief_cooldown"
            return
        if frontier < self.trigger_threshold:
            self.active = True
            self.remaining_windows = self.active_windows
            self.triggers += 1
            self.last_reason = "yaw_frontier_deficit"
        else:
            self.last_reason = "yaw_frontier_above_trigger"

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "relief_weight": self.relief_weight,
            "scope": self.scope,
            "trigger_threshold": self.trigger_threshold,
            "release_threshold": self.release_threshold,
            "active_windows": self.active_windows,
            "cooldown_windows": self.cooldown_windows,
            "active": self.active,
            "remaining_windows": self.remaining_windows,
            "cooldown_remaining": self.cooldown_remaining,
            "triggers": self.triggers,
            "last_frontier": self.last_frontier,
            "last_reason": self.last_reason,
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("version") not in (1, self.version):
            raise ValueError("unsupported action-rate relief version")
        saved_scope = "all" if payload["version"] == 1 else payload.get("scope")
        if saved_scope != self.scope:
            raise ValueError("action-rate relief scope mismatch")
        for name, expected in (
            ("relief_weight", self.relief_weight),
            ("trigger_threshold", self.trigger_threshold),
            ("release_threshold", self.release_threshold),
        ):
            if not math.isclose(float(payload.get(name)), expected, abs_tol=1e-12):
                raise ValueError(f"action-rate relief {name} mismatch")
        for name, expected in (
            ("active_windows", self.active_windows),
            ("cooldown_windows", self.cooldown_windows),
        ):
            if int(payload.get(name, -1)) != expected:
                raise ValueError(f"action-rate relief {name} mismatch")
        active = payload.get("active")
        remaining = payload.get("remaining_windows")
        cooldown = payload.get("cooldown_remaining")
        triggers = payload.get("triggers")
        if (
            not isinstance(active, bool)
            or type(remaining) is not int
            or not 0 <= remaining <= self.active_windows
            or active != (remaining > 0)
        ):
            raise ValueError("invalid action-rate relief active state")
        if (
            type(cooldown) is not int
            or not 0 <= cooldown <= self.cooldown_windows
            or (active and cooldown > 0)
            or type(triggers) is not int
            or triggers < 0
        ):
            raise ValueError("invalid action-rate relief counters")
        frontier = payload.get("last_frontier")
        if frontier is not None and (not math.isfinite(float(frontier)) or not 0.0 <= float(frontier) <= 1.0):
            raise ValueError("invalid action-rate relief frontier")
        reason = payload.get("last_reason", "initial")
        if not isinstance(reason, str):
            raise ValueError("invalid action-rate relief reason")
        self.active = active
        self.remaining_windows = remaining
        self.cooldown_remaining = cooldown
        self.triggers = triggers
        self.last_frontier = None if frontier is None else float(frontier)
        self.last_reason = reason

    def apply(self, env: object) -> None:
        """Expose the live override to the canonical reward curriculum."""
        setattr(env, "_adaptive_action_rate_weight", self.relief_weight if self.active else None)
        setattr(env, "_adaptive_action_rate_scope", self.scope)

@dataclass(frozen=True)
class AxisConfig:
    """Stages and hysteresis settings for one difficulty axis."""

    name: str
    stages: tuple[object, ...]
    upper_threshold: float
    lower_threshold: float
    pass_windows: int = 2
    fail_windows: int = 2
    min_dwell_steps: int = 0

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("an adaptive axis needs at least one stage")
        if not 0.0 <= self.lower_threshold <= self.upper_threshold:
            raise ValueError("thresholds must satisfy 0 <= lower <= upper")
        if self.pass_windows < 1 or self.fail_windows < 1:
            raise ValueError("pass_windows and fail_windows must be positive")
        if self.min_dwell_steps < 0:
            raise ValueError("min_dwell_steps cannot be negative")


@dataclass
class AxisState:
    current_stage: int = 0
    last_transition_step: int = -1
    pass_count: int = 0
    fail_count: int = 0
    ema_score: float | None = None


@dataclass(frozen=True)
class Transition:
    step: int
    axis: str
    old_stage: int
    new_stage: int
    score: float
    threshold: float
    checkpoint: str | None
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "axis": self.axis,
            "old_stage": self.old_stage,
            "new_stage": self.new_stage,
            "score": self.score,
            "threshold": self.threshold,
            "checkpoint": self.checkpoint,
            "seed": self.seed,
        }


class GateOutcome(StrEnum):
    HOLD = "hold"
    ADVANCE = "advance"
    REGRESS = "regress"
    PRESERVATION_FAILURE = "preservation_failure"
    EVALUATION_ERROR = "evaluation_error"


@dataclass(frozen=True)
class GateDecision:
    outcome: GateOutcome
    transition: Transition | None = None
    reason: str | None = None


class CapabilityGate:
    """Advance or regress one axis from a conservative bucket aggregate.

    ``metrics`` must contain every name in ``critical_buckets``.  Advancement
    uses the lower-tail score and a preservation check against the best score
    seen at the current frontier.  The controller changes at most one axis and
    one stage per call, making its trace deterministic and easy to resume.
    """

    def __init__(
        self,
        axes: tuple[AxisConfig, ...],
        *,
        critical_buckets: tuple[str, ...],
        axis_mode: str | None = None,
        ema_alpha: float = 0.25,
        preservation_tolerance: float = 0.05,
    ) -> None:
        if not axes:
            raise ValueError("at least one adaptive axis is required")
        if not critical_buckets:
            raise ValueError("at least one critical bucket is required")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= preservation_tolerance < 1.0:
            raise ValueError("preservation_tolerance must be in [0, 1)")
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValueError("axis names must be unique")
        self.axes = {axis.name: axis for axis in axes}
        self.axis_order = names
        self.axis_mode = axis_mode
        self.critical_buckets = critical_buckets
        self.ema_alpha = ema_alpha
        self.preservation_tolerance = preservation_tolerance
        self.states = {axis.name: AxisState() for axis in axes}
        self.best_metrics: dict[str, float] = {}
        self.trace: list[Transition] = []

    def update(
        self,
        step: int,
        metrics: Mapping[str, float],
        *,
        checkpoint: str | None = None,
        seed: int = 0,
    ) -> Transition | None:
        decision = self.decide(step, metrics, checkpoint=checkpoint, seed=seed)
        return decision.transition

    def preservation_failures(self, metrics: Mapping[str, float]) -> tuple[str, ...]:
        """Return mastered buckets that violate the rollback tolerance."""
        threshold = max(axis.upper_threshold for axis in self.axes.values())
        failures = []
        for name in self.critical_buckets:
            best = self.best_metrics.get(name)
            value = float(metrics[name])
            if best is not None and best >= threshold and value < best * (1.0 - self.preservation_tolerance):
                failures.append(name)
        return tuple(failures)

    def decide(
        self,
        step: int,
        metrics: Mapping[str, float],
        *,
        checkpoint: str | None = None,
        seed: int = 0,
    ) -> GateDecision:
        missing = [name for name in self.critical_buckets if name not in metrics]
        if missing:
            raise KeyError(f"missing capability buckets: {', '.join(missing)}")
        values = {name: float(metrics[name]) for name in self.critical_buckets}
        if any(value != value or value == float("inf") or value == float("-inf") for value in values.values()):
            raise ValueError("capability metrics must be finite")
        score = min(values.values())
        previous_best = self.best_metrics.copy()
        if self.preservation_failures(values):
            # Regressions below the pass threshold must not evade preservation.
            return GateDecision(GateOutcome.PRESERVATION_FAILURE, reason="mastered bucket regressed")
        for name, value in values.items():
            self.best_metrics[name] = max(self.best_metrics.get(name, value), value)

        # Evaluate axes in stable order and transition only the first eligible one.
        for axis_name in self.axis_order:
            axis = self.axes[axis_name]
            state = self.states[axis_name]
            state.ema_score = score if state.ema_score is None else (
                self.ema_alpha * score + (1.0 - self.ema_alpha) * state.ema_score
            )
            if state.last_transition_step >= 0 and step - state.last_transition_step < axis.min_dwell_steps:
                continue
            if state.ema_score >= axis.upper_threshold:
                state.pass_count += 1
                state.fail_count = 0
            elif state.ema_score < axis.lower_threshold:
                state.fail_count += 1
                state.pass_count = 0
            else:
                state.pass_count = 0
                state.fail_count = 0

            preservation = all(
                name not in previous_best
                or value >= previous_best[name] * (1.0 - self.preservation_tolerance)
                for name, value in values.items()
            )
            if state.pass_count >= axis.pass_windows and not preservation:
                return GateDecision(GateOutcome.PRESERVATION_FAILURE, reason="frontier bucket regressed")
            if state.pass_count >= axis.pass_windows and state.current_stage < len(axis.stages) - 1:
                return GateDecision(GateOutcome.ADVANCE, self._transition(axis_name, state, step, score, axis.upper_threshold, checkpoint, seed, 1))
            if state.fail_count >= axis.fail_windows and state.current_stage > 0:
                return GateDecision(GateOutcome.REGRESS, self._transition(axis_name, state, step, score, axis.lower_threshold, checkpoint, seed, -1))
        return GateDecision(GateOutcome.HOLD)

    def _transition(self, axis_name: str, state: AxisState, step: int, score: float, threshold: float, checkpoint: str | None, seed: int, direction: int) -> Transition:
        old_stage = state.current_stage
        state.current_stage += direction
        state.last_transition_step = step
        state.pass_count = 0
        state.fail_count = 0
        transition = Transition(step, axis_name, old_stage, state.current_stage, score, threshold, checkpoint, seed)
        self.trace.append(transition)
        return transition

    def state_dict(self) -> dict[str, object]:
        return {
            "axis_mode": self.axis_mode,
            "enabled_axes": list(self.axis_order),
            "states": {name: vars(state).copy() for name, state in self.states.items()},
            "best_metrics": self.best_metrics.copy(),
            "trace": [item.as_dict() for item in self.trace],
        }

    def reset_evidence(self, *, step: int = -1) -> None:
        """Keep difficulty/history, discard scores from another distribution."""
        self.best_metrics = {}
        for state in self.states.values():
            state.pass_count = 0
            state.fail_count = 0
            state.ema_score = None
            state.last_transition_step = step

    def freeze_at_final(self, *, step: int) -> None:
        """Freeze every owned axis at its canonical final stage.

        Final-range fine-tuning is a separate, explicit post-acquisition mode.
        The gate remains serializable for audit, but its live stage counters are
        reset so no further capability window can move an axis during the
        fixed-range segment.
        """
        if type(step) is not int or step < 0:
            raise ValueError("final fine-tuning step must be a nonnegative integer")
        for name, axis in self.axes.items():
            state = self.states[name]
            state.current_stage = len(axis.stages) - 1
            state.last_transition_step = step
            state.pass_count = 0
            state.fail_count = 0
            state.ema_score = None

    def stage_value(self, axis_name: str) -> object:
        """Return the live stage value an MJLab adapter should apply."""
        if axis_name not in self.axes:
            raise KeyError(f"unknown adaptive axis: {axis_name}")
        state = self.states[axis_name]
        return self.axes[axis_name].stages[state.current_stage]

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if ("axis_mode" in payload and payload.get("axis_mode") != self.axis_mode) or (
            "enabled_axes" in payload and list(payload.get("enabled_axes", ())) != list(self.axis_order)
        ):
            raise ValueError("adaptive state axis mode/allowlist mismatch")
        states = payload.get("states")
        if not isinstance(states, Mapping):
            raise ValueError("adaptive state is missing states")
        for name, state_payload in states.items():
            if name not in self.states or not isinstance(state_payload, Mapping):
                raise ValueError(f"unknown adaptive axis state: {name}")
            self.states[name] = AxisState(**dict(state_payload))
        best_metrics = payload.get("best_metrics", {})
        if not isinstance(best_metrics, Mapping):
            raise ValueError("adaptive state has invalid best_metrics")
        self.best_metrics = {str(name): float(value) for name, value in best_metrics.items()}
        trace = payload.get("trace", [])
        if not isinstance(trace, list):
            raise ValueError("adaptive state has invalid trace")
        self.trace = [Transition(**dict(item)) for item in trace]


ADAPTIVE_AXIS_CONFIGS = (
    AxisConfig(
        "com_range",
        (0.003, 0.005, 0.010, 0.015),
        upper_threshold=0.80,
        lower_threshold=0.60,
        pass_windows=2,
        fail_windows=2,
        min_dwell_steps=10 * 24,
    ),
    AxisConfig(
        "head_com_range",
        (0.003, 0.005, 0.010),
        upper_threshold=0.80,
        lower_threshold=0.60,
        pass_windows=2,
        fail_windows=2,
        min_dwell_steps=10 * 24,
    ),
)


def evaluation_com_widths(
    distribution: str, axis_mode: str, stage_values: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Resolve frozen evaluator widths without changing the product distribution.

    Stage evaluation substitutes only controller-owned axes. Other axes and
    curricula retain the final reference used by the product battery.
    """
    if distribution not in ("initial", "final", "stage"):
        raise ValueError("invalid evaluation distribution")
    axes = resolve_enabled_axes(axis_mode)
    widths = {
        axis.name: float(axis.stages[0 if distribution == "initial" else -1])
        for axis in ADAPTIVE_AXIS_CONFIGS
    }
    if distribution == "stage":
        if not axes or not isinstance(stage_values, Mapping) or set(stage_values) != set(axes):
            raise ValueError("stage evaluation requires exactly the enabled axis values")
        configs = {axis.name: axis for axis in ADAPTIVE_AXIS_CONFIGS}
        for name, value in stage_values.items():
            if isinstance(value, bool) or value not in configs[name].stages:
                raise ValueError(f"unrecognized evaluation stage value: {name}")
            widths[name] = float(value)
    elif stage_values:
        raise ValueError("stage values require stage evaluation")
    return widths


def apply_stage_to_env(env: object, axis_name: str, stage_value: object) -> None:
    """Apply a capability transition to the live MJLab event manager.

    Managers deepcopy configuration during construction, so this intentionally
    resolves the term through ``get_term_cfg``.  The helper accepts ``object``
    to keep the controller importable and unit-testable without MJLab.
    """

    event_name = {
        "com_range": "randomize_com",
        "head_com_range": "randomize_head_com",
    }.get(axis_name)
    if event_name is None:
        raise KeyError(f"no MJLab event binding for adaptive axis: {axis_name}")
    event_manager = getattr(env, "event_manager", None)
    if event_manager is None:
        raise AttributeError("adaptive environment has no event_manager")
    event_cfg = event_manager.get_term_cfg(event_name)
    event_cfg.params["ranges"] = (-float(stage_value), float(stage_value))
