"""Capability-gated curriculum state used by the MJLab adaptive experiment.

The gate is deliberately independent of MJLab and torch.  An evaluator feeds it
fixed-seed, bucket-level metrics at evaluation windows; the training adapter can
then apply the returned one-stage transition to live manager term configs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
from enum import StrEnum

from mjlab_microduck.evaluation.capability import BUCKETS


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
    directional buckets in ``frontier_order``.
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
    frontier_order = ("forward", "lateral", "yaw", "turn-left", "turn-right")

    def __init__(self) -> None:
        self.focus_bucket = self.frontier_order[0]
        self.probabilities = self._target(self.focus_bucket)
        self.windows = 0

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
        focus = self.frontier_order[-1]
        for name in self.frontier_order:
            if values[name] < self.focus_mastery:
                focus = name
                break
        self.focus_bucket = focus
        target = self._target(focus)
        for name in BUCKETS:
            self.probabilities[name] += self.update_rate * (target[name] - self.probabilities[name])
        self.windows += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "probabilities": self.probabilities.copy(),
            "windows": self.windows,
            "focus_bucket": self.focus_bucket,
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
        self.probabilities = values
        self.windows = windows
        self.focus_bucket = str(focus)

    def apply(self, env: object) -> None:
        # CommandManager owns a deepcopy. The live term consumes these values
        # on its next scheduled resample; current episodes are not interrupted.
        term = env.command_manager.get_term("twist")
        term.cfg.bucket_probabilities = tuple(self.probabilities[name] for name in BUCKETS)

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
        mastered_threshold = max(axis.upper_threshold for axis in self.axes.values())
        if any(
            previous_best.get(name, 0) >= mastered_threshold
            and value < previous_best[name] * (1.0 - self.preservation_tolerance)
            for name, value in values.items()
        ):
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
