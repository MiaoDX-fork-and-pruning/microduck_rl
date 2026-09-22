"""Dependency-free continuous capability report v2."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping

BUCKETS = ("zero", "forward", "lateral", "yaw", "turn-left", "turn-right")
AXIS_MODES = {
    "all_static": (),
    "com": ("com_range",),
    "head_com": ("head_com_range",),
    "composed": ("com_range", "head_com_range"),
}
REQUIRED_METADATA = (
    "task_id",
    "source_sha",
    "evaluator_config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "policy_format",
    "seed_set_id",
    "generated_at",
)
DEFAULT_THRESHOLDS = {
    "tracking_m_s": 0.12,
    "angular_tracking_rad_s": 0.6,
    "zero_drift_m": 0.06,
    "tilt_p95_rad": math.radians(35),
}

# The training-side acquisition term averages signed error over a 0.5 s gait
# cycle before taking its magnitude.  The native gate must measure the same
# command-following quantity; otherwise a healthy alternating gait is scored
# as if it were standing still.  Instantaneous error remains a hard stability
# cap so a violent oscillation cannot pass by averaging alone.
TRACKING_METRIC_SAMPLEWISE = "samplewise_mae_v1"
TRACKING_METRIC_SIGNED_EMA = "signed_ema_v1"
DEFAULT_TRACKING_METRIC = TRACKING_METRIC_SIGNED_EMA
DEFAULT_TRACKING_TAU_S = 0.5
DEFAULT_INSTANTANEOUS_CAPS = {
    "tracking_m_s": 0.25,
    "angular_tracking_rad_s": 0.60,
}


def resolve_enabled_axes(axis_mode: str) -> tuple[str, ...]:
    if axis_mode not in AXIS_MODES:
        raise ValueError(f"unsupported adaptive axis mode: {axis_mode}")
    return AXIS_MODES[axis_mode]


def _finite(v: Any) -> bool:
    return isinstance(v, Real) and math.isfinite(float(v))


def _clean(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, Real):
        value = float(v)
        return value if math.isfinite(value) else None
    return v


def canonical_sha256(v: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _clean(v), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _error(v: Any, threshold: float) -> float:
    return _clamp(1.0 - abs(float(v)) / threshold) if _finite(v) else 0.0


def tracking_error_metrics(
    actual: Any,
    commanded: Any,
    *,
    dt: float = 0.02,
    tau_s: float = DEFAULT_TRACKING_TAU_S,
) -> tuple[float, float]:
    """Return samplewise MAE and signed-EMA MAE for one command axis.

    The EMA is initialized from the first observed error and then updated with
    the same continuous-time coefficient used by the adaptive reward.  A
    constant command therefore reports DC tracking bias, while periodic gait
    ripple largely cancels before the absolute value is taken.
    """
    actual_values = list(actual)
    command_values = list(commanded)
    if len(actual_values) == 0 or len(actual_values) != len(command_values):
        return math.nan, math.nan
    if not _finite(dt) or float(dt) <= 0.0 or not _finite(tau_s) or float(tau_s) <= 0.0:
        return math.nan, math.nan
    errors = [float(a) - float(c) for a, c in zip(actual_values, command_values)]
    if not all(math.isfinite(value) for value in errors):
        return math.nan, math.nan
    alpha = 1.0 - math.exp(-float(dt) / float(tau_s))
    ema = errors[0]
    ema_abs_sum = abs(ema)
    for error in errors[1:]:
        ema += alpha * (error - ema)
        ema_abs_sum += abs(ema)
    return (
        sum(abs(error) for error in errors) / len(errors),
        ema_abs_sum / len(errors),
    )


def _tracking_metric_mode(evaluator_config: Mapping[str, Any]) -> str:
    mode = evaluator_config.get("tracking_metric", TRACKING_METRIC_SAMPLEWISE)
    if mode not in (TRACKING_METRIC_SAMPLEWISE, TRACKING_METRIC_SIGNED_EMA):
        raise ValueError(f"unsupported tracking metric: {mode}")
    return str(mode)


def _thresholds(evaluator_config: Mapping[str, Any]) -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if _tracking_metric_mode(evaluator_config) == TRACKING_METRIC_SIGNED_EMA:
        thresholds.update(DEFAULT_INSTANTANEOUS_CAPS)
    configured = evaluator_config.get("thresholds")
    if isinstance(configured, Mapping):
        for key, value in configured.items():
            if key in thresholds and _finite(value) and float(value) > 0:
                thresholds[key] = float(value)
    return thresholds


def _metadata(m: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_METADATA if not m.get(k)]
    if missing:
        raise ValueError("missing capability metadata: " + ", ".join(missing))
    return _clean(dict(m))


def _bucket(
    raw: Mapping[str, Any],
    name: str,
    t: Mapping[str, float],
    *,
    tracking_metric: str = TRACKING_METRIC_SAMPLEWISE,
) -> dict[str, Any]:
    tracking_key = (
        "angular_tracking_error_rad_s"
        if name in ("yaw", "turn-left", "turn-right")
        else "tracking_error_m_s"
    )
    instantaneous_key = (
        "angular_tracking_error_samplewise_rad_s"
        if name in ("yaw", "turn-left", "turn-right")
        else "tracking_error_samplewise_m_s"
    )
    required = ["survival_fraction", "tilt_p95_rad"] + (
        ["zero_drift_m"]
        if name == "zero"
        else ["angular_tracking_error_rad_s"]
        if name in ("yaw", "turn-left", "turn-right")
        else ["tracking_error_m_s"]
    )
    if name != "zero" and tracking_metric == TRACKING_METRIC_SIGNED_EMA:
        required.append(instantaneous_key)
    valid = all(_finite(raw.get(k)) for k in required)
    c = {
        "survival": _clamp(raw.get("survival_fraction", 0))
        if _finite(raw.get("survival_fraction"))
        else 0.0,
        "upright": _error(raw.get("tilt_p95_rad"), t["tilt_p95_rad"]),
    }
    key = "zero_drift_m" if name == "zero" else tracking_key
    c["drift" if name == "zero" else "tracking"] = _error(
        raw.get(key),
        t["zero_drift_m"]
        if name == "zero"
        else t["angular_tracking_rad_s"]
        if name in ("yaw", "turn-left", "turn-right")
        else t["tracking_m_s"],
    )
    if name != "zero" and tracking_metric == TRACKING_METRIC_SIGNED_EMA:
        cap_key = (
            "angular_tracking_rad_s"
            if name in ("yaw", "turn-left", "turn-right")
            else "tracking_m_s"
        )
        # This is deliberately a hard stability cap, rather than another
        # smooth score.  It preserves the product threshold for DC tracking
        # while rejecting visibly violent samplewise excursions.
        c["instantaneous_stability"] = float(
            _finite(raw.get(instantaneous_key))
            and abs(float(raw[instantaneous_key])) <= t[cap_key]
        )
    if not valid:
        c = {k: 0.0 for k in c}
    score = _clamp(min(c.values()))
    return {
        "raw": _clean(dict(raw)),
        "components": c,
        "score": score,
        "passed": bool(valid and score >= 0.8),
        "valid": valid,
    }


@dataclass(frozen=True)
class CapabilityReport:
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityReport":
        if not isinstance(payload, Mapping):
            raise ValueError("capability report must be a mapping")
        if payload.get("schema_version") != 2:
            raise ValueError("capability report schema_version must be 2")
        for k in ("metadata", "axis_mode", "enabled_axes", "buckets", "aggregate"):
            if k not in payload:
                raise ValueError(f"missing capability report field: {k}")
        _metadata(payload["metadata"])
        if tuple(payload["enabled_axes"]) != resolve_enabled_axes(payload["axis_mode"]):
            raise ValueError("enabled_axes does not match axis_mode")
        if tuple(payload["buckets"]) != BUCKETS:
            raise ValueError("report must contain canonical buckets")
        evaluator_config = payload.get("evaluator_config", {})
        if not isinstance(evaluator_config, Mapping):
            raise ValueError("invalid evaluator config")
        tracking_metric = _tracking_metric_mode(evaluator_config)
        thresholds = _thresholds(evaluator_config)
        expected_buckets: dict[str, dict[str, Any]] = {}
        for n in BUCKETS:
            b = payload["buckets"][n]
            if not isinstance(b, Mapping) or not isinstance(b.get("raw"), Mapping):
                raise ValueError(f"missing raw bucket evidence: {n}")
            expected = _bucket(
                b["raw"], n, thresholds, tracking_metric=tracking_metric
            )
            # Raw evidence is the source of truth.  Reject forged aggregates,
            # components, validity, or pass flags rather than silently trusting them.
            for key in ("components", "score", "passed", "valid"):
                if key not in b or canonical_sha256(b[key]) != canonical_sha256(expected[key]):
                    raise ValueError(f"bucket {n} {key} does not match raw evidence")
            expected_buckets[n] = expected
        aggregate = payload["aggregate"]
        if not isinstance(aggregate, Mapping):
            raise ValueError("invalid capability aggregate")
        expected_aggregate = {
            "lower_tail_score": min(x["score"] for x in expected_buckets.values()),
            "critical_buckets": list(BUCKETS),
            "passed": bool(all(x["passed"] for x in expected_buckets.values())),
            "valid": bool(all(x["valid"] for x in expected_buckets.values())),
        }
        if canonical_sha256(dict(aggregate)) != canonical_sha256(expected_aggregate):
            raise ValueError("aggregate does not match bucket evidence")
        return cls(_clean(dict(payload)))

    def canonical_json(self) -> str:
        return json.dumps(
            _clean(self.payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def metrics(self) -> dict[str, float]:
        return {n: float(self.payload["buckets"][n]["score"]) for n in BUCKETS}

    def to_gate_metrics(self) -> dict[str, float]:
        if not self.payload["aggregate"].get("valid", False):
            raise ValueError("invalid capability report cannot feed the gate")
        return self.metrics


def build_capability_report(
    bucket_raw: Mapping[str, Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
    axis_mode: str = "composed",
    evaluator_config: Mapping[str, Any] | None = None,
) -> CapabilityReport:
    if set(bucket_raw) != set(BUCKETS):
        raise ValueError("capability report requires exactly six buckets")
    resolve_enabled_axes(axis_mode)
    cfg = dict(evaluator_config or {})
    tracking_metric = _tracking_metric_mode(cfg)
    t = _thresholds(cfg)
    cfg.setdefault("tracking_metric", tracking_metric)
    if tracking_metric == TRACKING_METRIC_SIGNED_EMA:
        cfg.setdefault("tracking_metric_tau_s", DEFAULT_TRACKING_TAU_S)
        cfg.setdefault("instantaneous_caps", dict(DEFAULT_INSTANTANEOUS_CAPS))
    bs = {
        n: _bucket(bucket_raw[n], n, t, tracking_metric=tracking_metric)
        for n in BUCKETS
    }
    valid = all(x["valid"] for x in bs.values())
    cfg.setdefault("thresholds", t)
    payload = {
        "schema_version": 2,
        "metadata": _metadata(metadata),
        "axis_mode": axis_mode,
        "enabled_axes": list(resolve_enabled_axes(axis_mode)),
        "evaluator_config": _clean(cfg),
        "buckets": bs,
        "aggregate": {
            "lower_tail_score": min(x["score"] for x in bs.values()),
            "critical_buckets": list(BUCKETS),
            "passed": bool(valid and all(x["passed"] for x in bs.values())),
            "valid": valid,
        },
    }
    return CapabilityReport.from_dict(payload)


def raw_metrics_from_trace(
    trace: Mapping[str, Any],
    *,
    bucket: str,
    commanded: Any = None,
    control_hz: float = 50.0,
) -> dict[str, Any]:
    vel = trace.get("trunk_linear_velocity_m_s", [])
    ang = trace.get("trunk_angular_velocity_rad_s", [])
    tilt = trace.get("trunk_tilt_rad", [])
    key = (
        "zero_drift_m"
        if bucket == "zero"
        else "angular_tracking_error_rad_s"
        if bucket in ("yaw", "turn-left", "turn-right")
        else "tracking_error_m_s"
    )
    try:
        lengths = {len(x) for x in (vel, ang, tilt)}
    except TypeError:
        lengths = set()
    if not lengths or lengths != {next(iter(lengths))} or next(iter(lengths)) == 0:
        return {"survival_fraction": 0.0, "tilt_p95_rad": None, key: None}
    if any(not isinstance(row, (list, tuple)) or len(row) < 3 or
           any(not _finite(y) for y in row[:3]) for row in list(vel) + list(ang)) or any(
        not _finite(x) for x in tilt
    ):
        return {"survival_fraction": 0.0, "tilt_p95_rad": None, key: None}
    raw = {
        "survival_fraction": 1.0,
        "episode_length_mean": len(tilt) / control_hz,
        "tilt_p95_rad": float(sorted(tilt)[max(0, int(0.95 * len(tilt)) - 1)]),
    }
    if bucket == "zero":
        pos = trace.get("trunk_position_m", trace.get("trunk_position", []))
        raw[key] = (
            float(math.hypot(pos[-1][0] - pos[0][0], pos[-1][1] - pos[0][1]))
            if len(pos) == len(tilt) and len(pos) >= 2 and all(
                isinstance(row, (list, tuple)) and len(row) >= 2 and
                all(_finite(v) for v in row[:2]) for row in pos
            )
            else None
        )
    elif commanded is None:
        raw[key] = None
    else:
        idx = 2 if key.startswith("angular") else 1 if bucket == "lateral" else 0
        observed = ang if idx == 2 else vel
        if len(commanded) != len(observed) or any(
            not isinstance(row, (list, tuple)) or len(row) <= idx or not _finite(row[idx])
            for row in commanded
        ):
            raw[key] = None
        else:
            raw[key] = sum(
                abs(float(c[idx]) - float(o[idx])) for c, o in zip(commanded, observed)
            ) / len(observed)
    return raw
