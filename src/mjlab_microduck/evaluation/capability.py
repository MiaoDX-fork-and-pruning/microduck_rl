"""Dependency-free continuous capability report v2."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
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


def resolve_enabled_axes(axis_mode: str) -> tuple[str, ...]:
    if axis_mode not in AXIS_MODES:
        raise ValueError(f"unsupported adaptive axis mode: {axis_mode}")
    return AXIS_MODES[axis_mode]


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _clean(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return None if isinstance(v, float) and not math.isfinite(v) else v


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


def _metadata(m: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_METADATA if not m.get(k)]
    if missing:
        raise ValueError("missing capability metadata: " + ", ".join(missing))
    return _clean(dict(m))


def _bucket(
    raw: Mapping[str, Any], name: str, t: Mapping[str, float]
) -> dict[str, Any]:
    required = ["survival_fraction", "tilt_p95_rad"] + (
        ["zero_drift_m"]
        if name == "zero"
        else ["angular_tracking_error_rad_s"]
        if name in ("yaw", "turn-left", "turn-right")
        else ["tracking_error_m_s"]
    )
    valid = all(_finite(raw.get(k)) for k in required)
    c = {
        "survival": _clamp(raw.get("survival_fraction", 0))
        if _finite(raw.get("survival_fraction"))
        else 0.0,
        "upright": _error(raw.get("tilt_p95_rad"), t["tilt_p95_rad"]),
    }
    key = (
        "zero_drift_m"
        if name == "zero"
        else "angular_tracking_error_rad_s"
        if name in ("yaw", "turn-left", "turn-right")
        else "tracking_error_m_s"
    )
    c["drift" if name == "zero" else "tracking"] = _error(
        raw.get(key),
        t["zero_drift_m"]
        if name == "zero"
        else t["angular_tracking_rad_s"]
        if name in ("yaw", "turn-left", "turn-right")
        else t["tracking_m_s"],
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
        for n in BUCKETS:
            b = payload["buckets"][n]
            if (
                not isinstance(b, Mapping)
                or not _finite(b.get("score"))
                or not 0 <= b["score"] <= 1
            ):
                raise ValueError(f"invalid bucket score: {n}")
            if b.get("valid") and any(
                not _finite(v) for v in b.get("components", {}).values()
            ):
                raise ValueError(f"non-finite components: {n}")
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
    t = dict(DEFAULT_THRESHOLDS)
    if evaluator_config and isinstance(evaluator_config.get("thresholds"), Mapping):
        t.update({k: float(v) for k, v in evaluator_config["thresholds"].items()})
    bs = {n: _bucket(bucket_raw[n], n, t) for n in BUCKETS}
    valid = all(x["valid"] for x in bs.values())
    cfg = dict(evaluator_config or {})
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
    if len(vel) == 0 or len(tilt) == 0:
        return {"survival_fraction": 0.0, "tilt_p95_rad": None, key: None}
    if any(not _finite(y) for row in list(vel) + list(ang) for y in row) or any(
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
            if len(pos) >= 2
            else None
        )
    elif commanded is None:
        raw[key] = None
    else:
        idx = 2 if key.startswith("angular") else 1 if bucket == "lateral" else 0
        observed = ang if idx == 2 else vel
        raw[key] = sum(
            abs(float(c[idx]) - float(o[idx])) for c, o in zip(commanded, observed)
        ) / max(1, len(observed))
    return raw
