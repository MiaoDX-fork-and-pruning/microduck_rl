"""Canonical report contract and metrics for the generalist G0 battery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from mjlab_microduck.generalist_transition_graph import LEGAL_EDGES, UNSUPPORTED_EDGES

STATE_TO_BEHAVIOR = {
    "VELSTAND": "stand",
    "VELOCITY": "locomotion",
    "SITSTAND": "sit_stand",
}


@dataclass
class TraceMetrics:
    """Accumulate the common physics signals without depending on a simulator."""

    heights: list[float] = field(default_factory=list)
    tilts: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    finite: bool = True

    def append(self, *, height: float, tilt: float, position: Iterable[float], action: Iterable[float]) -> None:
        position_array = np.asarray(position, dtype=np.float64)
        action_array = np.asarray(action, dtype=np.float64)
        values_finite = bool(
            np.isfinite(height)
            and np.isfinite(tilt)
            and np.isfinite(position_array).all()
            and np.isfinite(action_array).all()
        )
        self.finite &= values_finite
        self.heights.append(float(height))
        self.tilts.append(float(tilt))
        self.positions.append(position_array)
        self.actions.append(action_array)

    def report(self) -> dict:
        displacement = (
            self.positions[-1] - self.positions[0]
            if len(self.positions) >= 2
            else np.zeros(3, dtype=np.float64)
        )
        actions = np.asarray(self.actions)
        jumps = np.abs(np.diff(actions, axis=0)) if len(actions) >= 2 else np.zeros((0, 14))
        passed = bool(self.finite and self.actions and max(self.tilts, default=np.inf) < np.deg2rad(65.0)
                      and np.max(np.abs(actions)) <= 1.0 + 1e-6)
        return {
            "steps": len(self.actions),
            "finite": self.finite,
            "height_m": _range_summary(self.heights),
            "tilt_rad": _range_summary(self.tilts),
            "world_displacement_m": displacement.tolist(),
            "displacement_m": float(np.linalg.norm(displacement[:2])),
            "max_abs_action": float(np.max(np.abs(actions))) if actions.size else None,
            "peak_action_jump": float(np.max(jumps)) if jumps.size else 0.0,
            "success": passed,
            "passed": passed,
        }


def _range_summary(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "final": None}
    return {"min": min(values), "max": max(values), "final": values[-1]}


def make_report(*, backend: str, seed: int, behaviors: list[dict], edges: list[dict]) -> dict:
    """Build and validate the complete G0 evaluation report surface."""
    behavior_names = {item["behavior"] for item in behaviors}
    if behavior_names != set(STATE_TO_BEHAVIOR.values()):
        raise ValueError(f"behavior battery must cover exactly {sorted(STATE_TO_BEHAVIOR.values())}")
    reported_edges = {(item["from"], item["to"]) for item in edges}
    if reported_edges != LEGAL_EDGES:
        raise ValueError("edge battery must cover exactly the four legal G0 edges")
    unsupported = [
        {"from": source, "to": destination, "supported": False, "exercised": False, "reason": reason}
        for (source, destination), reason in sorted(UNSUPPORTED_EDGES.items())
    ]
    finite = all(item["metrics"]["finite"] for item in behaviors + edges)
    for item in behaviors:
        item.setdefault("success", bool(item["metrics"].get("success", False)))
        item.setdefault("passed", item["success"])
    for item in edges:
        item.setdefault("success", bool(item["metrics"].get("success", False)) and item.get("reset_count", 0) == 0)
        item.setdefault("passed", item["success"])
    return {
        "schema": "generalist-g0-evaluation",
        "schema_version": 1,
        "backend": backend,
        "seed": seed,
        "finite": finite,
        "behaviors": behaviors,
        "legal_edges": edges,
        "unsupported_edges": unsupported,
    }
