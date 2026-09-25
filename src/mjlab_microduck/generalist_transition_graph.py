"""Frozen transition graph for the bounded generalist G0 experiment.

The graph is deliberately training-only: it describes which condition changes
may be sampled without silently inventing support for unvalidated edges.
"""

from __future__ import annotations

from typing import Mapping

STATES = ("VELSTAND", "VELOCITY", "SITSTAND")
LEGAL_EDGES = frozenset(
    {
        ("VELSTAND", "VELOCITY"),
        ("VELOCITY", "VELSTAND"),
        ("VELSTAND", "SITSTAND"),
        ("SITSTAND", "VELSTAND"),
    }
)
UNSUPPORTED_EDGES = {
    ("VELOCITY", "SITSTAND"): "route through VELSTAND; direct edge is unproven",
    ("SITSTAND", "VELOCITY"): "route through VELSTAND; direct edge is unproven",
}

GRAPH: Mapping[str, object] = {
    "schema": "generalist-g0-transition-graph",
    "version": 1,
    "states": STATES,
    "legal_edges": tuple({"from": a, "to": b} for a, b in sorted(LEGAL_EDGES)),
    "unsupported_edges": tuple(
        {"from": a, "to": b, "reason": reason}
        for (a, b), reason in sorted(UNSUPPORTED_EDGES.items())
    ),
}


def is_legal_transition(source: str, destination: str) -> bool:
    """Return whether a condition switch is an explicitly proven G0 edge."""
    return (source, destination) in LEGAL_EDGES


def validate_transition(source: str, destination: str) -> None:
    """Raise a useful error for unknown states or unsupported edges."""
    if source not in STATES or destination not in STATES:
        raise ValueError(f"unknown G0 transition state: {source!r} -> {destination!r}")
    if not is_legal_transition(source, destination):
        reason = UNSUPPORTED_EDGES.get((source, destination), "edge is not in the frozen graph")
        raise ValueError(f"unsupported G0 transition {source} -> {destination}: {reason}")


def route_transition(source: str, destination: str) -> tuple[str, ...]:
    """Return the graph path, routing cross-switches through VELSTAND."""
    if source == destination:
        if source not in STATES:
            raise ValueError(f"unknown G0 transition state: {source!r}")
        return (source,)
    if is_legal_transition(source, destination):
        return (source, destination)
    if (source, destination) in UNSUPPORTED_EDGES:
        return (source, "VELSTAND", destination)
    validate_transition(source, destination)
    raise AssertionError("unreachable")
