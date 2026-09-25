import json
from pathlib import Path

import pytest

from mjlab_microduck.generalist_transition_graph import (
    GRAPH,
    LEGAL_EDGES,
    route_transition,
    validate_transition,
)


def test_machine_readable_graph_matches_frozen_contract():
    artifact = json.loads((Path(__file__).parents[1] / "docs/generalist_g0_transition_graph.json").read_text())
    assert artifact["schema"] == GRAPH["schema"]
    assert {tuple((e["from"], e["to"])) for e in artifact["legal_edges"]} == LEGAL_EDGES
    assert len(artifact["unsupported_edges"]) == 2


@pytest.mark.parametrize("source,destination", sorted(LEGAL_EDGES))
def test_legal_edges_validate(source, destination):
    validate_transition(source, destination)
    assert route_transition(source, destination) == (source, destination)


@pytest.mark.parametrize("source,destination", [("VELOCITY", "SITSTAND"), ("SITSTAND", "VELOCITY")])
def test_unproven_direct_edges_route_through_velstand(source, destination):
    with pytest.raises(ValueError, match="unsupported G0 transition"):
        validate_transition(source, destination)
    assert route_transition(source, destination) == (source, "VELSTAND", destination)


def test_unknown_state_is_rejected():
    with pytest.raises(ValueError, match="unknown G0 transition state"):
        validate_transition("VELOCITY", "KICK")
