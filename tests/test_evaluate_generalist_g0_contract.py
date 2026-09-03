import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path(__file__).parents[1] / "scripts/evaluate_generalist_g0.py"
    spec = importlib.util.spec_from_file_location("evaluate_generalist_g0", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_segments_match_track_a_dwell_and_handoff():
    module = _module()
    assert module.CONTROL_HZ == 50
    assert [(s.state, s.command_x, s.ticks) for s in module.BEHAVIOR_SEGMENTS["VELOCITY"]] == [
        ("VELOCITY", 0.15, 700)
    ]
    sit_to_stand = module.EDGE_SEGMENTS[("SITSTAND", "VELSTAND")]
    assert [(s.state, s.command_x, s.ticks) for s in sit_to_stand] == [
        ("SITSTAND", 1.0, 300),
        ("SITSTAND", 0.0, 300),
        ("VELSTAND", 0.0, 400),
    ]
    assert sit_to_stand[0].score is False
    assert sit_to_stand[1].active_transition is True


def test_edge_preparation_is_not_scored():
    module = _module()
    for segments in module.EDGE_SEGMENTS.values():
        assert segments[0].score is False
        assert any(segment.score for segment in segments[1:])
