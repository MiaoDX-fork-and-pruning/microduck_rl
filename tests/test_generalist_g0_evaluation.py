import numpy as np
import pytest

from mjlab_microduck.generalist_g0_evaluation import TraceMetrics, make_report
from mjlab_microduck.generalist_transition_graph import LEGAL_EDGES


def test_trace_metrics_records_required_physics_signals():
    trace = TraceMetrics()
    trace.append(height=0.12, tilt=0.1, position=[0, 0, 0.12], action=np.zeros(14))
    trace.append(height=0.11, tilt=0.2, position=[0.3, 0.4, 0.11], action=np.ones(14))
    report = trace.report()
    assert report["finite"]
    assert report["height_m"] == {"min": 0.11, "max": 0.12, "final": 0.11}
    assert report["tilt_rad"]["max"] == 0.2
    assert report["displacement_m"] == pytest.approx(0.5)
    assert report["max_abs_action"] == 1.0
    assert report["success"] is True
    assert report["peak_action_jump"] == 1.0


def test_trace_metrics_rejects_action_contract_violation():
    trace = TraceMetrics()
    trace.append(height=0.12, tilt=0.1, position=[0, 0, 0.12], action=np.full(14, 1.01))
    report = trace.report()
    assert report["finite"] is True
    assert report["max_abs_action"] == pytest.approx(1.01)
    assert report["success"] is False


def test_nonfinite_trace_is_sticky():
    trace = TraceMetrics()
    trace.append(height=np.nan, tilt=0, position=[0, 0, 0], action=np.zeros(14))
    trace.append(height=1, tilt=0, position=[0, 0, 1], action=np.zeros(14))
    assert trace.report()["finite"] is False


def _entry(**values):
    return {**values, "metrics": {"finite": True}}


def test_report_requires_three_behaviors_four_edges_and_records_unsupported():
    behaviors = [_entry(behavior=name) for name in ("stand", "locomotion", "sit_stand")]
    edges = [_entry(**{"from": source, "to": target}) for source, target in LEGAL_EDGES]
    report = make_report(backend="onnx", seed=42, behaviors=behaviors, edges=edges)
    assert report["finite"]
    assert all("success" in item for item in report["behaviors"] + report["legal_edges"])
    assert len(report["legal_edges"]) == 4
    assert {(item["from"], item["to"]) for item in report["unsupported_edges"]} == {
        ("VELOCITY", "SITSTAND"), ("SITSTAND", "VELOCITY")
    }
    assert all(not item["supported"] and not item["exercised"] for item in report["unsupported_edges"])


def test_report_rejects_incomplete_battery():
    with pytest.raises(ValueError, match="behavior battery"):
        make_report(backend="pytorch", seed=1, behaviors=[], edges=[])
