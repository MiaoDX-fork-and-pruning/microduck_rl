import json

from mjlab_microduck.tasks.adaptive_curriculum import ADAPTIVE_AXIS_CONFIGS, CapabilityGate


def _metrics(value=1.0):
    return {bucket: value for bucket in ("zero", "forward", "lateral", "yaw", "turn-left", "turn-right")}


def test_com_state_contains_only_com_axis():
    axes = tuple(axis for axis in ADAPTIVE_AXIS_CONFIGS if axis.name == "com_range")
    gate = CapabilityGate(axes, critical_buckets=tuple(_metrics()), axis_mode="com")
    payload = gate.state_dict()
    assert payload["axis_mode"] == "com"
    assert payload["enabled_axes"] == ["com_range"]
    assert "head_com_range" not in json.dumps(payload)


def test_axis_mode_mismatch_is_rejected():
    axes = tuple(axis for axis in ADAPTIVE_AXIS_CONFIGS if axis.name == "com_range")
    gate = CapabilityGate(axes, critical_buckets=tuple(_metrics()), axis_mode="com")
    payload = gate.state_dict()
    other = CapabilityGate(axes, critical_buckets=tuple(_metrics()), axis_mode="head_com")
    try:
        other.load_state_dict(payload)
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("axis mismatch must fail closed")

