from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report, resolve_enabled_axes

META = {
    "task_id": "x", "source_sha": "s", "evaluator_config_sha256": "c",
    "checkpoint": "p", "checkpoint_sha256": "h", "policy_format": "synthetic",
    "seed_set_id": "test", "generated_at": "now",
}


def _raw(error=0.001, angular=0.001, drift=0.001):
    return {
        bucket: {
            "survival_fraction": 1.0,
            "tilt_p95_rad": 0.05,
            "tracking_error_m_s": error,
            "angular_tracking_error_rad_s": angular,
            "zero_drift_m": drift,
        }
        for bucket in BUCKETS
    }


def test_good_and_upright_but_stationary_reports_differ():
    good = build_capability_report(_raw(), metadata=META)
    bad_raw = _raw()
    bad_raw["forward"]["tracking_error_m_s"] = 0.20
    bad = build_capability_report(bad_raw, metadata=META)
    assert good.payload["aggregate"]["passed"]
    assert bad.payload["buckets"]["forward"]["score"] < good.payload["buckets"]["forward"]["score"]
    assert bad.payload["aggregate"]["lower_tail_score"] < good.payload["aggregate"]["lower_tail_score"]


def test_wrong_turn_is_not_hidden_by_upright():
    raw = _raw()
    raw["turn-left"]["angular_tracking_error_rad_s"] = 1.0
    report = build_capability_report(raw, metadata=META)
    assert report.payload["buckets"]["turn-left"]["score"] < 0.1
    assert not report.payload["aggregate"]["passed"]


def test_missing_or_nonfinite_data_is_invalid():
    raw = _raw()
    raw["zero"].pop("zero_drift_m")
    report = build_capability_report(raw, metadata=META)
    assert not report.payload["buckets"]["zero"]["valid"]
    assert not report.payload["aggregate"]["valid"]


def test_axis_modes_are_explicit_and_hash_is_deterministic():
    assert resolve_enabled_axes("com") == ("com_range",)
    assert resolve_enabled_axes("head_com") == ("head_com_range",)
    report = build_capability_report(_raw(), metadata=META, axis_mode="com")
    assert report.payload["enabled_axes"] == ["com_range"]
    assert report.sha256() == report.sha256()
