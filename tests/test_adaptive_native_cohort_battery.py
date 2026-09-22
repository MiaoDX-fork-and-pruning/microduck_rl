"""Contract tests for conservative native seed-cohort evaluation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from mjlab_microduck.evaluation.capability import (
    BUCKETS,
    DEFAULT_INSTANTANEOUS_CAPS,
    DEFAULT_THRESHOLDS,
    TRACKING_METRIC_SIGNED_EMA,
    build_capability_report,
    canonical_sha256,
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


native = _load("run_adaptive_native_battery")
cohort = _load("run_adaptive_native_cohort_battery")


def _member(tmp_path: Path, seed: int, *, yaw_error: float = 0.1) -> dict:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cohort-checkpoint")
    config = {
        "name": "native_mjlab_bam_v2",
        "steps": 300,
        "commands": native.BUCKETS,
        "tracking_metric": TRACKING_METRIC_SIGNED_EMA,
        "tracking_metric_tau_s": 0.5,
        "thresholds": DEFAULT_THRESHOLDS,
        "instantaneous_caps": DEFAULT_INSTANTANEOUS_CAPS,
    }
    raw = {}
    for bucket in BUCKETS:
        if bucket == "zero":
            raw[bucket] = {"survival_fraction": 1.0, "tilt_p95_rad": 0.05, "zero_drift_m": 0.01}
        elif bucket in ("forward", "lateral"):
            raw[bucket] = {
                "survival_fraction": 1.0,
                "tilt_p95_rad": 0.05,
                "tracking_error_m_s": 0.03,
                "tracking_error_samplewise_m_s": 0.03,
            }
        else:
            raw[bucket] = {
                "survival_fraction": 1.0,
                "tilt_p95_rad": 0.05,
                "angular_tracking_error_rad_s": yaw_error,
                "angular_tracking_error_samplewise_rad_s": yaw_error,
            }
    payload = build_capability_report(
        raw,
        axis_mode="composed",
        evaluator_config=config,
        metadata={
            "task_id": "fake-task",
            "source_sha": "source",
            "evaluator_config_sha256": "pending",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": native.sha256(checkpoint),
            "policy_format": "native_mjlab_bam_pt",
            "seed_set_id": "cohort-test",
            "evaluation_seed": seed,
            "generated_at": "test",
        },
    ).payload
    payload["metadata"]["evaluator_config_sha256"] = canonical_sha256(payload["evaluator_config"])
    cases = []
    for bucket in BUCKETS:
        trace_path = tmp_path / f"trace-{seed}-{bucket}.bin"
        trace_path.write_bytes(f"trace-{seed}-{bucket}".encode())
        cases.append({
            "bucket": bucket,
            "steps": 300,
            "trace": str(trace_path.resolve()),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        })
    payload["cases"] = cases
    payload["seed_manifest"] = {
        "version": "native-reset-dr-v3",
        "seed_set_id": "cohort-test",
        "startup_seed": seed,
        "cases": [{"bucket": bucket, "reset_seed": seed + offset} for offset, bucket in enumerate(BUCKETS)],
        "consumed_state_fields": [],
    }
    return payload


def _aggregate(tmp_path: Path, *, yaw_a: float = 0.1, yaw_b: float = 0.8) -> dict:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cohort-checkpoint")
    members = [
        {"seed": 10, "path": str((tmp_path / "member-10.json").resolve()), "payload": _member(tmp_path, 10, yaw_error=yaw_a)},
        {"seed": 11, "path": str((tmp_path / "member-11.json").resolve()), "payload": _member(tmp_path, 11, yaw_error=yaw_b)},
    ]
    for member in members:
        member["payload"]["report_sha256"] = canonical_sha256(member["payload"])
        Path(member["path"]).write_text(
            json.dumps(member["payload"]), encoding="utf-8"
        )
    return cohort.aggregate_cohort_reports(
        members,
        checkpoint=checkpoint,
        task="fake-task",
        axis_mode="composed",
        seed_set_id="cohort-test",
        cohort_seeds=(10, 11),
        source_sha="source",
    )


def test_cohort_seed_manifest_is_deterministic_and_rejects_duplicates():
    assert cohort.resolve_cohort_seeds(10, cohort_size=3) == (10, 11, 12)
    assert cohort.resolve_cohort_seeds(10, cohort_size=2, cohort_seeds=(10, 20)) == (10, 20)
    with pytest.raises(ValueError, match="unique"):
        cohort.resolve_cohort_seeds(10, cohort_seeds=(10, 10))
    with pytest.raises(ValueError, match="first cohort seed"):
        cohort.resolve_cohort_seeds(10, cohort_seeds=(11, 12))


def test_same_cohort_is_reproducible_and_worst_bucket_keeps_raw_evidence(tmp_path):
    first = _aggregate(tmp_path)
    second = _aggregate(tmp_path)
    assert first["metadata"]["cohort"] == second["metadata"]["cohort"]
    assert first["seed_manifest"]["selected_seed_by_bucket"] == {
        "zero": 10,
        "forward": 10,
        "lateral": 10,
        "yaw": 11,
        "turn-left": 11,
        "turn-right": 11,
    }
    assert first["buckets"]["yaw"]["raw"]["angular_tracking_error_rad_s"] == pytest.approx(0.8)
    assert first["aggregate"]["lower_tail_score"] == first["buckets"]["yaw"]["score"]
    assert cohort.CapabilityReport.from_dict(first).payload["schema_version"] == 2


def test_cohort_report_cannot_forge_conservative_aggregate(tmp_path):
    report = _aggregate(tmp_path)
    forged = copy.deepcopy(report)
    forged["aggregate"]["passed"] = True
    with pytest.raises(ValueError, match="aggregate"):
        cohort.CapabilityReport.from_dict(forged)


def test_cohort_report_cannot_forge_selected_member(tmp_path):
    report = _aggregate(tmp_path)
    forged = copy.deepcopy(report)
    forged["metadata"]["cohort"]["selected_seed_by_bucket"]["yaw"] = 10
    forged["cohort_manifest"]["selected_seed_by_bucket"]["yaw"] = 10
    forged.pop("report_sha256", None)
    forged["report_sha256"] = canonical_sha256(forged)
    from mjlab_microduck.tasks.adaptive_runner import _validate_cohort_envelope

    with pytest.raises(ValueError, match="worst member"):
        _validate_cohort_envelope(
            forged,
            expected_size=2,
            evaluation_seed=10,
            task_id="fake-task",
            axis_mode="composed",
            seed_set_id="cohort-test",
            checkpoint=tmp_path / "model.pt",
        )


def test_run_cohort_scrubs_training_transition_overrides(monkeypatch, tmp_path):
    for name in cohort._TRANSITION_ENV_NAMES:
        monkeypatch.setenv(name, "training-only")
    calls = []

    def fake_run_native(checkpoint, output, **kwargs):
        assert all(name not in os.environ for name in cohort._TRANSITION_ENV_NAMES)
        calls.append(kwargs["seed"])
        payload = _member(tmp_path, kwargs["seed"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "native_capability.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    monkeypatch.setattr(cohort, "run_native", fake_run_native)
    output = tmp_path / "cohort.json"
    report = cohort.run_cohort(
        tmp_path / "model.pt",
        output,
        task="fake-task",
        evaluation_seed=10,
        seed_set_id="cohort-test",
        axis_mode="composed",
        source_sha="source",
        cohort_size=2,
        device="cpu",
    )
    assert calls == [10, 11]
    assert output.exists()
    assert report["metadata"]["cohort"]["seeds"] == [10, 11]
    assert os.environ["MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY"] == "training-only"
