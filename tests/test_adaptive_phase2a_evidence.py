from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report, canonical_sha256


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest = _load("assemble_adaptive_phase2a_manifest")
calibration = _load("calibrate_adaptive_native_gate")


def _report(root, *, steps=300, task=manifest.FIXED_TASK, parity=False):
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {"name": "native_mjlab_bam_v2", "environment_profile": "training", "steps": steps, "commands": manifest.CANONICAL_COMMANDS, "bucket_isolation": "fresh_environment"}
    raw = {bucket: {"survival_fraction": 1.0, "tilt_p95_rad": 0.0, "zero_drift_m": 0.0, "tracking_error_m_s": 0.0, "angular_tracking_error_rad_s": 0.0, "episode_length_mean": steps * 0.02, "fall_rate": 0.0, "action_magnitude_mean": 0.0} for bucket in BUCKETS}
    report = build_capability_report(raw, metadata={"task_id": task, "source_sha": "test-source", "evaluator_config_sha256": "pending", "checkpoint": str(checkpoint), "checkpoint_sha256": manifest.sha256(checkpoint), "policy_format": "native_mjlab_bam_pt", "seed_set_id": "test", "evaluation_seed": 777, "generated_at": "now"}, evaluator_config=config).payload
    report["metadata"]["evaluator_config_sha256"] = canonical_sha256(report["evaluator_config"])
    cases, seed_cases = [], []
    for offset, bucket in enumerate(BUCKETS):
        path = root / f"{bucket}.npz"
        arrays = {"observation": np.zeros((steps, 61)), "action": np.zeros((steps, 14)),
                  "terminated": np.zeros(steps, dtype=bool), "truncated": np.zeros(steps, dtype=bool),
                  "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (steps, 1)),
                  "root_position": np.zeros((steps, 3)), "initial_root_position": np.zeros(3),
                  "command": np.zeros((steps, 13)), "linear_velocity_b": np.zeros((steps, 3)),
                  "angular_velocity_b": np.zeros((steps, 3)),
                  "initial_joint_pos": np.zeros(14), "initial_joint_vel": np.zeros(14),
                  "reset_body_ipos": np.zeros((1, 16, 3)), "reset_dof_armature": np.ones((1, 20)),
                  "reset_friction_scale": np.ones((1, 1))}
        arrays["command"][:, :3] = manifest.CANONICAL_COMMANDS[bucket]
        arrays["observation"][:, 48:] = arrays["command"]
        arrays["linear_velocity_b"][:] = arrays["command"][:, :3]
        arrays["angular_velocity_b"][:] = arrays["command"][:, :3]
        if parity:
            arrays["onnx_action"] = np.zeros((steps, 14))
        np.savez_compressed(path, **arrays)
        seed_cases.append({"bucket": bucket, "reset_seed": 777 + offset,
                           "consumed_state_sha256": canonical_sha256({field: arrays[field].tolist() for field in manifest.CONSUMED_STATE_FIELDS})})
        cases.append({"bucket": bucket, "steps": steps, "finite_61d_14d": True, "trace": str(path), "trace_sha256": manifest.sha256(path), "parity": {"passed": True} if parity else None})
    report["cases"] = cases
    report["seed_manifest"] = {"version": "native-reset-dr-v3", "seed_set_id": "test", "startup_seed": 777,
                               "cases": seed_cases, "consumed_state_fields": list(manifest.CONSUMED_STATE_FIELDS)}
    path = root / "native_capability.json"
    path.write_text(json.dumps(report))
    return path, report


def test_legacy_reports_and_vacuous_parity_cannot_complete(tmp_path):
    legacy = tmp_path / "parity.json"
    legacy.write_text(json.dumps({"schema_version": 1, "buckets": {}, "aggregate": None}))
    with pytest.raises(ValueError, match="legacy"):
        manifest.validate_native(legacy, require_parity=True)
    result = manifest.assemble(tmp_path / "ladder", tmp_path / "final", tmp_path / "cpu", legacy, legacy, tmp_path / "manifest.json")
    assert result["status"] == "incomplete"
    assert result["verdicts"]["export_observation_parity"] == "unverified"
    assert result["verdicts"]["seed_consumption"] == "unverified"
    assert result["verdicts"]["matched_budget_campaign_authorized"] is False


def test_valid_native_report_checks_actual_traces(tmp_path):
    path, _ = _report(tmp_path, parity=True)
    manifest.validate_native(path, require_parity=True)
    (tmp_path / "forward.npz").unlink()
    with pytest.raises(ValueError, match="trace missing"):
        manifest.validate_native(path)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(aggregate=None), "aggregate"),
    (lambda p: p.update(cases=[]), "six"),
    (lambda p: p["cases"][0].update(parity=None), "parity case"),
    (lambda p: p["metadata"].update(evaluator_config_sha256="wrong"), "config hash"),
])
def test_invalid_evidence_rejected(tmp_path, mutation, match):
    path, payload = _report(tmp_path, parity=True)
    mutation(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        manifest.validate_native(path, require_parity=True)


def test_short_native_trace_rejected(tmp_path):
    path, _ = _report(tmp_path, steps=20)
    with pytest.raises(ValueError, match="shorter"):
        manifest.validate_native(path)


def test_calibration_refuses_legacy_and_short_evidence(tmp_path):
    root = tmp_path / "reports"
    (root / "legacy").mkdir(parents=True)
    (root / "legacy" / "native_capability.json").write_text('{"schema_version": 1}')
    _report(root / "short", steps=20)
    result = calibration.calibrate(root, tmp_path / "calibration.json")
    assert result["status"] == "incomplete"
    assert result["accepted_report_count"] == 0
    assert len(result["errors"]) == 2
    assert result["calibrated_parameters"] == {}


def test_calibration_summarizes_only_fixed_without_inventing_ema(tmp_path):
    root = tmp_path / "reports"
    _report(root / "fixed")
    _report(root / "static", task=manifest.STATIC_TASK)
    result = calibration.calibrate(root, tmp_path / "calibration.json")
    assert result["status"] == "incomplete"
    assert result["accepted_report_count"] == 1
    assert len(result["excluded_reports"]) == 1
    assert "ema_alpha" in result["missing_calibration"]
    assert result["product_gate_error_limits"]["tracking_m_s"] == pytest.approx(0.024)


def _replay_fixture(tmp_path, *, different_seed=900, change_consumed=True):
    paths = [_report(tmp_path / name)[0] for name in ('a', 'b', 'different')]
    path = paths[2]
    payload = json.loads(path.read_text())
    payload['metadata'].update(evaluation_seed=different_seed, seed_set_id='heldout')
    payload['seed_manifest'].update(startup_seed=different_seed, seed_set_id='heldout')
    for index, (case, seed_case) in enumerate(zip(payload['cases'], payload['seed_manifest']['cases'], strict=True)):
        seed_case['reset_seed'] = different_seed + index
        trace_path = Path(case['trace'])
        with np.load(trace_path) as trace:
            arrays = dict(trace)
        if change_consumed:
            arrays['initial_joint_pos'] += 0.01
        np.savez_compressed(trace_path, **arrays)
        case['trace_sha256'] = manifest.sha256(trace_path)
        seed_case['consumed_state_sha256'] = canonical_sha256({field: arrays[field].tolist() for field in manifest.CONSUMED_STATE_FIELDS})
    path.write_text(json.dumps(payload))
    return paths


def test_replay_validates_exact_arrays_and_changed_consumed_state(tmp_path):
    paths = _replay_fixture(tmp_path)
    proof = manifest.validate_replay(*paths)
    assert proof['status'] == 'validated'
    assert len(proof['buckets']) == 6
    assert all(item['different_seed_changed_consumed_fields'] == ['initial_joint_pos'] for item in proof['buckets'])
    assert proof['reports'][0]['sha256'] == manifest.sha256(paths[0])


def test_replay_rejects_overlapping_consumed_seeds(tmp_path):
    paths = _replay_fixture(tmp_path, different_seed=780)
    with pytest.raises(ValueError, match='overlap'):
        manifest.validate_replay(*paths)


def test_different_seed_label_alone_is_not_proof(tmp_path):
    paths = _replay_fixture(tmp_path, change_consumed=False)
    with pytest.raises(ValueError, match='did not change consumed state'):
        manifest.validate_replay(*paths)


def test_replay_checks_unscored_trace_fields_too(tmp_path):
    paths = _replay_fixture(tmp_path)
    for number, path in enumerate(paths[:2]):
        payload = json.loads(path.read_text())
        case = payload['cases'][0]
        trace_path = Path(case['trace'])
        with np.load(trace_path) as trace:
            arrays = dict(trace)
        arrays['reward'] = np.full(300, number, dtype=np.float32)
        np.savez_compressed(trace_path, **arrays)
        case['trace_sha256'] = manifest.sha256(trace_path)
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='trace mismatch: zero.reward'):
        manifest.validate_replay(*paths)


def test_replay_rejects_different_valid_config(tmp_path):
    paths = _replay_fixture(tmp_path)
    payload = json.loads(paths[1].read_text())
    payload['evaluator_config']['implementation_sha256'] = 'another implementation'
    payload['metadata']['evaluator_config_sha256'] = canonical_sha256(payload['evaluator_config'])
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='evaluator_config_sha256 mismatch'):
        manifest.validate_replay(*paths)


def _ownership_xml(path, *, names=None, failure=False, skipped=False):
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    names = names if names is not None else ['test_adaptive_factory_is_separate_static_initial_slice', 'test_adaptive_factory_exposes_explicit_axis_modes']
    suite = Element('testsuite', tests=str(len(names)), failures='0', errors='0')
    for name in names:
        case = SubElement(suite, 'testcase', classname='tests.test_mjlab_adaptive_velocity_config', name=name)
        if failure:
            SubElement(case, 'failure')
        if skipped:
            SubElement(case, 'skipped')
    ElementTree(suite).write(path)
    return path


def test_ownership_junit_requires_real_named_testcases(tmp_path):
    path = _ownership_xml(tmp_path / 'ownership.xml')
    proof = manifest.validate_ownership_junit(path)
    assert proof['test_count'] == 2
    assert proof['sha256'] == manifest.sha256(path)


@pytest.mark.parametrize('kwargs,match', [
    ({'names': []}, 'no tests'),
    ({'names': ['test_something_else']}, 'term-set tests'),
    ({'failure': True}, 'failed tests'),
    ({'skipped': True}, 'term-set tests'),
])
def test_invalid_ownership_junit_rejected(tmp_path, kwargs, match):
    path = _ownership_xml(tmp_path / 'ownership.xml', **kwargs)
    with pytest.raises(ValueError, match=match):
        manifest.validate_ownership_junit(path)


def test_assembly_binds_proofs_without_claiming_missing_ladder_complete(tmp_path):
    a, b, different = _replay_fixture(tmp_path)
    junit = _ownership_xml(tmp_path / 'ownership.xml')
    args = dict(native_ladder=tmp_path / 'missing', native_final=tmp_path / 'missing', cpu_ladder=tmp_path / 'missing', calibration=tmp_path / 'missing.json', parity=a, output=tmp_path / 'manifest.json', ownership_junit=junit, replay_a=a, replay_b=b, replay_different=different)
    report = manifest.assemble(**args)
    assert report['status'] == 'incomplete'
    assert report['verdicts']['seed_consumption'] == 'pass'
    assert report['verdicts']['adaptive_config_ownership'] == 'pass'
    assert report['seed_replay_evidence']['reports'][0]['sha256'] == manifest.sha256(a)
    args['replay_different'] = None
    report = manifest.assemble(**args)
    assert report['verdicts']['seed_consumption'] == 'unverified'
    assert any('not attached' in error for error in report['errors'])
