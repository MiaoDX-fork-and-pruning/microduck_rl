"""Native trace evidence must support the existing product gate, including failures."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.evaluation.capability import build_capability_report, canonical_sha256


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native = _load("run_adaptive_native_battery")
manifest = _load("assemble_adaptive_phase2a_manifest")


@pytest.mark.parametrize("distribution,widths", [("initial", (0.003, 0.003)), ("final", (0.015, 0.010))])
def test_native_construction_disables_training_rehearsal(monkeypatch, tmp_path, distribution, widths):
    from mjlab.envs.mdp import dr

    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION", "0.2")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY", "0.2")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE", "0.2")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE", "zero")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE", "1")

    class CapturedConfiguration(Exception):
        pass

    def construct(*, cfg, device):
        assert cfg.adaptive_final_com_fraction == 0.0
        assert not cfg.adaptive_transition_acquisition
        assert cfg.adaptive_transition_probability_override is None
        assert not cfg.adaptive_transition_bootstrap_mode_override
        assert cfg.commands["twist"].transition_probability == 0.0
        for name, width in zip(("randomize_com", "randomize_head_com"), widths, strict=True):
            assert cfg.events[name].func is dr.body_ipos
            assert cfg.events[name].params["ranges"] == (-width, width)
        raise CapturedConfiguration

    monkeypatch.setattr("mjlab.envs.ManagerBasedRlEnv", construct)
    with pytest.raises(CapturedConfiguration):
        native.run_native(tmp_path / "actor.pt", tmp_path / "evidence", task="Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck",
                          seed=17, steps=300, device="cpu", distribution=distribution)


def _trace(bucket, steps=300, terminal=False):
    command = np.tile((*native.BUCKETS[bucket], *([0.] * 10)), (steps, 1))
    observation = np.zeros((steps, 61))
    observation[:, 48:] = command
    terminated = np.zeros(steps, dtype=bool)
    terminated[-1] = terminal
    return {
        "observation": observation, "action": np.zeros((steps, 14)), "command": command,
        "terminated": terminated, "truncated": np.zeros(steps, dtype=bool),
        "root_quaternion_wxyz": np.tile([1., 0., 0., 0.], (steps, 1)),
        "root_position": np.zeros((steps, 3)), "initial_root_position": np.zeros(3),
        "initial_joint_pos": np.zeros(14), "initial_joint_vel": np.zeros(14),
        "reset_body_ipos": np.zeros((1, 16, 3)), "reset_dof_armature": np.ones((1, 20)),
        "reset_friction_scale": np.ones((1, 1)),
        "linear_velocity_b": command[:, :3].copy(), "angular_velocity_b": command[:, :3].copy(),
        "onnx_action": np.zeros((steps, 14)),
    }


def _fixture(tmp_path, *, steps=300, terminal=False):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"real checkpoint fixture")
    config = {"name": "native_mjlab_bam_v2", "environment_profile": "training", "steps": 300,
              "commands": native.BUCKETS, "step_dt": 0.02, "bucket_isolation": "fresh_environment",
              "tracking_metric": "signed_ema_v1", "tracking_metric_tau_s": 0.5,
              "zero_mode": "nominal"}
    raw, cases, seed_cases = {}, [], []
    for offset, bucket in enumerate(native.BUCKETS):
        trace = _trace(bucket, steps, terminal)
        raw[bucket] = native.raw_from_trace(trace, bucket=bucket, expected_steps=300, dt=0.02)
        trace_path = tmp_path / f"{bucket}.npz"
        np.savez_compressed(trace_path, **trace)
        seed_cases.append({"bucket": bucket, "reset_seed": 777 + offset,
                           "consumed_state_sha256": canonical_sha256({field: trace[field].tolist() for field in manifest.CONSUMED_STATE_FIELDS})})
        cases.append({"bucket": bucket, "steps": steps, "requested_steps": 300, "trace": str(trace_path),
                      "trace_sha256": native.sha256(trace_path), "finite_61d_14d": True, "parity": {"passed": True}})
    report = build_capability_report(raw, evaluator_config=config, metadata={
        "task_id": manifest.FIXED_TASK, "source_sha": "source", "checkpoint": str(checkpoint),
        "checkpoint_sha256": native.sha256(checkpoint), "evaluator_config_sha256": "pending",
        "policy_format": "native_mjlab_bam_pt", "seed_set_id": "test", "evaluation_seed": 777, "generated_at": "now",
    }).payload
    report["metadata"]["evaluator_config_sha256"] = canonical_sha256(report["evaluator_config"])
    report["cases"] = cases
    report["seed_manifest"] = {"version": "native-reset-dr-v3", "seed_set_id": "test", "startup_seed": 777,
                               "cases": seed_cases, "consumed_state_fields": list(manifest.CONSUMED_STATE_FIELDS)}
    path = tmp_path / "native_capability.json"
    path.write_text(json.dumps(report))
    return path, report


def _mutate_trace(path, report, field, change):
    case = report["cases"][0]
    trace_path = Path(case["trace"])
    with np.load(trace_path) as source:
        trace = dict(source)
    trace[field] = change(trace[field])
    np.savez_compressed(trace_path, **trace)
    case["trace_sha256"] = manifest.sha256(trace_path)
    path.write_text(json.dumps(report))


def test_raw_axis_error_does_not_average_away_tracking_failure():
    trace = _trace("forward")
    trace["linear_velocity_b"][:, 0] = 0.0
    raw = native.raw_from_trace(trace, bucket="forward", expected_steps=300, dt=0.02)
    assert raw["tracking_error_m_s"] == pytest.approx(0.12)
    trace["linear_velocity_b"][:, 1] = 123.0
    assert native.raw_from_trace(trace, bucket="forward", expected_steps=300, dt=0.02)["tracking_error_m_s"] == pytest.approx(0.12)


def test_raw_drift_uses_reset_position_and_preserves_terminal_evidence():
    trace = _trace("zero", steps=20, terminal=True)
    trace["initial_root_position"][0] = 0.5
    raw = native.raw_from_trace(trace, bucket="zero", expected_steps=300, dt=0.02)
    assert raw["zero_drift_m"] == pytest.approx(0.5)
    assert raw["survival_fraction"] == pytest.approx(19 / 300)
    assert raw["episode_length_mean"] == pytest.approx(0.4)
    assert raw["fall_rate"] == 1.0


def test_early_terminal_is_valid_negative_evidence(tmp_path):
    path, report = _fixture(tmp_path, steps=20, terminal=True)
    assert report["aggregate"]["valid"]
    assert not report["aggregate"]["passed"]
    manifest.validate_native(path, expected_task=manifest.FIXED_TASK, require_parity=True)


def test_short_trace_without_terminal_is_incomplete(tmp_path):
    path, _ = _fixture(tmp_path, steps=20)
    with pytest.raises(ValueError, match="without terminal"):
        manifest.validate_native(path)


@pytest.mark.parametrize("field,change,match", [
    ("observation", lambda x: x[:, :60], "shape"),
    ("action", lambda x: x[:, :13], "shape"),
    ("action", lambda x: x * np.nan, "nonfinite"),
    ("root_position", lambda x: x * np.nan, "nonfinite"),
    ("root_position", lambda x: x + 1, "raw metric mismatch"),
    ("terminated", lambda x: np.ones(len(x), dtype=bool), "continues after terminal"),
    ("onnx_action", lambda x: x + 1, "parity trace fails"),
])
def test_trace_cannot_contradict_summary_even_with_updated_hash(tmp_path, field, change, match):
    path, report = _fixture(tmp_path)
    _mutate_trace(path, report, field, change)
    with pytest.raises(ValueError, match=match):
        manifest.validate_native(path, require_parity=True)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["metadata"].update(task_id="wrong"), "task"),
    (lambda p: p["metadata"].update(checkpoint_sha256="wrong"), "checkpoint"),
    (lambda p: p["metadata"].update(evaluator_config_sha256="wrong"), "config hash"),
    (lambda p: p.update(cases=[]), "six"),
    (lambda p: p["cases"][0].update(bucket="forward"), "each canonical"),
    (lambda p: p["cases"][0].update(parity={}), "parity case"),
])
def test_native_report_provenance_and_cases_are_required(tmp_path, mutation, match):
    path, report = _fixture(tmp_path)
    mutation(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        manifest.validate_native(path, expected_task=manifest.FIXED_TASK, require_parity=True)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["evaluator_config"].pop("bucket_isolation"), "fresh_environment"),
    (lambda p: p["evaluator_config"]["commands"].update(forward=[0.2, 0., 0.]), "canonical product commands"),
    (lambda p: p["seed_manifest"].update(cases=[]), "six canonical"),
    (lambda p: p["seed_manifest"]["cases"][0].update(bucket="forward"), "six canonical"),
    (lambda p: p["seed_manifest"].update(startup_seed=778), "startup seed"),
    (lambda p: p["seed_manifest"].update(seed_set_id="heldout"), "seed set mismatch"),
    (lambda p: p["seed_manifest"]["cases"][0].update(reset_seed=778), "reset seed mismatch"),
    (lambda p: p["seed_manifest"]["cases"][0].update(consumed_state_sha256="wrong"), "consumed state hash"),
    (lambda p: p["seed_manifest"].update(consumed_state_fields=["initial_root_position"]), "consumed state fields"),
])
def test_product_evidence_requires_isolation_commands_and_consumed_seeds(tmp_path, mutation, match):
    path, report = _fixture(tmp_path)
    mutation(report)
    # Rehash config to show a self-consistent hash cannot legitimize a changed protocol.
    report["metadata"]["evaluator_config_sha256"] = canonical_sha256(report["evaluator_config"])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        manifest.validate_native(path)


def test_reset_state_change_is_detected_even_if_trace_hash_updated(tmp_path):
    path, report = _fixture(tmp_path)
    _mutate_trace(path, report, "initial_joint_pos", lambda x: x + 0.1)
    with pytest.raises(ValueError, match="consumed state hash"):
        manifest.validate_native(path)
