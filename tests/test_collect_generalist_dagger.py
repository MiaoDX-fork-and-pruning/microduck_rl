from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def test_dagger_entrypoint_exists():
    assert (ROOT / "scripts" / "collect_generalist_dagger.py").exists()

def test_replay_validation_requires_contract_fields():
    mod = _load("dagger", "scripts/collect_generalist_dagger.py")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_replay_batch({})

def test_replay_validation_checks_lengths_and_shapes():
    mod = _load("dagger2", "scripts/collect_generalist_dagger.py")
    data = {"observation": np.zeros((2, 61), np.float32), "requested_command": np.zeros((2, 13), np.float32), "raw_action": np.zeros((2, 14), np.float32), "previous_action": np.zeros((1, 14), np.float32)}
    with pytest.raises(ValueError, match="inconsistent lengths"):
        mod.validate_replay_batch(data)

def test_behavior_balancing_is_three_way_and_seeded():
    mod = _load("bc", "scripts/train_generalist_bc.py")
    labels = np.array([0, 0, 1, 2, 2, 2])
    a = mod.balanced_indices(labels, seed=11)
    b = mod.balanced_indices(labels, seed=11)
    assert np.array_equal(a, b)
    assert [int(np.sum(labels[a] == i)) for i in range(3)] == [3, 3, 3]

def test_dataset_rejects_non_g0_behavior_labels():
    mod = _load("bc2", "scripts/train_generalist_bc.py")
    x = np.zeros((1, 71), np.float32); y = np.zeros((1, 14), np.float32)
    x[0, 51] = 1.0
    with pytest.raises(ValueError, match="out-of-scope"):
        mod.validate_dataset(x, y)


def _track_a_report():
    import json
    return json.loads((ROOT / "artifacts/generalist-v0/specialist-switch-track-a-final.json").read_text())


def test_extract_boundary_windows_covers_four_legal_edges():
    mod = _load("dagger_windows", "scripts/collect_generalist_dagger.py")
    report = _track_a_report()
    frames = [{"step": step, "policy_id": "velstand_flat"} for step in range(4500)]
    for transition in report["transitions"]:
        for step in range(transition["start_step"], transition["end_step"] + 1):
            frames[step]["policy_id"] = transition["policy_id"]
    windows = mod.extract_boundary_windows(report, frames, before=2, after=2)
    assert {w["transition_bucket"] for w in windows} == {
        "VELSTAND->VELOCITY", "VELOCITY->VELSTAND",
        "VELSTAND->SITSTAND", "SITSTAND->VELSTAND",
    }
    assert {w["transition_phase"] for w in windows} == {"pre", "post"}


def test_extract_boundary_windows_rejects_direct_velocity_sitstand_edge():
    mod = _load("dagger_bad_edge", "scripts/collect_generalist_dagger.py")
    report = _track_a_report()
    report["transitions"] = [
        {"start_step": 0, "end_step": 9, "policy_id": "velocity_flat"},
        {"start_step": 10, "end_step": 19, "policy_id": "sitstand_flat"},
    ]
    frames = [{"step": i, "policy_id": "velocity_flat" if i < 10 else "sitstand_flat"} for i in range(20)]
    with pytest.raises(ValueError, match="unsupported G0 transition"):
        mod.extract_boundary_windows(report, frames, before=0, after=1)


def test_velstand_frontier_is_exactly_8_plus_1_plus_8():
    mod = _load("dagger_frontier", "scripts/collect_generalist_dagger.py")
    frames = [{"reset_bucket": "recovery_face_up", "frontier_crossed": i >= 12} for i in range(25)]
    window = mod.first_frontier_window(frames)
    assert len(window) == 17
    assert [row["frontier_offset"] for row in window] == list(range(-8, 9))


def test_frontier_excludes_nominal_and_unrecoverable_buckets():
    mod = _load("dagger_recovery", "scripts/collect_generalist_dagger.py")
    nominal = [{"reset_bucket": "upright", "frontier_crossed": i == 10} for i in range(20)]
    unrecoverable = [{"reset_bucket": "recovery_face_down", "frontier_crossed": i == 10,
                      "physically_unrecoverable": True} for i in range(20)]
    assert mod.first_frontier_window(nominal) == []
    assert mod.first_frontier_window(unrecoverable) == []


def test_dagger_rounds_are_cumulative():
    mod = _load("dagger_cumulative", "scripts/collect_generalist_dagger.py")
    base = {"inputs": np.zeros((2, 71)), "actions": np.zeros((2, 14))}
    one = {"inputs": np.ones((3, 71)), "actions": np.ones((3, 14))}
    two = {"inputs": np.full((4, 71), 2), "actions": np.full((4, 14), 2)}
    merged = mod.cumulative_rounds(base, [one, two])
    assert merged["inputs"].shape == (9, 71)
    assert np.all(merged["inputs"][-4:] == 2)
