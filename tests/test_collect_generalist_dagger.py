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
