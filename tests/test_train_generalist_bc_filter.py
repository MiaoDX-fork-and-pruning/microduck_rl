import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import pytest
import numpy as np
from train_generalist_bc import collect, dense_architecture, train


def test_collect_can_filter_to_velstand():
    x, y, manifest = collect(Path("artifacts/generalist-v0/p0-action-battery-deadband-final"), ("stand",))
    assert len(x) == 300
    assert y.shape == (300, 14)
    assert manifest["behavior_counts"] == {"stand": 300, "locomotion": 0, "sit_stand": 0}


def test_capacity_arms_are_explicit_and_distinct():
    assert dense_architecture() == [71, 512, 256, 128, 14]
    assert dense_architecture(capacity_2x=True) == [71, 768, 384, 192, 14]
    assert dense_architecture(capacity_4x=True) == [71, 1088, 544, 272, 14]
    with pytest.raises(ValueError, match="exactly one"):
        dense_architecture(small=True, capacity_2x=True)


def test_gated_adapter_training_emits_reconstructable_metadata(tmp_path):
    x = np.zeros((6, 71), dtype=np.float32)
    y = np.zeros((6, 14), dtype=np.float32)
    for index in range(3):
        x[index, 48 + index] = 1.0
        x[index + 3, 48 + index] = 1.0
    metrics = train(x, y, tmp_path / "run", epochs=1, seed=3,
                    balance=False, gated_adapter=True)
    assert metrics["model_kind"] == "gated_adapter"
    assert metrics["hidden_dim"] == 256
    assert metrics["adapter_dim"] == 32
    assert metrics["behavior_count"] == 3

    import torch
    from mjlab_microduck.generalist_model import build_actor
    trained = build_actor(metrics)
    payload = torch.load(tmp_path / "run" / "model.pt", weights_only=False)
    trained.load_state_dict(payload["state_dict"], strict=True)
    assert not any(name.endswith("output_tanh") for name, _ in trained.named_modules())


def test_film_training_emits_reconstructable_metadata(tmp_path):
    x = np.zeros((6, 71), dtype=np.float32)
    y = np.zeros((6, 14), dtype=np.float32)
    for index in range(3):
        x[index, 48 + index] = 1.0
        x[index + 3, 48 + index] = 1.0
    metrics = train(x, y, tmp_path / "run", epochs=1, seed=3,
                    balance=False, film=True, bounded=True)
    assert metrics["model_kind"] == "film"
    assert metrics["hidden_dim"] == 512
    assert metrics["output_hidden_dim"] == 256


def test_canonical_training_entrypoint_exists():
    assert (Path(__file__).parents[1] / "scripts" / "train_generalist_canonical.py").exists()


def test_canonical_dagger_shards_preserve_segment_trajectories(tmp_path):
    # The entrypoint must keep adjacent student-state frames together so the
    # trajectory split cannot leak neighboring frames into validation.
    source = Path(__file__).parents[1] / "scripts" / "train_generalist_canonical.py"
    assert "np.unique(extra_segments.astype(str), return_inverse=True)" in source.read_text()


def test_gated_adapter_can_initialize_from_generalist_model(tmp_path):
    x = np.zeros((6, 71), dtype=np.float32)
    y = np.zeros((6, 14), dtype=np.float32)
    for index in range(3):
        x[index, 48 + index] = 1.0
        x[index + 3, 48 + index] = 1.0
    source = tmp_path / "source"
    target = tmp_path / "target"
    train(x, y, source, epochs=1, seed=3, balance=False,
          gated_adapter=True, bounded=True)
    metrics = train(x, y, target, epochs=1, seed=3, balance=False,
                    gated_adapter=True, bounded=True,
                    init_model=source / "model.pt")
    assert metrics["init_model"] == str(source / "model.pt")
