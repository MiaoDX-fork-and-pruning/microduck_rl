import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import pytest
from train_generalist_bc import collect, dense_architecture


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
