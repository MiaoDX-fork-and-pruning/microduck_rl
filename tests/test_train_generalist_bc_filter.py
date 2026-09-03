import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from train_generalist_bc import collect


def test_collect_can_filter_to_velstand():
    x, y, manifest = collect(Path("artifacts/generalist-v0/p0-action-battery-deadband-final"), ("stand",))
    assert len(x) == 300
    assert y.shape == (300, 14)
    assert manifest["behavior_counts"] == {"stand": 300, "locomotion": 0, "sit_stand": 0}
