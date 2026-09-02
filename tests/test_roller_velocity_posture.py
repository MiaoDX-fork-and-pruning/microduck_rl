import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_roller_velocity_posture.py"


def _module():
    spec = importlib.util.spec_from_file_location("diagnose_roller_velocity_posture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_posture_variants_are_mirrored():
    variants = _module().VARIANTS
    assert variants["canonical"] == (0.0, 0.0)
    assert variants["left_leg_ahead"] == tuple(-value for value in variants["right_leg_ahead"])
    assert variants["left_leg_ahead"][0] == variants["left_leg_ahead"][1]
