import importlib.util
from pathlib import Path
import sys

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "run_official_runtime_mujoco",
    Path(__file__).parents[1] / "scripts" / "run_official_runtime_mujoco.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_official_joint_mapping_has_one_explicit_mouth_slot():
    assert len(_MODULE.OFFICIAL_JOINTS) == 15
    assert _MODULE.MOUTH_INDEX == 9
    assert _MODULE.OFFICIAL_JOINTS[_MODULE.MOUTH_INDEX] == "mouth"
    policy_order = _MODULE.OFFICIAL_JOINTS[:9] + _MODULE.OFFICIAL_JOINTS[10:]
    assert len(policy_order) == 14


def test_target_mapping_removes_only_the_mouth_slot():
    official = np.arange(15, dtype=float)
    servo = np.concatenate(
        (official[: _MODULE.MOUTH_INDEX], official[_MODULE.MOUTH_INDEX + 1 :])
    )
    assert servo.tolist() == list(range(9)) + list(range(10, 15))
