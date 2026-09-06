import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "replay_specialist_recovery_trace",
    ROOT / "scripts" / "replay_specialist_recovery_trace.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_legacy_recovery_trace_is_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, tilt_rad=np.array([0.9], dtype=np.float32))
    with pytest.raises(ValueError, match="integration state"):
        _MODULE.replay(path)
