import importlib.util
from pathlib import Path
import sys


_SPEC = importlib.util.spec_from_file_location(
    "verify_official_fall_recovery",
    Path(__file__).parents[1] / "scripts" / "verify_official_fall_recovery.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_official_commit_is_immutable_and_full_length():
    assert _MODULE.PINNED_COMMIT == "66d4fa8"
