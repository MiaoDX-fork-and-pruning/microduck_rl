import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("validate_generalist_fallback", ROOT / "scripts/validate_generalist_fallback.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest(tmp_path):
    source = json.loads((ROOT / "docs/plans/generalist-g0-teacher-manifest.json").read_text())
    for teacher in source["teachers"]:
        for path in teacher["artifacts"].values():
            src = ROOT / path
            dst = tmp_path / teacher["id"] / Path(path).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            teacher["artifacts"][next(k for k, v in teacher["artifacts"].items() if v == path)] = str(dst)
    # hashes remain valid because bytes are copied unchanged
    result = tmp_path / "manifest.json"
    result.write_text(json.dumps(source))
    return result, source


def test_frozen_specialist_fallback_is_preserved(tmp_path):
    path, _ = _manifest(tmp_path)
    report = MODULE.validate_fallback(path, expected_source_commit="facd4f4")
    assert report["valid"] and report["preserved"]


def test_rejects_specialist_abi_change(tmp_path):
    path, data = _manifest(tmp_path)
    data["teachers"][1]["action_dim"] = 15
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="specialist ABI mismatch"):
        MODULE.validate_fallback(path)
