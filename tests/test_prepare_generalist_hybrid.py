import importlib.util
import json
from pathlib import Path

import torch

from mjlab_microduck.generalist_model import G0MultiHeadActor


def _module():
    path = Path(__file__).parents[1] / "scripts/prepare_generalist_hybrid.py"
    spec = importlib.util.spec_from_file_location("bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_validates_and_emits_distinct_commands(tmp_path):
    root = tmp_path / "bc"
    root.mkdir()
    torch.save({"state_dict": G0MultiHeadActor().state_dict()}, root / "model.pt")
    (root / "manifest.json").write_text(json.dumps({"schema": "generalist-v0", "schema_version": 2,
        "input_dim": 71, "action_dim": 14, "model_kind": "g0_multihead", "bounded_actions": True}))
    result = _module().prepare(root, tmp_path / "bridge.json", 64, 5)
    assert result["direct"]["experiment_name"] != result["hybrid"]["experiment_name"]
    assert result["hybrid"]["initialization"] == "bc_actor"
    assert "--env.scene.num-envs 64" in result["direct"]["command"]


def test_prepare_rejects_wrong_schema(tmp_path):
    root = tmp_path / "bc"
    root.mkdir()
    (root / "model.pt").write_bytes(b"x")
    (root / "manifest.json").write_text(json.dumps({"schema": "other", "schema_version": 1}))
    import pytest
    with pytest.raises(ValueError, match="schema"):
        _module().validate_bc_artifact(root)
