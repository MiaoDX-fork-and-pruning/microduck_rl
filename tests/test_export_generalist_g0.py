import json
import importlib.util
from pathlib import Path

import torch

from mjlab_microduck.generalist_model import G0MultiHeadActor
_SPEC = importlib.util.spec_from_file_location("export_generalist_g0", Path(__file__).parents[1] / "scripts/export_generalist_g0.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
export_g0 = _MODULE.export_g0


def _run(tmp_path, multihead=False):
    model = G0MultiHeadActor() if multihead else torch.nn.Sequential(
        torch.nn.Linear(71, 8), torch.nn.Tanh(), torch.nn.Linear(8, 14)
    )
    run = tmp_path / ("multi" if multihead else "dense")
    run.mkdir()
    torch.save({"state_dict": model.state_dict()}, run / "model.pt")
    (run / "manifest.json").write_text(json.dumps({"metrics": {
        "model_kind": "g0_multihead" if multihead else "dense",
        "architecture": [71, 256, 256, 14] if multihead else [71, 8, 14],
        "bounded_actions": multihead,
    }}))
    report = export_g0(run, run / "policy.onnx", run / "golden.npz", run / "parity.json", samples=5)
    assert report["passed"] is True
    assert report["input_dim"] == 71 and report["action_dim"] == 14
    with torch.no_grad():
        pass
    return run


def test_export_dense_and_golden_parity(tmp_path):
    run = _run(tmp_path)
    import numpy as np
    with np.load(run / "golden.npz") as data:
        assert data["observations"].shape == (5, 71)
        assert data["pt_actions"].shape == (5, 14)


def test_export_multihead(tmp_path):
    _run(tmp_path, multihead=True)
