import json
from pathlib import Path

import torch

from mjlab_microduck.generalist_temporal import H4Actor


def test_h4_export_reports_215d(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    source = Path(__file__).parents[1] / "scripts" / "export_generalist_g0.py"
    spec = spec_from_file_location("export_h4", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "h4"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "schema": "generalist-g0-h4", "input_dim": 215,
        "metrics": {"bounded_actions": True, "input_dim": 215},
    }))
    torch.save({"state_dict": H4Actor().state_dict()}, run / "model.pt")
    report = module.export_g0(run, run / "policy.onnx", run / "golden.npz", run / "parity.json", samples=2)
    assert report["passed"] is True
    assert report["input_dim"] == 215
