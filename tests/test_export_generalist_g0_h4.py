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
    import onnx
    metadata = {item.key: item.value for item in onnx.load(run / "policy.onnx").metadata_props}
    assert metadata == {
        "schema": "generalist-g0-h4", "schema_version": "1", "input_dim": "215",
        "action_dim": "14", "history_frames": "4", "proprioception_dim": "48",
        "condition_dim": "23", "frame_order": "oldest_to_newest",
        "padding": "repeat_first_frame", "segment_reset": "true", "control_hz": "50",
    }
