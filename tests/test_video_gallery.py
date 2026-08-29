import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_video_gallery",
    Path(__file__).parents[1] / "scripts" / "build_video_gallery.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_manifest_gallery_indexes_evidence_and_escapes_text(tmp_path):
    video = tmp_path / "diagnostic.mp4"
    video.write_bytes(b"video")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "success_rate": 0.875,
        "main_task_metric": 1.25,
        "penalty_terms": {"impact": -0.25},
        "video_review": "upright <review>",
        "failure_note": "falls > threshold",
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"policies": [{
        "id": "stand&walk",
        "task": "Mjlab-Velocity-Flat-MicroDuck",
        "accepted": True,
        "artifacts": {
            "diagnostic_video": str(video),
            "evaluation_report": str(evaluation),
        },
        "sha256": {"diagnostic_video": "a" * 64, "onnx": "b" * 64},
    }]}))

    entries, evidence = _MODULE.collect_manifest_videos(manifest)
    index = _MODULE.build_gallery(entries, tmp_path / "gallery", "Review", evidence)
    document = index.read_text()
    assert "87.5%" in document
    assert "main metric: 1.25" in document
    assert "impact=-0.25" in document
    assert "upright &lt;review&gt;" in document
    assert "falls &gt; threshold" in document
    assert "stand&amp;walk" in document
    assert "b" * 64 in document
    assert (index.parent / "videos" / "stand_walk" / video.name).read_bytes() == b"video"


def test_manifest_rejects_missing_diagnostic_video(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"policies": [{
        "id": "stand", "artifacts": {"diagnostic_video": str(tmp_path / "missing.mp4")}
    }]}))
    try:
        _MODULE.collect_manifest_videos(manifest)
    except ValueError as exc:
        assert "missing or unsupported" in str(exc)
    else:
        raise AssertionError("missing video was accepted")
