import importlib.util
from pathlib import Path
import sys


_SPEC = importlib.util.spec_from_file_location(
    "render_specialist_evaluation_jobs",
    Path(__file__).parents[1] / "scripts" / "render_specialist_evaluation_jobs.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


ROOT = Path(__file__).parents[1]


def test_matrix_renders_unique_calibration_jobs_for_remaining_policies():
    jobs = _MODULE.load_jobs(
        ROOT / "cloudml/specialist-s2-evaluations.json",
        ROOT / "cloudml/specialist-final-checkpoints-652b7ce.json",
    )
    rendered = [
        _MODULE.render(
            job,
            source_prefix="/dongxu/microduck_rl/source/deadbee-20260831T1200",
            revision="deadbee",
        )
        for job in jobs
    ]

    assert len(rendered) == 12
    assert len({job["jobName"] for job in rendered}) == 12
    assert all(job["preemptible"] is True for job in rendered)
    assert all(job["queueId"] == "11759" for job in rendered)
    assert all(
        job["resourceConfigs"][0]["perNodeResourceSpec"]
        == {
            "resourcePriority": "GUARANTEED",
            "resourceName": "cloudml.ng1r49-8-8.13-107",
            "resourceNumber": 1,
        }
        for job in rendered
    )
    assert all(
        "--episodes 32" in job["imageConfig"]["imageCommand"] for job in rendered
    )
    assert all(
        "--video-seconds 15" in job["imageConfig"]["imageCommand"] for job in rendered
    )
    assert all(
        "--report-only" in job["imageConfig"]["imageCommand"] for job in rendered
    )
    assert all(job["juiceFsMountConfigs"][1]["readOnly"] is True for job in rendered)
    assert all(job["juiceFsMountConfigs"][2]["readOnly"] is False for job in rendered)
    assert all(
        job["juiceFsMountConfigs"][2]["subPath"].endswith("-deadbee-v1")
        for job in rendered
    )
