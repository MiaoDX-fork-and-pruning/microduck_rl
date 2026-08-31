import importlib.util
import json
from pathlib import Path
import sys

import torch


_SPEC = importlib.util.spec_from_file_location(
    "evaluate_specialist_policy",
    Path(__file__).parents[1] / "scripts" / "evaluate_specialist_policy.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _episode(index=0, *, main=2.0, penalty=-0.2, finite=True, success=True):
    return {
        "id": index,
        "length_steps": 500,
        "length_seconds": 10.0,
        "total_reward": main + penalty,
        "reward_terms": {"task_progress": main, "action_rate_l2": penalty},
        "termination_reasons": ["time_out"],
        "success": success,
        "finite": finite,
    }


def _report(episodes):
    return _MODULE.build_evaluation_report(
        task="Mjlab-Test-Flat-MicroDuck",
        checkpoint=Path("model_100.pt"),
        checkpoint_sha256="a" * 64,
        seed=7,
        episodes=episodes,
        main_task_term="task_progress",
        success_threshold=1.0,
        minimum_success_rate=0.5,
        minimum_main_task_metric=1.0,
        penalty_names=["action_rate_l2"],
        video_path=Path("evaluation.mp4"),
        video_frame_count=500,
        video_fps=50.0,
        video_reset_count=0,
        video_review="Motion matches the task metric without a fall.",
    )


def test_report_is_deterministic_and_has_validator_contract():
    first = _report([_episode(0), _episode(1, main=1.5)])
    second = _report([_episode(0), _episode(1, main=1.5)])

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["accepted"] is True
    assert first["finite"] is True
    assert first["success_rate"] == 1.0
    assert first["main_task_metric"] == 1.75
    assert first["penalty_terms"] == {"action_rate_l2": -0.2}
    assert first["termination_counts"] == {"time_out": 2}
    assert first["diagnostic_video_env_id"] == 0
    assert first["diagnostic_video_first_episode_id"] == 0
    assert first["diagnostic_video_duration_seconds"] == 10.0
    assert first["diagnostic_video_reset_count"] == 0
    assert len(first["episodes"]) == 2


def test_positive_penalty_cannot_be_accepted():
    report = _report([_episode(penalty=0.01)])

    assert report["penalty_terms"]["action_rate_l2"] == 0.01
    assert report["positive_penalty_terms"] == {"action_rate_l2": 0.01}
    assert report["acceptance_checks"]["penalties_non_positive"] is False
    assert report["accepted"] is False


def test_positive_penalty_in_one_episode_cannot_be_hidden_by_mean():
    report = _report(
        [
            _episode(0, penalty=0.1),
            _episode(1, penalty=-1.0),
        ]
    )

    assert report["penalty_terms"]["action_rate_l2"] == -0.45
    assert report["positive_penalty_terms"] == {"action_rate_l2": 0.1}
    assert report["acceptance_checks"]["penalties_non_positive"] is False
    assert report["accepted"] is False


def test_nonfinite_episode_forces_finite_and_acceptance_false():
    report = _report([_episode(finite=False, success=False)])

    assert report["finite"] is False
    assert report["acceptance_checks"]["finite"] is False
    assert report["accepted"] is False


def test_nan_termination_is_nonfinite_even_after_auto_reset():
    assert _MODULE.episode_is_finite(True, ["time_out"])
    assert not _MODULE.episode_is_finite(True, ["nan_state"])
    assert not _MODULE.episode_is_finite(False, [])


def test_failure_termination_cannot_count_as_success():
    assert _MODULE.episode_is_success(2.0, 1.0, ["time_out"])
    assert not _MODULE.episode_is_success(2.0, 1.0, ["fell_over"])
    assert not _MODULE.episode_is_success(2.0, 1.0, ["fallen_too_long"])
    assert not _MODULE.episode_is_success(2.0, 1.0, ["out_of_terrain_bounds"])


def test_penalty_discovery_covers_both_repository_conventions():
    assert _MODULE.is_penalty_term("joint_torques_l2", -0.01)
    assert _MODULE.is_penalty_term("height_stand_l1", 1.0)
    assert _MODULE.is_penalty_term("head_impact_penalty", 1.0)
    assert not _MODULE.is_penalty_term("task_progress", 1.0)


def test_default_video_path_is_stable():
    output = Path("artifacts/policy/evaluation.json")
    assert _MODULE.diagnostic_video_path(output, None) == Path(
        "artifacts/policy/evaluation.mp4"
    )
    requested = Path("artifacts/policy/clip.mp4")
    assert _MODULE.diagnostic_video_path(output, requested) == requested


def test_report_only_is_explicit_and_disabled_by_default():
    parser = _MODULE.make_parser()
    common = [
        "task",
        "--checkpoint",
        "model.pt",
        "--output",
        "evaluation.json",
        "--main-task-term",
        "task_progress",
        "--success-threshold",
        "1",
        "--minimum-success-rate",
        "0.5",
    ]

    assert parser.parse_args(common).report_only is False
    assert parser.parse_args([*common, "--report-only"]).report_only is True


def test_video_review_recomputes_acceptance_from_existing_checks():
    report = _report([_episode()])
    report["video_review"] = ""
    report["acceptance_checks"]["video_reviewed"] = False
    report["accepted"] = False

    result = _MODULE.apply_video_review(report, "  Motion matches metrics.  ")

    assert result["video_review"] == "Motion matches metrics."
    assert result["acceptance_checks"]["video_reviewed"] is True
    assert result["accepted"] is True


def test_video_review_cannot_override_a_failed_computed_check():
    report = _report([_episode(success=False)])
    report["acceptance_checks"]["success_rate"] = False
    report["accepted"] = False

    result = _MODULE.apply_video_review(report, "Motion reviewed.")

    assert result["acceptance_checks"]["video_reviewed"] is True
    assert result["accepted"] is False


def test_reassessment_recomputes_episode_and_report_gates_from_raw_evidence():
    report = _report([_episode(0, main=2.0), _episode(1, main=1.5)])

    result = _MODULE.reassess_evaluation_report(
        report,
        success_threshold=1.75,
        minimum_success_rate=0.5,
        minimum_main_task_metric=1.7,
    )

    assert [episode["success"] for episode in result["episodes"]] == [True, False]
    assert result["success_rate"] == 0.5
    assert result["main_task_metric"] == 1.75
    assert result["battery"]["success_threshold"] == 1.75
    assert result["acceptance_checks"] == {
        "finite": True,
        "main_task_metric": True,
        "penalties_non_positive": True,
        "success_rate": True,
        "video_reviewed": True,
    }
    assert result["accepted"] is True


def test_reassessment_cannot_hide_positive_penalty_or_missing_video_review():
    report = _report([_episode(penalty=0.1)])
    report["video_review"] = ""

    result = _MODULE.reassess_evaluation_report(
        report,
        success_threshold=0.0,
        minimum_success_rate=0.0,
        minimum_main_task_metric=0.0,
    )

    assert result["positive_penalty_terms"] == {"action_rate_l2": 0.1}
    assert result["acceptance_checks"]["penalties_non_positive"] is False
    assert result["acceptance_checks"]["video_reviewed"] is False
    assert result["accepted"] is False


def test_reassessment_rejects_high_metric_episode_with_failure_termination():
    episode = _episode(main=10.0)
    episode["termination_reasons"] = ["fell_over"]
    report = _report([episode])

    result = _MODULE.reassess_evaluation_report(
        report,
        success_threshold=1.0,
        minimum_success_rate=1.0,
        minimum_main_task_metric=1.0,
    )

    assert result["episodes"][0]["success"] is False
    assert result["success_rate"] == 0.0
    assert result["acceptance_checks"]["success_rate"] is False
    assert result["accepted"] is False


def test_cuda_tensor_state_check_does_not_call_numpy():
    class SimData:
        qpos = torch.ones(1, device="cuda" if torch.cuda.is_available() else "cpu")
        qvel = torch.ones(1, device=qpos.device)
        ctrl = torch.ones(1, device=qpos.device)

    class Env:
        sim = type("Sim", (), {"data": SimData()})()

    assert _MODULE._sim_state_finite(Env()) is True


def test_tensor_like_cuda_state_uses_torch_finiteness():
    class TensorLike:
        def __init__(self, value):
            self.value = value

        def __torch_function__(self, func, types, args=(), kwargs=None):
            del types
            kwargs = kwargs or {}
            converted = [
                arg.value if isinstance(arg, TensorLike) else arg for arg in args
            ]
            return func(*converted, **kwargs)

    class SimData:
        qpos = TensorLike(
            torch.ones(1, device="cuda" if torch.cuda.is_available() else "cpu")
        )
        qvel = TensorLike(qpos.value.clone())
        ctrl = TensorLike(qpos.value.clone())

    class Env:
        sim = type("Sim", (), {"data": SimData()})()

    assert _MODULE._sim_state_finite(Env()) is True
