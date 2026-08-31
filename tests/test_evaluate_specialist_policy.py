import importlib.util
import json
from pathlib import Path
import sys


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
