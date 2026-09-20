from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    AdaptiveMicroduckRlCfg,
    AdaptiveMicroduckComRlCfg,
    AdaptiveMicroduckHeadComRlCfg,
    AdaptiveMicroduckStaticRlCfg,
    AdaptiveMicroduckStandingRlCfg,
    AdaptiveMicroduckActionRateRlCfg,
    make_microduck_adaptive_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from mjlab_microduck.tasks import AdaptiveMicroduckOnPolicyRunner


def test_adaptive_factory_is_separate_static_initial_slice() -> None:
    canonical = make_microduck_velocity_env_cfg()
    adaptive = make_microduck_adaptive_velocity_env_cfg()
    assert adaptive is not canonical
    assert set(adaptive.curriculum) == set(canonical.curriculum) - {"com_range", "head_com_range"}
    assert adaptive.observations == canonical.observations
    assert adaptive.actions == canonical.actions
    assert adaptive.commands["twist"].rel_standing_envs == 0.02


def test_adaptive_runner_has_distinct_experiment_name() -> None:
    assert AdaptiveMicroduckRlCfg.experiment_name == "velocity_adaptive"
    assert AdaptiveMicroduckRlCfg.run_name == "velocity_adaptive"
    assert AdaptiveMicroduckOnPolicyRunner.__name__ == "AdaptiveMicroduckOnPolicyRunner"


def test_adaptive_factory_exposes_explicit_axis_modes() -> None:
    canonical = make_microduck_velocity_env_cfg()
    all_static = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static")
    com = make_microduck_adaptive_velocity_env_cfg(axis_mode="com")
    head = make_microduck_adaptive_velocity_env_cfg(axis_mode="head_com")
    composed = make_microduck_adaptive_velocity_env_cfg(axis_mode="composed")
    assert all_static.adaptive_axis_mode == "all_static"
    assert com.adaptive_axis_mode == "com"
    assert set(all_static.curriculum) == set(canonical.curriculum) - {"com_range", "head_com_range"}
    assert set(com.curriculum) == set(canonical.curriculum) - {"com_range"}
    assert set(head.curriculum) == set(canonical.curriculum) - {"head_com_range"}
    assert set(composed.curriculum) == set(canonical.curriculum) - {"com_range", "head_com_range"}


def test_adaptive_experiment_branches_have_distinct_log_names() -> None:
    names = {
        AdaptiveMicroduckStaticRlCfg.experiment_name,
        AdaptiveMicroduckComRlCfg.experiment_name,
        AdaptiveMicroduckHeadComRlCfg.experiment_name,
        AdaptiveMicroduckStandingRlCfg.experiment_name,
        AdaptiveMicroduckActionRateRlCfg.experiment_name,
    }
    assert len(names) == 5


def test_diagnostic_modes_isolate_one_canonical_curriculum_term() -> None:
    standing = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="standing")
    action_rate = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="action_rate")
    assert list(standing.curriculum) == ["standing_envs"]
    assert list(action_rate.curriculum) == ["action_rate_weight"]
