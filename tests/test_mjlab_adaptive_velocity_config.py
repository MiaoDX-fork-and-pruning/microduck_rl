from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    AdaptiveMicroduckRlCfg,
    make_microduck_adaptive_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg


def test_adaptive_factory_is_separate_static_initial_slice() -> None:
    canonical = make_microduck_velocity_env_cfg()
    adaptive = make_microduck_adaptive_velocity_env_cfg()
    assert adaptive is not canonical
    assert adaptive.curriculum == {}
    assert adaptive.observations == canonical.observations
    assert adaptive.actions == canonical.actions


def test_adaptive_runner_has_distinct_experiment_name() -> None:
    assert AdaptiveMicroduckRlCfg.experiment_name == "velocity_adaptive"
    assert AdaptiveMicroduckRlCfg.run_name == "velocity_adaptive"

