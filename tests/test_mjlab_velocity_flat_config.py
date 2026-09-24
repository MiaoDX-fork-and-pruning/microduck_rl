from __future__ import annotations

from pathlib import Path

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)


ROOT = Path(__file__).resolve().parents[1]
CFG_SOURCE = ROOT / "src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py"


def _stage_steps(cfg, curriculum_name: str, stage_name: str) -> list[int]:
    stages = cfg.curriculum[curriculum_name].params[stage_name]
    return [int(stage["step"]) for stage in stages]


def test_velocity_curriculum_stages_use_control_step_units() -> None:
    cfg = make_microduck_velocity_env_cfg()

    assert _stage_steps(cfg, "action_rate_weight", "weight_stages") == [
        step * NUM_STEPS_PER_ENV
        for step in (0, 500, 750, 1000, 1250, 1500)
    ]
    assert _stage_steps(cfg, "standing_envs", "standing_stages") == [
        step * NUM_STEPS_PER_ENV
        for step in (0, 500, 750, 1000, 1500, 2000)
    ]
    assert _stage_steps(cfg, "head_pose_range", "range_stages") == [
        step * NUM_STEPS_PER_ENV
        for step in (0, 500, 1000, 1500, 2000)
    ]
    assert _stage_steps(cfg, "com_range", "range_stages") == [
        step * NUM_STEPS_PER_ENV for step in (0, 500, 1000, 1500)
    ]
    assert _stage_steps(cfg, "head_com_range", "range_stages") == [
        step * NUM_STEPS_PER_ENV for step in (0, 500, 1000)
    ]

    # Keep the source from regressing to scattered rollout-length literals.
    assert "* 24" not in CFG_SOURCE.read_text()


def test_velocity_com_curricula_match_documented_final_ranges() -> None:
    cfg = make_microduck_velocity_env_cfg()

    trunk_stages = cfg.curriculum["com_range"].params["range_stages"]
    head_stages = cfg.curriculum["head_com_range"].params["range_stages"]
    assert trunk_stages[0]["range"] == 0.003
    assert trunk_stages[-1]["range"] == 0.015
    assert head_stages[0]["range"] == 0.003
    assert head_stages[-1]["range"] == 0.010
