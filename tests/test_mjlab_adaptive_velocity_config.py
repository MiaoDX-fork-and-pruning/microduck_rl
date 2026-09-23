import pytest

from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    AdaptiveMicroduckRlCfg,
    AdaptiveMicroduckComRlCfg,
    AdaptiveMicroduckHeadComRlCfg,
    AdaptiveMicroduckStaticRlCfg,
    AdaptiveMicroduckStandingRlCfg,
    AdaptiveMicroduckActionRateRlCfg,
    AdaptiveMicroduckLateralRlCfg,
    AdaptiveMicroduckTrackingRlCfg,
    AdaptiveMicroduckStrictificationRlCfg,
    AdaptiveMicroduckPushRlCfg,
    AdaptiveMicroduckAcquisitionFeedbackRlCfg,
    AdaptiveMicroduckLateralDriveRlCfg,
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


def test_resume_inherits_action_rate_relief_contract(monkeypatch, tmp_path) -> None:
    import torch

    from mjlab_microduck.tasks.adaptive_curriculum import AdaptiveActionRateRelief

    relief = AdaptiveActionRateRelief(
        relief_weight=-0.4, trigger_threshold=0.5, release_threshold=0.85,
        active_windows=3, cooldown_windows=2,
    )
    checkpoint = tmp_path / "smoothing.pt"
    torch.save({"infos": {"adaptive_curriculum": {"action_rate_relief": relief.state_dict()}}}, checkpoint)
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", str(checkpoint))
    cfg = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="lateral_drive", command_exposure=True)
    restored = AdaptiveActionRateRelief(
        relief_weight=cfg.adaptive_action_rate_relief_weight,
        trigger_threshold=cfg.adaptive_action_rate_relief_trigger,
        release_threshold=cfg.adaptive_action_rate_relief_release,
        active_windows=cfg.adaptive_action_rate_relief_windows,
        cooldown_windows=cfg.adaptive_action_rate_relief_cooldown_windows,
        scope=cfg.adaptive_action_rate_relief_scope,
    )
    restored.load_state_dict(relief.state_dict())
    assert restored.state_dict() == relief.state_dict()


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


def test_final_range_finetune_parses_as_fixed_final_distribution(monkeypatch) -> None:
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_RANGE_FINETUNE", "1")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL", "250")
    cfg = make_microduck_adaptive_velocity_env_cfg(axis_mode="composed")

    assert cfg.adaptive_final_range_finetune is True
    assert cfg.adaptive_evaluation_interval == 0
    assert cfg.adaptive_evaluation_distribution == "final"
    assert cfg.adaptive_final_com_fraction == 0.0
    assert cfg.adaptive_entropy_consolidation is False


def test_final_range_finetune_rejects_rehearsal_fraction(monkeypatch) -> None:
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_RANGE_FINETUNE", "1")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION", "0.2")
    with pytest.raises(ValueError, match="final-range fine-tuning"):
        make_microduck_adaptive_velocity_env_cfg(axis_mode="composed")


def test_final_range_environment_does_not_break_static_task_registration(monkeypatch) -> None:
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_RANGE_FINETUNE", "1")
    cfg = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static")
    assert cfg.adaptive_final_range_finetune is False


def test_adaptive_experiment_branches_have_distinct_log_names() -> None:
    names = {
        AdaptiveMicroduckStaticRlCfg.experiment_name,
        AdaptiveMicroduckComRlCfg.experiment_name,
        AdaptiveMicroduckHeadComRlCfg.experiment_name,
        AdaptiveMicroduckStandingRlCfg.experiment_name,
        AdaptiveMicroduckActionRateRlCfg.experiment_name,
        AdaptiveMicroduckLateralRlCfg.experiment_name,
        AdaptiveMicroduckTrackingRlCfg.experiment_name,
        AdaptiveMicroduckPushRlCfg.experiment_name,
    }
    names.add(AdaptiveMicroduckAcquisitionFeedbackRlCfg.experiment_name)
    names.add(AdaptiveMicroduckLateralDriveRlCfg.experiment_name)
    assert len(names) == 10


def test_diagnostic_modes_isolate_one_canonical_curriculum_term() -> None:
    standing = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="standing")
    action_rate = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="action_rate")
    assert list(standing.curriculum) == ["standing_envs"]
    assert list(action_rate.curriculum) == ["action_rate_weight"]


def test_lateral_diagnostic_adds_explicit_command_bucket() -> None:
    lateral = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="lateral")
    assert lateral.adaptive_axis_mode == "all_static"
    assert lateral.commands["twist"].rel_lateral_envs == 0.20
    assert "standing_envs" in lateral.curriculum
    assert "action_rate_weight" in lateral.curriculum


def test_tracking_diagnostic_changes_only_linear_reward_width() -> None:
    from copy import deepcopy
    from mjlab.tasks.registry import load_env_cfg

    baseline = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static")
    tracking = load_env_cfg("Mjlab-Velocity-Flat-Adaptive-Tracking-MicroDuck")
    assert tracking.task_id == "Mjlab-Velocity-Flat-Adaptive-Tracking-MicroDuck"
    assert tracking.adaptive_axis_mode == "all_static"
    assert tracking.observations == baseline.observations
    assert tracking.actions == baseline.actions
    assert tracking.commands == baseline.commands
    assert tracking.events == baseline.events
    assert tracking.curriculum == baseline.curriculum
    expected = deepcopy(baseline.rewards)
    expected["track_linear_velocity"].params["std"] = 0.12
    assert tracking.rewards == expected
    assert baseline.rewards["track_linear_velocity"].params["std"] ** 2 > 0.099


def test_tracking_reward_separates_stationary_from_accurate_motion() -> None:
    from types import SimpleNamespace
    import torch

    cfg = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static", diagnostic_mode="tracking")
    term = cfg.rewards["track_linear_velocity"]
    command = torch.tensor([[0., 0.12, 0.]]).repeat(3, 1)
    actual = torch.tensor([[0., 0., 0.], [0., 0.096, 0.], [0., 0.12, 0.]])
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=actual))},
    )
    reward = term.func(env, **term.params) * term.weight
    assert reward[0] < 0.74
    assert reward[1] > 1.92
    assert reward[2] == 2.0


def test_acquisition_diagnostic_freezes_other_wall_clock_curricula() -> None:
    base = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static")
    cfg = make_microduck_adaptive_velocity_env_cfg(
        axis_mode="all_static", diagnostic_mode="acquisition"
    )
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-Acquisition-MicroDuck"
    assert cfg.commands["twist"].rel_lateral_envs == 0.20
    assert set(cfg.curriculum) == set(base.curriculum) | {"tracking_std"}
    assert {name: cfg.curriculum[name] for name in base.curriculum} == base.curriculum
    assert cfg.curriculum["tracking_std"].params["std_stages"][-1]["std"] == 0.12
    assert cfg.rewards["track_linear_velocity"].params["std"] ** 2 > 0.099


def test_acquisition_lateral_preserves_anchor_buckets() -> None:
    base = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static")
    cfg = make_microduck_adaptive_velocity_env_cfg(
        axis_mode="all_static", diagnostic_mode="acquisition_lateral"
    )
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-AcquisitionLateral-MicroDuck"
    assert cfg.commands["twist"].rel_lateral_envs == 0.50
    assert set(cfg.curriculum) == set(base.curriculum) | {"tracking_std"}
    assert {name: cfg.curriculum[name] for name in base.curriculum} == base.curriculum


def test_acquisition_feedback_combines_staged_tracking_with_adaptive_exposure() -> None:
    from mjlab.tasks.registry import load_env_cfg

    base = make_microduck_adaptive_velocity_env_cfg(axis_mode="composed")
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Adaptive-AcquisitionFeedback-MicroDuck")
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-AcquisitionFeedback-MicroDuck"
    assert cfg.adaptive_axis_mode == "composed"
    assert cfg.adaptive_command_exposure is True
    assert cfg.adaptive_initial_focus == "lateral"
    assert cfg.adaptive_frontier_order[:2] == ("lateral", "forward")
    assert set(cfg.curriculum) == set(base.curriculum) - {"standing_envs"} | {"tracking_std"}
    assert {name: cfg.curriculum[name] for name in base.curriculum if name != "standing_envs"} == {
        name: base.curriculum[name] for name in base.curriculum if name != "standing_envs"
    }
    assert cfg.curriculum["tracking_std"].params["std_stages"][1]["std"] == 0.22
    assert {"linear_velocity_error_l1", "yaw_velocity_error_l1"} <= set(cfg.rewards)
    assert cfg.observations == make_microduck_adaptive_velocity_env_cfg().observations
    assert cfg.actions == make_microduck_adaptive_velocity_env_cfg().actions


def test_lateral_drive_increases_only_lateral_feedback_mass() -> None:
    from mjlab.tasks.registry import load_env_cfg

    base = make_microduck_adaptive_velocity_env_cfg(axis_mode="composed")
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck")
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck"
    assert cfg.adaptive_axis_mode == "composed"
    assert cfg.adaptive_command_exposure is True
    assert cfg.adaptive_initial_focus == "lateral"
    assert cfg.adaptive_frontier_order[0] == "lateral"
    assert cfg.rewards["linear_velocity_error_l1"].weight == 2.0
    assert cfg.rewards["yaw_velocity_error_l1"].weight == 1.0
    assert cfg.rewards["linear_velocity_error_l1"].params["yaw_deadband"] == 0.05
    assert cfg.adaptive_frontier_stall_windows == 4
    assert cfg.adaptive_frontier_stall_improvement == 0.05
    assert cfg.adaptive_action_rate_relief is True
    assert cfg.adaptive_action_rate_relief_weight == -0.2
    assert cfg.adaptive_action_rate_relief_windows == 4
    assert set(cfg.curriculum) == set(base.curriculum) - {"standing_envs"} | {"tracking_std"}
    assert {name: cfg.curriculum[name] for name in base.curriculum if name != "standing_envs"} == {
        name: base.curriculum[name] for name in base.curriculum if name != "standing_envs"
    }
    assert cfg.observations == make_microduck_adaptive_velocity_env_cfg().observations
    assert cfg.actions == make_microduck_adaptive_velocity_env_cfg().actions


def test_action_rate_relief_is_opt_in_to_lateral_drive_recipe() -> None:
    base = make_microduck_adaptive_velocity_env_cfg(axis_mode="composed")
    assert base.adaptive_action_rate_relief is False
    assert make_microduck_adaptive_velocity_env_cfg(
        axis_mode="composed", diagnostic_mode="acquisition_feedback"
    ).adaptive_action_rate_relief is False


@pytest.mark.parametrize("saved_scope", [None, "pure_yaw"])
def test_resume_restores_action_rate_relief_scope_before_env_creation(monkeypatch, tmp_path, saved_scope):
    import torch
    from mjlab_microduck.tasks import mdp

    relief = {"version": 1} if saved_scope is None else {"version": 2, "scope": saved_scope}
    checkpoint = tmp_path / "relief.pt"
    torch.save({"infos": {"adaptive_curriculum": {"action_rate_relief": relief}}}, checkpoint)
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", str(checkpoint))
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE", raising=False)
    cfg = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="lateral_drive")
    assert cfg.adaptive_action_rate_relief_scope == (saved_scope or "all")
    assert (cfg.rewards["action_rate_l2"].func is mdp.adaptive_action_rate_l2) == (saved_scope == "pure_yaw")


def test_pure_yaw_relief_experiment_is_explicit_and_validated(monkeypatch):
    from mjlab_microduck.tasks import mdp

    monkeypatch.setenv("MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE", "pure_yaw")
    cfg = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="lateral_drive")
    assert cfg.adaptive_action_rate_relief_scope == "pure_yaw"
    assert cfg.rewards["action_rate_l2"].func is mdp.adaptive_action_rate_l2
    base = make_microduck_adaptive_velocity_env_cfg()
    assert base.rewards["action_rate_l2"].func is not mdp.adaptive_action_rate_l2
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE", "invalid")
    with pytest.raises(ValueError, match="relief scope"):
        make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="lateral_drive")


def test_frontier_stall_window_override_is_explicit_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FRONTIER_STALL_WINDOWS", "12")
    cfg = make_microduck_adaptive_velocity_env_cfg(
        diagnostic_mode="lateral_drive", command_exposure=True
    )
    assert cfg.adaptive_frontier_stall_windows == 12
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FRONTIER_STALL_WINDOWS", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        make_microduck_adaptive_velocity_env_cfg(
            diagnostic_mode="lateral_drive", command_exposure=True
        )


def test_strictification_bootstrap_is_bounded_and_restores_strict_profile() -> None:
    cfg = make_microduck_adaptive_velocity_env_cfg(
        axis_mode="all_static", diagnostic_mode="strictification"
    )
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-Strictification-MicroDuck"
    assert cfg.commands["twist"].rel_forward_envs == 0.0
    assert cfg.commands["twist"].rel_lateral_envs == 0.25
    assert cfg.rewards["track_linear_velocity"].weight == 4.0
    assert cfg.rewards["track_angular_velocity"].weight == 6.0
    assert cfg.rewards["pose"].weight == 0.5
    assert cfg.rewards["air_time"].weight == 1.0
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    assert cfg.terminations["root_height"].params["min_height"] == 0.0
    stages = cfg.curriculum["strictification_profile"].params["profile_stages"]
    assert stages[0]["rel_lateral_envs"] == 0.25
    assert stages[1]["step"] == 500 * 24
    assert stages[-1]["action_rate_l2"] == -0.6
    assert AdaptiveMicroduckStrictificationRlCfg.experiment_name == (
        "velocity_adaptive_strictification_diagnostic"
    )


def test_push_diagnostic_only_adds_live_push_curriculum() -> None:
    base = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static")
    push = make_microduck_adaptive_velocity_env_cfg(axis_mode="all_static", diagnostic_mode="push")
    assert push.task_id == "Mjlab-Velocity-Flat-Adaptive-Push-MicroDuck"
    assert set(push.curriculum) == set(base.curriculum) | {"push_strength"}
    assert push.events["push_robot"].params["velocity_range"] == {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
    assert push.rewards == base.rewards
    assert push.commands == base.commands


def test_feedback_sampler_has_single_owner_and_preserves_policy_contract():
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.mdp import AdaptiveVelocityCommandCfg

    base = make_microduck_adaptive_velocity_env_cfg()
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Adaptive-Feedback-MicroDuck")
    assert cfg.adaptive_command_exposure
    assert cfg.adaptive_axis_mode == "composed"
    assert isinstance(cfg.commands["twist"], AdaptiveVelocityCommandCfg)
    assert set(cfg.curriculum) == set(base.curriculum) - {"standing_envs"}
    assert cfg.observations == base.observations
    assert cfg.actions == base.actions
    assert set(cfg.rewards) == set(base.rewards) | {
        "linear_velocity_error_l1", "yaw_velocity_error_l1"
    }
    assert {name: cfg.rewards[name] for name in base.rewards} == base.rewards
    assert cfg.events == base.events
    for name in ("head_pose", "body_pose"):
        assert cfg.commands[name] == base.commands[name]
    assert cfg.commands["twist"].ranges == base.commands["twist"].ranges


def test_legacy_cohort_migration_requires_explicit_launch_flag(monkeypatch) -> None:
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_EVALUATION_COHORT_SIZE", "3")
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_ALLOW_LEGACY_COHORT_MIGRATION", raising=False)
    cfg = make_microduck_adaptive_velocity_env_cfg()
    assert cfg.adaptive_evaluation_cohort_size == 3
    assert cfg.adaptive_allow_legacy_cohort_migration is False
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_ALLOW_LEGACY_COHORT_MIGRATION", "1")
    cfg = make_microduck_adaptive_velocity_env_cfg()
    assert cfg.adaptive_allow_legacy_cohort_migration is True


def test_sensor_corner_coverage_is_opt_in_and_preserves_fixed_recipe(monkeypatch) -> None:
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION", raising=False)
    base = make_microduck_adaptive_velocity_env_cfg()
    assert base.adaptive_sensor_corner_fraction == 0.0
    assert "adaptive_sensor_corners" not in base.events

    monkeypatch.setenv("MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION", "0.25")
    cfg = make_microduck_adaptive_velocity_env_cfg()
    assert cfg.adaptive_sensor_corner_fraction == 0.25
    assert cfg.events["adaptive_sensor_corners"].mode == "startup"
    assert cfg.events["adaptive_sensor_corners"].params["fraction"] == 0.25
    assert cfg.observations == base.observations
    assert cfg.actions == base.actions

    monkeypatch.setenv("MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION", "1.1")
    with pytest.raises(ValueError, match="sensor corner fraction"):
        make_microduck_adaptive_velocity_env_cfg()


def test_sensor_reset_resampling_is_opt_in_and_exclusive(monkeypatch) -> None:
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION", raising=False)
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION", raising=False)
    base = make_microduck_adaptive_velocity_env_cfg()
    assert base.adaptive_sensor_reset_fraction == 0.0
    assert "adaptive_sensor_resample" not in base.events

    monkeypatch.setenv("MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION", "1.0")
    cfg = make_microduck_adaptive_velocity_env_cfg()
    assert cfg.adaptive_sensor_reset_fraction == 1.0
    assert cfg.events["adaptive_sensor_resample"].mode == "reset"
    assert cfg.events["adaptive_sensor_resample"].params["fraction"] == 1.0
    assert cfg.observations == base.observations
    assert cfg.actions == base.actions

    monkeypatch.setenv("MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION", "0.25")
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_microduck_adaptive_velocity_env_cfg()

    monkeypatch.delenv("MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION", raising=False)
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION", "1.1")
    with pytest.raises(ValueError, match="sensor reset fraction"):
        make_microduck_adaptive_velocity_env_cfg()


def test_resume_checkpoint_recreates_sensor_reset_event(monkeypatch, tmp_path) -> None:
    import torch

    checkpoint = tmp_path / "sensor-reset.pt"
    torch.save(
        {
            "infos": {
                "adaptive_curriculum": {"sensor_reset_fraction": 1.0}
            }
        },
        checkpoint,
    )
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION", raising=False)
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", str(checkpoint))
    cfg = make_microduck_adaptive_velocity_env_cfg()
    assert cfg.adaptive_sensor_reset_fraction == 1.0
    assert cfg.events["adaptive_sensor_resample"].params["fraction"] == 1.0


def test_registered_task_ids_match_checkpoint_provenance():
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import ADAPTIVE_RECIPES

    for axis, diagnostic, feedback, _ in ADAPTIVE_RECIPES:
        cfg = make_microduck_adaptive_velocity_env_cfg(
            axis_mode=axis, diagnostic_mode=diagnostic, command_exposure=feedback
        )
        assert load_env_cfg(cfg.task_id).task_id == cfg.task_id
