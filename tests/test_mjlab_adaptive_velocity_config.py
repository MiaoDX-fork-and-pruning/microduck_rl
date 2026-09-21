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
    cfg = make_microduck_adaptive_velocity_env_cfg(
        axis_mode="all_static", diagnostic_mode="acquisition"
    )
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-Acquisition-MicroDuck"
    assert cfg.commands["twist"].rel_lateral_envs == 0.20
    assert list(cfg.curriculum) == ["tracking_std"]
    assert cfg.curriculum["tracking_std"].params["std_stages"][-1]["std"] == 0.12
    assert cfg.rewards["track_linear_velocity"].params["std"] ** 2 > 0.099


def test_acquisition_lateral_preserves_anchor_buckets() -> None:
    cfg = make_microduck_adaptive_velocity_env_cfg(
        axis_mode="all_static", diagnostic_mode="acquisition_lateral"
    )
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-AcquisitionLateral-MicroDuck"
    assert cfg.commands["twist"].rel_lateral_envs == 0.50
    assert list(cfg.curriculum) == ["tracking_std"]


def test_acquisition_feedback_combines_staged_tracking_with_adaptive_exposure() -> None:
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Adaptive-AcquisitionFeedback-MicroDuck")
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-AcquisitionFeedback-MicroDuck"
    assert cfg.adaptive_axis_mode == "composed"
    assert cfg.adaptive_command_exposure is True
    assert cfg.adaptive_initial_focus == "lateral"
    assert cfg.adaptive_frontier_order[:2] == ("lateral", "forward")
    assert list(cfg.curriculum) == ["tracking_std"]
    assert cfg.curriculum["tracking_std"].params["std_stages"][1]["std"] == 0.22
    assert {"linear_velocity_error_l1", "yaw_velocity_error_l1"} <= set(cfg.rewards)
    assert cfg.observations == make_microduck_adaptive_velocity_env_cfg().observations
    assert cfg.actions == make_microduck_adaptive_velocity_env_cfg().actions


def test_lateral_drive_increases_only_lateral_feedback_mass() -> None:
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck")
    assert cfg.task_id == "Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck"
    assert cfg.adaptive_axis_mode == "composed"
    assert cfg.adaptive_command_exposure is True
    assert cfg.adaptive_initial_focus == "lateral"
    assert cfg.adaptive_frontier_order[0] == "lateral"
    assert cfg.rewards["linear_velocity_error_l1"].weight == 2.0
    assert cfg.rewards["yaw_velocity_error_l1"].weight == 0.0
    assert cfg.adaptive_frontier_stall_windows == 4
    assert cfg.adaptive_frontier_stall_improvement == 0.05
    assert cfg.observations == make_microduck_adaptive_velocity_env_cfg().observations
    assert cfg.actions == make_microduck_adaptive_velocity_env_cfg().actions


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


def test_registered_task_ids_match_checkpoint_provenance():
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import ADAPTIVE_RECIPES

    for axis, diagnostic, feedback, _ in ADAPTIVE_RECIPES:
        cfg = make_microduck_adaptive_velocity_env_cfg(
            axis_mode=axis, diagnostic_mode=diagnostic, command_exposure=feedback
        )
        assert load_env_cfg(cfg.task_id).task_id == cfg.task_id
