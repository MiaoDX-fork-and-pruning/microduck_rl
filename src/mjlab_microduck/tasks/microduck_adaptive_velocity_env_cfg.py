"""MJLab-only adaptive Velocity experiment configuration.

This factory is intentionally separate from the production fixed-schedule
Velocity recipe.  The capability controller lives in ``adaptive_curriculum``;
an evaluator or training adapter owns feeding it frozen battery metrics.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from . import mdp as microduck_mdp
from mjlab_microduck.evaluation.capability import resolve_enabled_axes


# At a 0.12 m/s command the canonical std=sqrt(0.1) awards a stationary
# robot 86.6% of the tracking maximum. Test a sharper signal separately
# from command sampling; do not change the canonical reward or product gate.
DIAGNOSTIC_LINEAR_TRACKING_STD = 0.12

# Acquisition diagnostic: keep the early walking signal reachable, then tighten
# the absolute tracking target only after a gait has had time to form. A fixed
# 0.12 m/s std makes the initial ±0.3/±0.4 m/s command support too sparse for
# this low-torque biped and causes early falls; the staged signal removes the
# stationary lateral-reward loophole without making iteration zero impossible.
ACQUISITION_TRACKING_STD_STAGES = (
    {"step": 0, "std": 0.31622776601683794},
    {"step": 500 * 24, "std": 0.22},
    {"step": 1000 * 24, "std": 0.16},
    {"step": 1500 * 24, "std": 0.12},
)

# Bounded bootstrap profile adapted from the independently validated
# strictification experiment. It is diagnostic-only until a native MJLab run
# shows that the early lateral basin transfers when the later strict stages are
# restored. Values are applied through the live managers by
# ``adaptive_strictification_profile``.
STRICTIFICATION_PROFILE_STAGES = (
    {
        "step": 0,
        "rel_forward_envs": 0.0,
        "rel_lateral_envs": 0.25,
        "track_linear_velocity": 4.0,
        "track_angular_velocity": 6.0,
        "pose": 0.5,
        "air_time": 1.0,
        "root_height": 0.0,
        "action_rate_l2": -0.1,
    },
    {
        "step": 500 * 24,
        "rel_forward_envs": 0.2,
        "rel_lateral_envs": 0.0,
        "track_linear_velocity": 2.0,
        "track_angular_velocity": 2.0,
        "pose": 1.0,
        "air_time": 3.0,
        "root_height": 0.055,
        "action_rate_l2": -0.2,
    },
    {
        "step": 750 * 24,
        "rel_forward_envs": 0.2,
        "rel_lateral_envs": 0.0,
        "track_linear_velocity": 2.0,
        "track_angular_velocity": 2.0,
        "pose": 1.0,
        "air_time": 3.0,
        "root_height": 0.055,
        "action_rate_l2": -0.4,
    },
    {
        "step": 1000 * 24,
        "rel_forward_envs": 0.2,
        "rel_lateral_envs": 0.0,
        "track_linear_velocity": 2.0,
        "track_angular_velocity": 2.0,
        "pose": 1.0,
        "air_time": 3.0,
        "root_height": 0.055,
        "action_rate_l2": -0.6,
    },
)


def _resume_sensor_reset_fraction() -> float | None:
    """Read the persisted sensor coverage before constructing a resume env.

    The adaptive runner restores its state after the environment exists, so a
    reset event must be registered during config construction. The launcher
    normally propagates this value as an environment variable; this fallback
    also makes direct ``train`` resumes reproduce the checkpoint distribution.
    """

    checkpoint_name = os.environ.get("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT")
    if not checkpoint_name:
        return None
    checkpoint = Path(checkpoint_name)
    if not checkpoint.is_file():
        raise ValueError(f"adaptive resume checkpoint does not exist: {checkpoint}")
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = (payload.get("infos") or {}).get("adaptive_curriculum")
    if not isinstance(state, dict) or "sensor_reset_fraction" not in state:
        return None
    try:
        fraction = float(state["sensor_reset_fraction"])
    except (TypeError, ValueError) as exc:
        raise ValueError("adaptive checkpoint sensor reset fraction must be numeric") from exc
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("adaptive checkpoint sensor reset fraction must be in [0, 1]")
    return fraction

# Feedback-only acquisition signal: at low commanded speeds the Gaussian
# tracking reward is nearly flat around standing. These dimensionless,
# command-aligned L1 penalties keep a direct cost for missing the requested
# direction while leaving orthogonal gait motion free. The normalizer floors
# deliberately remain at gait scale: smaller denominators would tax measured
# lateral/yaw oscillation of a walking biped heavily.
FEEDBACK_LINEAR_L1_WEIGHT = 0.5
FEEDBACK_YAW_L1_WEIGHT = 0.35
FEEDBACK_LINEAR_MIN_SCALE_M_S = 0.12
FEEDBACK_YAW_MIN_SCALE_RAD_S = 0.8
FEEDBACK_LINEAR_DEADBAND_M_S = 0.01
FEEDBACK_YAW_DEADBAND_RAD_S = 0.05
# Measured lateral gait: 0.049 m/s mean toward a 0.12 command, but 0.141
# instantaneous MAE (worse than standing). Average signed error over a gait
# cycle before L1 so learning that motion is not penalized as a regression.
FEEDBACK_TRACKING_TAU_S = 0.5
LATERAL_DRIVE_LINEAR_L1_WEIGHT = 2.0
# Once bounded frontier rotation hands focus to yaw, keep a direct command
# aligned signal alive.  The previous lateral-drive recipe set this to zero,
# so the controller increased yaw exposure without giving PPO a yaw-specific
# acquisition gradient.
LATERAL_DRIVE_YAW_L1_WEIGHT = 1.0


DIAGNOSTIC_NAMES = {
    "standing": "Standing", "action_rate": "ActionRate", "lateral": "Lateral",
    "tracking": "Tracking", "acquisition": "Acquisition",
    "acquisition_lateral": "AcquisitionLateral", "acquisition_feedback": "AcquisitionFeedback",
    "lateral_drive": "LateralDrive",
    "strictification": "Strictification",
    "push": "Push",
}


@dataclass(kw_only=True)
class AdaptiveVelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Persist adaptive launch settings in MJLab's dataclass configuration."""

    task_id: str = "Mjlab-Velocity-Flat-Adaptive-MicroDuck"
    adaptive_axis_mode: str = "composed"
    adaptive_command_exposure: bool = False
    adaptive_transition_acquisition: bool = False
    adaptive_transition_probability: float = 0.0
    adaptive_transition_bootstrap_mode: str = "forward"
    # A launch-only value is applied after a full checkpoint restore.  Keeping
    # this separate from the persisted controller state makes explicit CLI
    # overrides deterministic while ordinary resumes remain exact.
    adaptive_transition_probability_override: float | None = None
    adaptive_transition_bootstrap_mode_override: bool = False
    # A bounded reward relief used by the lateral-drive adaptive recipe when
    # the yaw frontier is still unacquired.  The runner owns its state and
    # persists it in the adaptive checkpoint; the canonical action-rate
    # curriculum remains the fallback outside the relief window.
    adaptive_action_rate_relief: bool = False
    adaptive_action_rate_relief_weight: float = -0.2
    adaptive_action_rate_relief_trigger: float = 0.55
    adaptive_action_rate_relief_release: float = 0.80
    adaptive_action_rate_relief_windows: int = 4
    adaptive_action_rate_relief_cooldown_windows: int = 1
    adaptive_initial_focus: str = "forward"
    adaptive_frontier_order: tuple[str, ...] = ()
    adaptive_frontier_stall_windows: int = 0
    adaptive_frontier_stall_improvement: float = 0.05
    adaptive_linear_feedback_weight: float = FEEDBACK_LINEAR_L1_WEIGHT
    adaptive_yaw_feedback_weight: float = FEEDBACK_YAW_L1_WEIGHT
    # Optional adaptive-only startup coverage for coupled sensor DR corners.
    # The canonical fixed Velocity recipe keeps the ordinary independent draws.
    adaptive_sensor_corner_fraction: float = 0.0
    # Optional adaptive-only reset resampling for the coupled sensor realization.
    # The realization stays fixed within an episode and is redrawn on reset.
    adaptive_sensor_reset_fraction: float = 0.0
    # None inherits the checkpoint (or zero on a fresh run). An explicit value
    # starts a recorded experiment override after resume; at most 20% is final.
    adaptive_final_com_fraction: float | None = None
    adaptive_evaluation_interval: int = 0
    adaptive_evaluation_seed: int = 20260916
    # Native gate cohort size.  The legacy single-seed evaluator remains the
    # default; campaign launchers opt into a fixed multi-seed cohort explicitly.
    adaptive_evaluation_cohort_size: int = 1
    # A single-seed checkpoint has incomparable mastery/rollback evidence. A
    # campaign must opt in explicitly when it re-baselines that evidence for a
    # larger native cohort while preserving the PPO state and cumulative budget.
    adaptive_allow_legacy_cohort_migration: bool = False
    # None inherits a resumed checkpoint, otherwise preserves the legacy final
    # gate. Changing this contract requires explicit evidence rebaselining.
    adaptive_evaluation_distribution: str | None = None
    adaptive_allow_distribution_migration: bool = False
    adaptive_evaluator_schema_version: int = 2
    adaptive_seed_set_id: str = "adaptive-gate-20260916"
    adaptive_evaluation_timeout_s: int = 900
    adaptive_source_sha: str = ""
    adaptive_evaluator_config_sha256: str = ""


def make_microduck_adaptive_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
    axis_mode: str = "composed",
    diagnostic_mode: str | None = None,
    command_exposure: bool = False,
    transition_probability: float | None = None,
    transition_bootstrap_mode: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Return the static initial slice used by adaptive curriculum experiments."""

    base = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg = AdaptiveVelocityEnvCfg(**{field.name: getattr(base, field.name) for field in fields(base)})
    canonical_curriculum = deepcopy(cfg.curriculum)
    # Preserve canonical curricula unless the adaptive controller explicitly
    # owns that axis.  The controller mutates the live EventManager ranges for
    # the selected DR axes; all standing, smoothing, command, pose, terrain,
    # and reward schedules remain canonical and continue to run normally.
    resolve_enabled_axes(axis_mode)
    owned_curriculum = {
        "com": {"com_range"},
        "head_com": {"head_com_range"},
        "composed": {"com_range", "head_com_range"},
        "all_static": {"com_range", "head_com_range"},
    }[axis_mode]
    cfg.curriculum = {
        name: term
        for name, term in canonical_curriculum.items()
        if name not in owned_curriculum
    }
    # This metadata is consumed by AdaptiveMicroduckOnPolicyRunner. It is kept
    # on the env config so each CloudML branch has an explicit axis contract.
    cfg.adaptive_axis_mode = axis_mode
    requested_transition = transition_probability
    if requested_transition is None and command_exposure:
        transition_env = os.environ.get("MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY")
        requested_transition = None if transition_env is None else float(transition_env)
    if requested_transition is not None:
        if not command_exposure:
            raise ValueError("transition acquisition requires command_exposure=True")
        if not 0.0 <= requested_transition <= 0.40:
            raise ValueError("transition acquisition probability must be in [0, 0.40]")
        cfg.adaptive_transition_probability = float(requested_transition)
        # Keep the controller constructible at probability zero so a saved
        # zero-probability state can still be restored exactly.  A zero value
        # remains behaviorally disabled; the object only owns its counters.
        cfg.adaptive_transition_acquisition = True
    if command_exposure:
        requested_bootstrap_mode = transition_bootstrap_mode
        if requested_bootstrap_mode is None:
            mode_env = os.environ.get("MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE")
            requested_bootstrap_mode = "forward" if mode_env is None else mode_env
        if requested_bootstrap_mode not in ("forward", "zero"):
            raise ValueError("transition bootstrap mode must be 'forward' or 'zero'")
        cfg.adaptive_transition_bootstrap_mode = requested_bootstrap_mode
        cfg.adaptive_transition_bootstrap_mode_override = transition_bootstrap_mode is not None
        if os.environ.get("MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE") == "1":
            cfg.adaptive_transition_bootstrap_mode_override = True
        transition_override = os.environ.get("MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE")
        if transition_override is not None:
            cfg.adaptive_transition_probability_override = float(transition_override)
            if not 0.0 <= cfg.adaptive_transition_probability_override <= 0.40:
                raise ValueError("transition acquisition override must be in [0, 0.40]")
    cfg.task_id = {
        "all_static": "Mjlab-Velocity-Flat-Adaptive-Static-MicroDuck",
        "com": "Mjlab-Velocity-Flat-Adaptive-CoM-MicroDuck",
        "head_com": "Mjlab-Velocity-Flat-Adaptive-HeadCoM-MicroDuck",
        "composed": "Mjlab-Velocity-Flat-Adaptive-MicroDuck",
    }[axis_mode]
    cfg.adaptive_source_sha = os.environ.get("MICRODUCK_SOURCE_SHA", "")
    cfg.adaptive_evaluator_config_sha256 = os.environ.get("MICRODUCK_ADAPTIVE_EVALUATOR_CONFIG_SHA256", "")
    cfg.adaptive_evaluation_interval = int(os.environ.get("MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL", "0"))
    cfg.adaptive_evaluation_seed = int(os.environ.get("MICRODUCK_ADAPTIVE_EVALUATION_SEED", "20260916"))
    cfg.adaptive_evaluation_cohort_size = int(
        os.environ.get("MICRODUCK_ADAPTIVE_EVALUATION_COHORT_SIZE", "1")
    )
    if cfg.adaptive_evaluation_cohort_size < 1:
        raise ValueError("adaptive evaluation cohort size must be positive")
    cfg.adaptive_allow_legacy_cohort_migration = (
        os.environ.get("MICRODUCK_ADAPTIVE_ALLOW_LEGACY_COHORT_MIGRATION", "0") == "1"
    )
    cfg.adaptive_evaluation_distribution = os.environ.get("MICRODUCK_ADAPTIVE_EVALUATION_DISTRIBUTION")
    if cfg.adaptive_evaluation_distribution not in (None, "final", "stage"):
        raise ValueError("adaptive evaluation distribution must be final or stage")
    cfg.adaptive_allow_distribution_migration = (
        os.environ.get("MICRODUCK_ADAPTIVE_ALLOW_DISTRIBUTION_MIGRATION", "0") == "1"
    )
    cfg.adaptive_evaluator_schema_version = 2
    cfg.adaptive_seed_set_id = os.environ.get("MICRODUCK_ADAPTIVE_SEED_SET_ID", "adaptive-gate-20260916")
    cfg.adaptive_evaluation_timeout_s = 900
    sensor_corner_fraction = os.environ.get("MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION")
    if sensor_corner_fraction is not None:
        try:
            cfg.adaptive_sensor_corner_fraction = float(sensor_corner_fraction)
        except ValueError as exc:
            raise ValueError("adaptive sensor corner fraction must be numeric") from exc
        if not 0.0 <= cfg.adaptive_sensor_corner_fraction <= 1.0:
            raise ValueError("adaptive sensor corner fraction must be in [0, 1]")
    if cfg.adaptive_sensor_corner_fraction > 0.0:
        cfg.events["adaptive_sensor_corners"] = EventTermCfg(
            func=microduck_mdp.randomize_sensor_corners,
            mode="startup",
            params={
                "fraction": cfg.adaptive_sensor_corner_fraction,
                "max_angle_deg": 6.0,
                "bias_range": (-0.015, 0.015),
            },
        )
    sensor_reset_fraction = os.environ.get("MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION")
    if sensor_reset_fraction is None:
        resumed_fraction = _resume_sensor_reset_fraction()
        if resumed_fraction is not None:
            cfg.adaptive_sensor_reset_fraction = resumed_fraction
    else:
        try:
            cfg.adaptive_sensor_reset_fraction = float(sensor_reset_fraction)
        except ValueError as exc:
            raise ValueError("adaptive sensor reset fraction must be numeric") from exc
        if not 0.0 <= cfg.adaptive_sensor_reset_fraction <= 1.0:
            raise ValueError("adaptive sensor reset fraction must be in [0, 1]")
    if cfg.adaptive_sensor_corner_fraction > 0.0 and cfg.adaptive_sensor_reset_fraction > 0.0:
        raise ValueError("sensor corner and sensor reset coverage are mutually exclusive")
    if cfg.adaptive_sensor_reset_fraction > 0.0:
        cfg.events["adaptive_sensor_resample"] = EventTermCfg(
            func=microduck_mdp.randomize_sensor_realization,
            mode="reset",
            params={
                "fraction": cfg.adaptive_sensor_reset_fraction,
                "max_angle_deg": 6.0,
                "bias_range": (-0.015, 0.015),
            },
        )
    final_com_fraction = os.environ.get("MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION")
    if final_com_fraction is not None:
        cfg.adaptive_final_com_fraction = float(final_com_fraction)
        if not 0.0 <= cfg.adaptive_final_com_fraction <= 0.20:
            raise ValueError("final CoM rehearsal fraction must be in [0, 0.20]")
    if cfg.adaptive_evaluation_interval < 0:
        raise ValueError("adaptive evaluation interval must be nonnegative")
    if diagnostic_mode is not None:
        if diagnostic_mode not in DIAGNOSTIC_NAMES:
            raise ValueError(f"unsupported adaptive diagnostic mode: {diagnostic_mode}")
        if diagnostic_mode == "standing":
            cfg.curriculum = {"standing_envs": canonical_curriculum["standing_envs"]}
        elif diagnostic_mode == "action_rate":
            cfg.curriculum = {"action_rate_weight": canonical_curriculum["action_rate_weight"]}
        elif diagnostic_mode == "lateral":
            # Preserve the canonical standing/action-rate/pose schedules while
            # adding an explicit pure-lateral command bucket. This is a
            # bounded recipe diagnostic and leaves the product task unchanged.
            cfg.commands["twist"].rel_lateral_envs = 0.20
        else:
            if diagnostic_mode == "tracking":
                cfg.rewards["track_linear_velocity"].params["std"] = DIAGNOSTIC_LINEAR_TRACKING_STD
            elif diagnostic_mode in (
                "acquisition", "acquisition_lateral", "acquisition_feedback", "lateral_drive"
            ):
                cfg.commands["twist"].rel_lateral_envs = 0.20 if diagnostic_mode == "acquisition" else 0.50
                # Add the diagnostic tracking signal on top of the filtered
                # canonical curriculum.  These recipes do not own action-rate,
                # standing, pose, terrain, or head-bias schedules; replacing
                # the mapping here silently disabled those schedules during
                # the long acquisition campaigns.
                cfg.curriculum["tracking_std"] = CurriculumTermCfg(
                    func=microduck_mdp.velocity_tracking_std_curriculum,
                    params={
                        "reward_name": "track_linear_velocity",
                        "std_stages": list(ACQUISITION_TRACKING_STD_STAGES),
                    },
                )
                if diagnostic_mode in ("acquisition_feedback", "lateral_drive"):
                    # The feedback controller owns command allocation. Start
                    # directly on the hard frontier instead of waiting for a
                    # failed gate window to switch from forward.
                    cfg.adaptive_initial_focus = "lateral"
                    cfg.adaptive_frontier_order = (
                        "lateral", "forward", "yaw", "turn-left", "turn-right"
                    )
                    # Keep a hard frontier from starving the remaining
                    # capabilities forever when its score oscillates below
                    # mastery. The controller still retains all bucket floors.
                    cfg.adaptive_frontier_stall_windows = 4
                    cfg.adaptive_frontier_stall_improvement = 0.05
                if diagnostic_mode == "lateral_drive":
                    cfg.adaptive_linear_feedback_weight = LATERAL_DRIVE_LINEAR_L1_WEIGHT
                    cfg.adaptive_yaw_feedback_weight = LATERAL_DRIVE_YAW_L1_WEIGHT
                    # The current blocker is a severe yaw acquisition deficit,
                    # not sensor noise. Enable the bounded adaptive relief so
                    # the controller can temporarily release the -1.0
                    # action-rate tax while it searches for that first motion.
                    cfg.adaptive_action_rate_relief = True
            elif diagnostic_mode == "strictification":
                # A bounded adapted-to-strict bootstrap. The command sampler
                # stays on the normal velocity path so the live curriculum can
                # mutate its explicit forward/lateral buckets.
                cfg.commands["twist"].rel_forward_envs = 0.0
                cfg.commands["twist"].rel_lateral_envs = 0.25
                cfg.rewards["track_linear_velocity"].weight = 4.0
                cfg.rewards["track_angular_velocity"].weight = 6.0
                cfg.rewards["pose"].weight = 0.5
                cfg.rewards["air_time"].weight = 1.0
                cfg.rewards["action_rate_l2"].weight = -0.1
                cfg.terminations["root_height"].params["min_height"] = 0.0
                cfg.curriculum = {
                    "strictification_profile": CurriculumTermCfg(
                        func=microduck_mdp.adaptive_strictification_profile,
                        params={"profile_stages": list(STRICTIFICATION_PROFILE_STAGES)},
                    )
                }
            else:
                cfg.curriculum["push_strength"] = CurriculumTermCfg(
                    func=microduck_mdp.push_curriculum,
                    params={
                        "event_name": "push_robot",
                        "push_stages": [
                            {"step": 0, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                            {"step": 500 * 24, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                            {"step": 1000 * 24, "velocity_range": {"x": (-0.22, 0.22), "y": (-0.22, 0.22)}},
                            {"step": 1500 * 24, "velocity_range": {"x": (-0.30, 0.30), "y": (-0.30, 0.30)}},
                        ],
                    },
                )
        cfg.task_id = f"Mjlab-Velocity-Flat-Adaptive-{DIAGNOSTIC_NAMES[diagnostic_mode]}-MicroDuck"
        if diagnostic_mode not in ("acquisition_feedback", "lateral_drive"):
            cfg.adaptive_axis_mode = "all_static"

    if command_exposure:
        if axis_mode != "composed" or diagnostic_mode not in (
            None, "acquisition_feedback", "lateral_drive"
        ):
            raise ValueError("feedback exposure requires the composed recipe or an acquisition diagnostic")
        cfg.task_id = (
            "Mjlab-Velocity-Flat-Adaptive-AcquisitionFeedback-MicroDuck"
            if diagnostic_mode == "acquisition_feedback"
            else "Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck"
            if diagnostic_mode == "lateral_drive"
            else "Mjlab-Velocity-Flat-Adaptive-Feedback-MicroDuck"
        )
        cfg.adaptive_command_exposure = True
        # This controller owns the full twist mixture. Keep a uniform nominal
        # pool and remove the competing standing-fraction schedule.
        cfg.curriculum.pop("standing_envs", None)
        command = microduck_mdp.AdaptiveVelocityCommandCfg(**vars(cfg.commands["twist"]))
        command.rel_standing_envs = 0.0
        command.rel_forward_envs = 0.0
        command.rel_turn_in_place_envs = 0.0
        command.rel_lateral_envs = 0.0
        command.rel_heading_envs = 0.0
        command.rel_world_envs = 0.0
        command.init_velocity_prob = 0.0
        cfg.commands["twist"] = command
        command.transition_probability = cfg.adaptive_transition_probability
        command.transition_bootstrap_mode = cfg.adaptive_transition_bootstrap_mode
        if cfg.adaptive_transition_acquisition:
            command.transition_duration_s = (1.0, 2.0)
            command.transition_forward_fraction = (0.25, 0.75)
        # The MDP functions self-negate; positive weights keep them penalties.
        cfg.rewards["linear_velocity_error_l1"] = RewardTermCfg(
            func=microduck_mdp.command_normalized_linear_velocity_l1,
            weight=cfg.adaptive_linear_feedback_weight,
            params={
                "command_name": "twist",
                "minimum_scale": FEEDBACK_LINEAR_MIN_SCALE_M_S,
                "deadband": FEEDBACK_LINEAR_DEADBAND_M_S,
                "tau_s": FEEDBACK_TRACKING_TAU_S,
                "yaw_deadband": FEEDBACK_YAW_DEADBAND_RAD_S,
            },
        )
        cfg.rewards["yaw_velocity_error_l1"] = RewardTermCfg(
            func=microduck_mdp.command_normalized_yaw_velocity_l1,
            weight=cfg.adaptive_yaw_feedback_weight,
            params={
                "command_name": "twist",
                "minimum_scale": FEEDBACK_YAW_MIN_SCALE_RAD_S,
                "deadband": FEEDBACK_YAW_DEADBAND_RAD_S,
                "tau_s": FEEDBACK_TRACKING_TAU_S,
            },
        )
    frontier_stall_override = os.environ.get("MICRODUCK_ADAPTIVE_FRONTIER_STALL_WINDOWS")
    if frontier_stall_override is not None:
        try:
            override = int(frontier_stall_override)
        except ValueError as exc:
            raise ValueError("adaptive frontier stall override must be an integer") from exc
        if override < 0:
            raise ValueError("adaptive frontier stall override must be nonnegative")
        cfg.adaptive_frontier_stall_windows = override
    if play:
        cfg.adaptive_evaluation_interval = 0
    return cfg


AdaptiveMicroduckRlCfg = deepcopy(MicroduckRlCfg)
AdaptiveMicroduckRlCfg.experiment_name = "velocity_adaptive"
AdaptiveMicroduckRlCfg.run_name = "velocity_adaptive"


def _adaptive_rl_cfg(name: str):
    cfg = deepcopy(AdaptiveMicroduckRlCfg)
    cfg.experiment_name = name
    cfg.run_name = name
    return cfg


AdaptiveMicroduckStaticRlCfg = _adaptive_rl_cfg("velocity_adaptive_static")
AdaptiveMicroduckComRlCfg = _adaptive_rl_cfg("velocity_adaptive_com")
AdaptiveMicroduckHeadComRlCfg = _adaptive_rl_cfg("velocity_adaptive_head_com")
AdaptiveMicroduckStandingRlCfg = _adaptive_rl_cfg("velocity_adaptive_standing_diagnostic")
AdaptiveMicroduckActionRateRlCfg = _adaptive_rl_cfg("velocity_adaptive_action_rate_diagnostic")
AdaptiveMicroduckLateralRlCfg = _adaptive_rl_cfg("velocity_adaptive_lateral_diagnostic")
AdaptiveMicroduckTrackingRlCfg = _adaptive_rl_cfg("velocity_adaptive_tracking_diagnostic")
AdaptiveMicroduckAcquisitionRlCfg = _adaptive_rl_cfg("velocity_adaptive_acquisition_diagnostic")
AdaptiveMicroduckAcquisitionLateralRlCfg = _adaptive_rl_cfg("velocity_adaptive_acquisition_lateral_diagnostic")
AdaptiveMicroduckAcquisitionFeedbackRlCfg = _adaptive_rl_cfg("velocity_adaptive_acquisition_feedback")
AdaptiveMicroduckLateralDriveRlCfg = _adaptive_rl_cfg("velocity_adaptive_lateral_drive")
AdaptiveMicroduckStrictificationRlCfg = _adaptive_rl_cfg("velocity_adaptive_strictification_diagnostic")
AdaptiveMicroduckPushRlCfg = _adaptive_rl_cfg("velocity_adaptive_push_diagnostic")

AdaptiveMicroduckFeedbackRlCfg = _adaptive_rl_cfg("velocity_adaptive_feedback")

# Historical recipes remain replayable, but all registrations share this table.
# Values are (axis mode, diagnostic recipe, feedback sampler, PPO log config).
ADAPTIVE_RECIPES = (
    ("all_static", None, False, AdaptiveMicroduckStaticRlCfg),
    ("com", None, False, AdaptiveMicroduckComRlCfg),
    ("head_com", None, False, AdaptiveMicroduckHeadComRlCfg),
    ("composed", None, False, AdaptiveMicroduckRlCfg),
    ("composed", "standing", False, AdaptiveMicroduckStandingRlCfg),
    ("composed", "action_rate", False, AdaptiveMicroduckActionRateRlCfg),
    ("composed", "lateral", False, AdaptiveMicroduckLateralRlCfg),
    ("all_static", "tracking", False, AdaptiveMicroduckTrackingRlCfg),
    ("all_static", "acquisition", False, AdaptiveMicroduckAcquisitionRlCfg),
    ("all_static", "acquisition_lateral", False, AdaptiveMicroduckAcquisitionLateralRlCfg),
    ("composed", "acquisition_feedback", True, AdaptiveMicroduckAcquisitionFeedbackRlCfg),
    ("composed", "lateral_drive", True, AdaptiveMicroduckLateralDriveRlCfg),
    ("all_static", "strictification", False, AdaptiveMicroduckStrictificationRlCfg),
    ("all_static", "push", False, AdaptiveMicroduckPushRlCfg),
    ("composed", None, True, AdaptiveMicroduckFeedbackRlCfg),
)
