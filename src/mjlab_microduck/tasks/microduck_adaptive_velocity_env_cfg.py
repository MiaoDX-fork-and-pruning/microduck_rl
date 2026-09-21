"""MJLab-only adaptive Velocity experiment configuration.

This factory is intentionally separate from the production fixed-schedule
Velocity recipe.  The capability controller lives in ``adaptive_curriculum``;
an evaluator or training adapter owns feeding it frozen battery metrics.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg

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


DIAGNOSTIC_NAMES = {
    "standing": "Standing", "action_rate": "ActionRate", "lateral": "Lateral",
    "tracking": "Tracking", "acquisition": "Acquisition",
    "acquisition_lateral": "AcquisitionLateral", "push": "Push",
}


@dataclass(kw_only=True)
class AdaptiveVelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Persist adaptive launch settings in MJLab's dataclass configuration."""

    task_id: str = "Mjlab-Velocity-Flat-Adaptive-MicroDuck"
    adaptive_axis_mode: str = "composed"
    adaptive_command_exposure: bool = False
    adaptive_evaluation_interval: int = 0
    adaptive_evaluation_seed: int = 20260916
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
    cfg.adaptive_evaluator_schema_version = 2
    cfg.adaptive_seed_set_id = os.environ.get("MICRODUCK_ADAPTIVE_SEED_SET_ID", "adaptive-gate-20260916")
    cfg.adaptive_evaluation_timeout_s = 900
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
            elif diagnostic_mode in ("acquisition", "acquisition_lateral"):
                cfg.commands["twist"].rel_lateral_envs = 0.20 if diagnostic_mode == "acquisition" else 0.50
                cfg.curriculum = {
                    "tracking_std": CurriculumTermCfg(
                        func=microduck_mdp.velocity_tracking_std_curriculum,
                        params={
                            "reward_name": "track_linear_velocity",
                            "std_stages": list(ACQUISITION_TRACKING_STD_STAGES),
                        },
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
        cfg.adaptive_axis_mode = "all_static"

    if command_exposure:
        if diagnostic_mode is not None or axis_mode != "composed":
            raise ValueError("feedback exposure requires the composed recipe without diagnostics")
        cfg.task_id = "Mjlab-Velocity-Flat-Adaptive-Feedback-MicroDuck"
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
    ("all_static", "push", False, AdaptiveMicroduckPushRlCfg),
    ("composed", None, True, AdaptiveMicroduckFeedbackRlCfg),
)
