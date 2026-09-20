"""MJLab-only adaptive Velocity experiment configuration.

This factory is intentionally separate from the production fixed-schedule
Velocity recipe.  The capability controller lives in ``adaptive_curriculum``;
an evaluator or training adapter owns feeding it frozen battery metrics.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from mjlab_microduck.evaluation.capability import resolve_enabled_axes


# At a 0.12 m/s command the canonical std=sqrt(0.1) awards a stationary
# robot 86.6% of the tracking maximum. Test a sharper signal separately
# from command sampling; do not change the canonical reward or product gate.
DIAGNOSTIC_LINEAR_TRACKING_STD = 0.12


@dataclass(kw_only=True)
class AdaptiveVelocityEnvCfg(ManagerBasedRlEnvCfg):
    """Persist adaptive launch settings in MJLab's dataclass configuration."""

    task_id: str = "Mjlab-Velocity-Flat-Adaptive-MicroDuck"
    adaptive_axis_mode: str = "composed"
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
) -> ManagerBasedRlEnvCfg:
    """Return the static initial slice used by adaptive curriculum experiments."""

    base = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg = AdaptiveVelocityEnvCfg(**{field.name: getattr(base, field.name) for field in fields(base)})
    canonical_curriculum = deepcopy(cfg.curriculum)
    # Preserve canonical curricula unless the adaptive controller explicitly
    # owns that axis.  The controller mutates the live EventManager ranges for
    # the selected DR axes; all standing, smoothing, command, pose, terrain,
    # and reward schedules remain canonical and continue to run normally.
    enabled_axes = resolve_enabled_axes(axis_mode)
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
        if diagnostic_mode not in {"standing", "action_rate", "lateral", "tracking"}:
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
            cfg.task_id = "Mjlab-Velocity-Flat-Adaptive-Lateral-MicroDuck"
        else:
            cfg.rewards["track_linear_velocity"].params["std"] = DIAGNOSTIC_LINEAR_TRACKING_STD
            cfg.task_id = "Mjlab-Velocity-Flat-Adaptive-Tracking-MicroDuck"
        cfg.adaptive_axis_mode = "all_static"

    stage_file = os.environ.get("MICRODUCK_ADAPTIVE_STAGE_FILE")
    if stage_file:
        payload = json.loads(Path(stage_file).read_text(encoding="utf-8"))
        stages = payload.get("stages", payload)
        if not isinstance(stages, dict):
            raise ValueError("adaptive stage file must contain a stages object")
        for axis_name, stage_value in stages.items():
            if axis_name not in enabled_axes:
                raise ValueError(f"adaptive stage file axis {axis_name!r} is disabled by mode {axis_mode!r}")
            event_name = {"com_range": "randomize_com", "head_com_range": "randomize_head_com"}.get(axis_name)
            if event_name is None or event_name not in cfg.events:
                raise ValueError(f"adaptive stage file contains unknown axis: {axis_name}")
            cfg.events[event_name].params["ranges"] = (-float(stage_value), float(stage_value))
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
