"""MJLab-only adaptive Velocity experiment configuration.

This factory is intentionally separate from the production fixed-schedule
Velocity recipe.  The capability controller lives in ``adaptive_curriculum``;
an evaluator or training adapter owns feeding it frozen battery metrics.
"""

from copy import deepcopy
import json
import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from mjlab_microduck.evaluation.capability import resolve_enabled_axes


def make_microduck_adaptive_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
    axis_mode: str = "composed",
    diagnostic_mode: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Return the static initial slice used by adaptive curriculum experiments."""

    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    canonical_curriculum = deepcopy(cfg.curriculum)
    # The canonical factory owns the initial command/DR/reward values. Remove
    # only its wall-clock curriculum terms; the adaptive adapter applies live
    # manager changes after a frozen evaluation window.
    for name in list(cfg.curriculum):
        del cfg.curriculum[name]
    resolve_enabled_axes(axis_mode)
    # This metadata is consumed by AdaptiveMicroduckOnPolicyRunner. It is kept
    # on the env config so each CloudML branch has an explicit axis contract.
    cfg.adaptive_axis_mode = axis_mode
    if diagnostic_mode is not None:
        if diagnostic_mode not in {"standing", "action_rate"}:
            raise ValueError(f"unsupported adaptive diagnostic mode: {diagnostic_mode}")
        if diagnostic_mode == "standing":
            cfg.curriculum = {"standing_envs": canonical_curriculum["standing_envs"]}
        else:
            cfg.curriculum = {"action_rate_weight": canonical_curriculum["action_rate_weight"]}
        cfg.adaptive_axis_mode = "all_static"

    stage_file = os.environ.get("MICRODUCK_ADAPTIVE_STAGE_FILE")
    if stage_file:
        payload = json.loads(Path(stage_file).read_text(encoding="utf-8"))
        stages = payload.get("stages", payload)
        if not isinstance(stages, dict):
            raise ValueError("adaptive stage file must contain a stages object")
        for axis_name, stage_value in stages.items():
            if axis_name not in resolve_enabled_axes(axis_mode):
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
