"""MJLab-only adaptive Velocity experiment configuration.

This factory is intentionally separate from the production fixed-schedule
Velocity recipe.  The capability controller lives in ``adaptive_curriculum``;
an evaluator or training adapter owns feeding it frozen battery metrics.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg


def make_microduck_adaptive_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
    axis_mode: str = "composed",
) -> ManagerBasedRlEnvCfg:
    """Return the static initial slice used by adaptive curriculum experiments."""

    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    # The canonical factory owns the initial command/DR/reward values. Remove
    # only its wall-clock curriculum terms; the adaptive adapter applies live
    # manager changes after a frozen evaluation window.
    for name in list(cfg.curriculum):
        del cfg.curriculum[name]
    if axis_mode not in {"all_static", "com", "head_com", "composed"}:
        raise ValueError(f"unsupported adaptive axis mode: {axis_mode}")
    # This metadata is consumed by AdaptiveMicroduckOnPolicyRunner. It is kept
    # on the env config so each CloudML branch has an explicit axis contract.
    cfg.adaptive_axis_mode = axis_mode
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
