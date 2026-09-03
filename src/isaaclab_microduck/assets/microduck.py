"""IsaacLab articulation contract for the converted Microduck walk asset."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_microduck.actuators import BamActuatorCfg

from .report import ACTUATED_ORDER, ROOT

USD_PATH = ROOT / ".cache/isaaclab-assets/microduck_walk.usd"

MICRODUCK_ACTUATOR_CFG = BamActuatorCfg(
    joint_names_expr=list(ACTUATED_ORDER),
    kp_fw=200.0,
    vin_range=(6.5, 8.2),
    effort_limit=1.0e9,
    effort_limit_sim=1.0e9,
    velocity_limit=100.0,
    velocity_limit_sim=100.0,
)

MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),
        joint_pos=dict.fromkeys(ACTUATED_ORDER, 0.0),
        joint_vel={".*": 0.0},
    ),
    actuators={"servos": MICRODUCK_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.9,
)


def require_converted_asset() -> Path:
    """Return the USD path or explain which conversion step is missing."""

    if not USD_PATH.is_file():
        raise FileNotFoundError(
            f"Microduck USD is missing at {USD_PATH}; run the validated MJCF conversion first."
        )
    return USD_PATH
