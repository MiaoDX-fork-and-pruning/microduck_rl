"""IsaacLab articulation contract for the converted Microduck walk asset."""

from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_microduck.actuators import BamActuatorCfg
from isaaclab_microduck.policy_abi import reorder_policy_joints

from .report import ACTUATED_ORDER, ROOT

USD_PATH = ROOT / ".cache/isaaclab-assets/microduck_walk.usd"

# mjlab's BAM ``edit_spec`` removes MuJoCo dof_frictionloss and supplies the
# friction budget from the actuator.  Keep these values explicit so the USD
# inspection and parity ledger cannot mistake the converter's nominal joint
# friction for the training dynamics.  Damping is the authored MJCF default.
MJLAB_BAM_JOINT_FRICTION = 0.0
# BamActuator.edit_spec zeros authored MJCF damping and frictionloss before
# each solve, then writes its own viscous/friction budget.  IsaacLab must not
# inherit the converter's 0.0048 joint friction as a second hidden budget.
MJLAB_JOINT_DAMPING = 0.0
MJLAB_BAM_ARMATURE = 0.0018077432831600838
MJLAB_BAM_VISCOUS_FRICTION = 0.005359668274599504
ASSET_DYNAMICS_CONTRACT = {
    "joint_friction": MJLAB_BAM_JOINT_FRICTION,
    "joint_damping": MJLAB_JOINT_DAMPING,
    "joint_armature": MJLAB_BAM_ARMATURE,
    "joint_viscous_friction": MJLAB_BAM_VISCOUS_FRICTION,
    "friction_bridge": "motor_only_external_effort_unavailable",
}

MICRODUCK_ACTUATOR_CFG = BamActuatorCfg(
    joint_names_expr=list(ACTUATED_ORDER),
    kp_fw=200.0,
    vin_range=(6.5, 8.2),
    effort_limit=1.0e9,
    effort_limit_sim=1.0e9,
    velocity_limit=100.0,
    velocity_limit_sim=100.0,
    # Match BAM's edit_spec/compute path.  Static and dynamic friction are
    # supplied by the explicit actuator bridge when load data is available;
    # the authored USD friction is deliberately overridden to zero.
    armature=MJLAB_BAM_ARMATURE,
    friction=0.0,
    dynamic_friction=0.0,
    viscous_friction=MJLAB_BAM_VISCOUS_FRICTION,
)

MICRODUCK_CFG = ArticulationCfg(
    # The imported USD authors its articulation root at Geometry/trunk_base;
    # make the path explicit because auto-discovery duplicates trunk_base in
    # Isaac Sim 6.0.1's PhysX tensor view.
    articulation_root_prim_path="/Geometry/trunk_base",
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
        # Matches scene_walk.xml INIT/STAND freejoint z; 0.18 leaves the feet
        # several centimetres above the plane and turns settle into free fall.
        pos=(0.0, 0.0, 0.12),
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


def policy_target_to_sim(target: torch.Tensor, sim_joint_names: list[str]) -> torch.Tensor:
    """Reorder a canonical 14D policy target into PhysX joint traversal order."""

    indices = reorder_policy_joints(torch.arange(target.shape[-1]), sim_joint_names)
    return target[..., torch.as_tensor(indices, device=target.device, dtype=torch.long)]
