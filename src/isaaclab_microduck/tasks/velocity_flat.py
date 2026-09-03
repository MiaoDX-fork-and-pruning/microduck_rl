"""Direct-RL Velocity-Flat task for the IsaacLab Microduck backend.

This module intentionally keeps the task small while the backend is being
validated.  The semantic contract is the existing mjlab velocity recipe, but
the observation and action terms are explicit so PhysX joint traversal order
cannot leak into the hardware-facing policy ABI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

import isaaclab.sim as sim_utils

from isaaclab_microduck.assets.microduck import MICRODUCK_CFG
from isaaclab_microduck.policy_abi import HOME_POSITION, POLICY_JOINT_ORDER

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _policy_indices(asset) -> torch.Tensor:
    """Return simulator joint indices in canonical policy order."""

    by_name = {name: i for i, name in enumerate(asset.joint_names)}
    try:
        return torch.as_tensor(
            [by_name[name] for name in POLICY_JOINT_ORDER],
            device=asset.device,
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(f"Microduck task is missing policy joint {exc.args[0]!r}") from exc


def _home(asset) -> torch.Tensor:
    return torch.as_tensor(HOME_POSITION, device=asset.device, dtype=torch.float32)


def _asset(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def policy_gyro(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _asset(env, asset_cfg).data.root_ang_vel_b.torch


def policy_projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return _asset(env, asset_cfg).data.projected_gravity_b.torch


def policy_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)
    return asset.data.joint_pos.torch[:, ids] - _home(asset)


def policy_joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)
    return asset.data.joint_vel.torch[:, ids] - asset.data.default_joint_vel.torch[:, ids]


def policy_command_block(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.cat((command, torch.zeros(command.shape[0], 10, device=command.device)), dim=-1)


def fallen(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    gravity_xy = asset.data.projected_gravity_b.torch[:, :2]
    height = asset.data.root_link_pos_w.torch[:, 2]
    return (torch.linalg.norm(gravity_xy, dim=-1) > 0.75) | (height < 0.055)


def spawn_ground_after_clone(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Create the flat plane after USD cloning and before PhysX reset.

    Isaac Sim 6.0.1 has a scene-construction failure when a ground-plane
    entity is cloned alongside this imported articulation.  ``prestartup`` is
    the framework-supported point after ``InteractiveScene`` cloning and
    before the first ``SimulationContext.reset``.
    """

    del env, env_ids
    cfg = sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    cfg.func("/World/ground", cfg)


def _task_robot_cfg() -> ArticulationCfg:
    """Copy the shared asset and use canonical HOME only for this task."""

    home = {name: float(value) for name, value in zip(POLICY_JOINT_ORDER, HOME_POSITION)}
    # The converted USD has a tighter right-hip-yaw upper limit (0.436 rad)
    # than the canonical hardware HOME (0.4579 rad). IsaacLab validates
    # initial state against authored limits, so only the simulator spawn value
    # is clamped; policy observations and action offsets stay canonical.
    home["right_hip_yaw"] = 0.436
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12),
        joint_pos=home,
        joint_vel={".*": 0.0},
    )
    return MICRODUCK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=init_state,
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    """The imported Microduck articulation.

    The Isaac Sim 6.0.1 runtime currently exits during manager-scene
    construction when a ground-plane entity is cloned alongside this imported
    USD.  The smoke harness adds the plane after cloning, matching the proven
    physics battery workaround.
    """

    robot: ArticulationCfg = _task_robot_cfg()

    # Required by IsaacLab for USD-level ``prestartup`` events. The plane is a
    # single global prim, so replication is not useful here anyway.
    replicate_physics = False


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 4.0),
        rel_standing_envs=0.10,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(POLICY_JOINT_ORDER),
        preserve_order=True,
        scale=1.0,
        offset={name: float(value) for name, value in zip(POLICY_JOINT_ORDER, HOME_POSITION)},
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        gyro = ObsTerm(func=policy_gyro)
        projected_gravity = ObsTerm(func=policy_projected_gravity)
        joint_pos = ObsTerm(func=policy_joint_pos)
        joint_vel = ObsTerm(func=policy_joint_vel)
        previous_raw_action = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=policy_command_block)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    spawn_ground = EventTerm(func=spawn_ground_after_clone, mode="prestartup")
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    # Keep survival useful but small enough that standing still cannot dominate
    # a commanded velocity error.
    alive = RewTerm(func=mdp.is_alive, weight=0.20)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.5,
        params={"std": 0.25, "command_name": "base_velocity"},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"std": 0.50, "command_name": "base_velocity"},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!passive_).*"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=fallen, time_out=False)


@configclass
class IsaacLabVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.seed = 7
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


@configclass
class IsaacLabVelocityFlatEnvCfg_PLAY(IsaacLabVelocityFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False


def make_velocity_flat_env_cfg(*, play: bool = False, num_envs: int = 1):
    cfg = IsaacLabVelocityFlatEnvCfg_PLAY() if play else IsaacLabVelocityFlatEnvCfg()
    cfg.scene.num_envs = num_envs
    return cfg


__all__ = [
    "IsaacLabVelocityFlatEnvCfg",
    "IsaacLabVelocityFlatEnvCfg_PLAY",
    "make_velocity_flat_env_cfg",
    "policy_command_block",
    "policy_gyro",
    "policy_joint_pos",
    "policy_joint_vel",
    "policy_projected_gravity",
]
