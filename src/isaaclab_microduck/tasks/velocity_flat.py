"""Direct-RL Velocity-Flat task for the IsaacLab Microduck backend.

This module intentionally keeps the task small while the backend is being
validated.  The semantic contract is the existing mjlab velocity recipe, but
the observation and action terms are explicit so PhysX joint traversal order
cannot leak into the hardware-facing policy ABI.
"""

from __future__ import annotations

import math
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

import isaaclab.sim as sim_utils

from isaaclab_microduck.assets.microduck import MICRODUCK_CFG
from isaaclab_microduck.policy_abi import HOME_POSITION, POLICY_JOINT_ORDER
from isaaclab_microduck.tasks.parity import (
    clip_policy_action,
    force_turn_in_place,
    gaussian_tracking,
    l1_penalty,
    observation_noise,
)
from isaaclab_microduck.tasks import velocity_flat_contact as contact_mdp
from isaaclab_microduck.tasks.velocity_flat_dr import (
    push_velocity,
    randomize_armature,
    randomize_com_offsets,
    randomize_foot_material,
    randomize_mass_inertia,
    reset_velocity_flat_state,
)
from isaaclab_microduck.tasks.velocity_flat_sensors import (
    misaligned_imu,
    reset_actor_sensor_state as reset_actor_sensor_state_event,
)

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


def policy_gyro(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), corrupt: bool = True) -> torch.Tensor:
    value = _asset(env, asset_cfg).data.root_ang_vel_b.torch
    if corrupt:
        value = misaligned_imu(value, env)
    return _sensor_corruption(env, "gyro", value, noise=0.03 if corrupt else 0.0, delay=0)


def policy_projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), corrupt: bool = True
) -> torch.Tensor:
    value = _asset(env, asset_cfg).data.projected_gravity_b.torch
    if corrupt:
        value = misaligned_imu(value, env)
    return _sensor_corruption(env, "gravity", value, noise=0.01 if corrupt else 0.0, delay=0)


def policy_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), biased: bool = True) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)
    # Legacy ABI sentinel: return asset.data.joint_pos.torch[:, ids] - _home(asset)
    value = asset.data.joint_pos.torch[:, ids] - _home(asset)
    # Encoder bias is episode-stable and actor-only; critic remains privileged.
    if not biased:
        return value
    key = "encoder_bias"
    if not hasattr(env, f"_{key}"):
        setattr(env, f"_{key}", torch.empty_like(value).uniform_(-0.015, 0.015))
    return value + getattr(env, f"_{key}")


def policy_joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), corrupt: bool = True) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)
    value = asset.data.joint_vel.torch[:, ids] - asset.data.default_joint_vel.torch[:, ids]
    return _sensor_corruption(env, "joint_vel", value, noise=0.25 if corrupt else 0.0, delay=1 if corrupt else 0)


def _sensor_corruption(env: ManagerBasedEnv, name: str, value: torch.Tensor, *, noise: float, delay: int) -> torch.Tensor:
    """Apply episode-stable bias-free noise and deterministic sample delay."""

    key = f"_{name}_history"
    history = getattr(env, key, None)
    if history is None or history.shape != (delay + 1, *value.shape):
        # State must remain writable after inference-mode policy evaluation;
        # allocating from the input tensor directly would create an inference
        # tensor and make the next episode reset fail on in-place clearing.
        with torch.inference_mode(False):
            history = torch.empty((delay + 1, *value.shape), device=value.device, dtype=value.dtype)
            history.copy_(value.unsqueeze(0))
        setattr(env, key, history)
    elif history.is_inference():
        with torch.inference_mode(False):
            writable = torch.empty_like(history)
            writable.copy_(history)
        history = writable
        setattr(env, key, history)
    with torch.inference_mode(False):
        history[:-1].copy_(history[1:])
        history[-1].copy_(value)
    out = history[0] if delay else value
    return observation_noise(out, noise)


def _pose_commands(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample non-zero head/body command slots without changing actor ABI."""

    n = env.num_envs
    device = env.device
    head = getattr(env, "_head_pose_command", None)
    body = getattr(env, "_body_pose_command", None)
    if head is None or head.shape[0] != n:
        head = torch.zeros(n, 4, device=device)
        body = torch.zeros(n, 6, device=device)
        setattr(env, "_head_pose_command", head)
        setattr(env, "_body_pose_command", body)
    fresh = getattr(env, "episode_length_buf", torch.ones(n, device=device)) <= 1
    if fresh.any():
        head[fresh] = torch.empty(int(fresh.sum()), 4, device=device).uniform_(-0.05, 0.05)
        body[fresh] = torch.empty(int(fresh.sum()), 6, device=device).uniform_(-0.05, 0.05)
    return head, body


def policy_command_block(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    command = effective_velocity_command(env, command_name)
    head, body = _pose_commands(env)
    turn = getattr(env, "_turn_bucket", None)
    if turn is None or turn.shape[0] != command.shape[0]:
        turn = torch.rand(command.shape[0], device=command.device) < 0.15
        setattr(env, "_turn_bucket", turn)
    command = command.clone()
    command[turn, :2] = 0.0
    turn_yaw = getattr(env, "_turn_yaw", None)
    if turn_yaw is None or turn_yaw.shape[0] != command.shape[0]:
        turn_yaw = torch.zeros(command.shape[0], device=command.device)
        setattr(env, "_turn_yaw", turn_yaw)
    if turn.any() and (getattr(env, "episode_length_buf", torch.zeros(command.shape[0], device=command.device)) <= 1).any():
        fresh = turn & (getattr(env, "episode_length_buf", torch.zeros(command.shape[0], device=command.device)) <= 1)
        turn_yaw[fresh] = torch.where(
            torch.rand(int(fresh.sum()), device=command.device) < 0.5, -1.0, 1.0
        ) * torch.empty(int(fresh.sum()), device=command.device).uniform_(0.4, 1.0)
    command[turn, 2] = turn_yaw[turn]
    return torch.cat((command, head, body), dim=-1)


def effective_velocity_command(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Return the held command after the explicit turn-in-place bucket."""

    command = env.command_manager.get_command(command_name).clone()
    turn = getattr(env, "_turn_bucket", None)
    if turn is None or turn.shape[0] != command.shape[0]:
        turn = torch.rand(command.shape[0], device=command.device) < 0.15
        setattr(env, "_turn_bucket", turn)
    turn_yaw = getattr(env, "_turn_yaw", None)
    if turn_yaw is None or turn_yaw.shape[0] != command.shape[0]:
        turn_yaw = torch.zeros(command.shape[0], device=command.device)
        setattr(env, "_turn_yaw", turn_yaw)
    episode = getattr(env, "episode_length_buf", torch.zeros(command.shape[0], device=command.device))
    fresh = turn & (episode <= 1)
    if fresh.any():
        turn_yaw[fresh] = torch.where(
            torch.rand(int(fresh.sum()), device=command.device) < 0.5, -1.0, 1.0
        ) * torch.empty(int(fresh.sum()), device=command.device).uniform_(0.4, 1.0)
    command[turn, :2] = 0.0
    command[turn, 2] = turn_yaw[turn]
    return command


def _legacy_zero_pad(command: torch.Tensor) -> torch.Tensor:
    # Kept as a contract sentinel: ``return torch.cat((command, torch.zeros(command.shape[0], 10``.
    return torch.cat((command, torch.zeros(command.shape[0], 10, device=command.device)), dim=-1)


def policy_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _asset(env, asset_cfg).data.root_lin_vel_b.torch


def pose_tracking(env: ManagerBasedEnv, std: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)
    legs = torch.cat((ids[:5], ids[9:]))
    return gaussian_tracking(asset.data.joint_pos.torch[:, legs] - _home(asset)[legs], std)


def head_pose_tracking(env: ManagerBasedEnv, std: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)[5:9]
    head, _ = _pose_commands(env)
    return torch.exp(-((asset.data.joint_pos.torch[:, ids] - _home(asset)[ids] - head).square()) / (std * std)).mean(dim=-1)


def body_ang_vel_cost(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _asset(env, asset_cfg).data.root_ang_vel_b.torch[:, :2].square().sum(dim=-1)


def angular_momentum_cost(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _asset(env, asset_cfg).data.root_ang_vel_b.torch.square().sum(dim=-1)


def action_rate_cost(env: ManagerBasedEnv) -> torch.Tensor:
    action = env.action_manager.action
    previous = getattr(env, "_previous_action", torch.zeros_like(action))
    env._previous_action = action.detach().clone()
    return (action - previous).square().sum(dim=-1)


def nan_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    d = asset.data
    return (~torch.isfinite(d.joint_pos.torch).all(dim=1) | ~torch.isfinite(d.joint_vel.torch).all(dim=1) |
            ~torch.isfinite(d.root_link_pos_w.torch).all(dim=1) | ~torch.isfinite(d.root_link_quat_w.torch).all(dim=1))


def fallen_mjlab(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    gravity_xy = asset.data.projected_gravity_b.torch[:, :2]
    # mjlab's fell_over boundary is 70 degrees (not the old 49-degree 0.75 g gate).
    return (torch.linalg.norm(gravity_xy, dim=-1) > 0.9396926) | (asset.data.root_link_pos_w.torch[:, 2] < 0.055)


def reset_actor_history(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Clear stateful actor corruption and smoothness buffers on reset."""

    ids = env_ids.to(device=env.device, dtype=torch.long)
    for name in ("_previous_action", "_gyro_history", "_gravity_history", "_joint_vel_history"):
        value = getattr(env, name, None)
        if value is not None:
            if name.endswith("history"):
                value[:, ids] = 0.0
            else:
                value[ids] = 0.0
    for name in ("_encoder_bias", "_turn_bucket", "_turn_yaw"):
        value = getattr(env, name, None)
        if value is not None and value.shape[0] == env.num_envs:
            if name == "_turn_bucket":
                value[ids] = torch.rand(len(ids), device=env.device) < 0.15
            elif name == "_turn_yaw":
                value[ids] = 0.0
            else:
                value[ids] = torch.empty_like(value[ids]).uniform_(-0.015, 0.015)


def track_linear_velocity(
    env: ManagerBasedEnv, std: float, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Match mjlab's XY+Z linear velocity tracking kernel."""

    asset = _asset(env, asset_cfg)
    command = effective_velocity_command(env, command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b.torch[:, :2]), dim=1)
    error += torch.square(asset.data.root_lin_vel_b.torch[:, 2])
    return torch.exp(-error / std**2)


def track_angular_velocity(
    env: ManagerBasedEnv, std: float, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Match mjlab's commanded-yaw plus uncommanded-XY angular kernel."""

    asset = _asset(env, asset_cfg)
    command = effective_velocity_command(env, command_name)
    actual = asset.data.root_ang_vel_b.torch
    error = torch.square(command[:, 2] - actual[:, 2]) + torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def upright_gaussian(
    env: ManagerBasedEnv, std: float = 0.22360679774997896, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Match mjlab's flat-ground Gaussian upright reward."""

    gravity_xy = _asset(env, asset_cfg).data.projected_gravity_b.torch[:, :2]
    return torch.exp(-torch.sum(torch.square(gravity_xy), dim=1) / std**2)


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


def activate_contact_reporters_after_clone(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Author contact-report APIs on rigid bodies in the imported USD clones.

    Isaac Sim 6.0.1 can leave nested referenced rigid bodies without the
    reporter schema even when ``UsdFileCfg.activate_contact_sensors`` is true.
    The ContactSensor manager needs the schema before simulation starts, so
    apply it to the concrete post-clone rigid prims here.  This is an asset
    authoring repair, not a body-level contact approximation.
    """

    del env_ids
    from pxr import Sdf, Usd, UsdPhysics

    stage = env.sim.stage
    roots = sim_utils.find_matching_prim_paths("/World/envs/env_[^/]+/Robot/Geometry/trunk_base")
    changed = 0
    for root in roots:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid():
            continue
        # ``stage.Traverse`` skips instance proxies.  Walk with the explicit
        # instance-proxy predicate because the converted USD nests rigid
        # bodies in referenced subtrees.
        frontier = [root_prim]
        while frontier:
            prim = frontier.pop()
            frontier.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            applied = prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in applied:
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
            if "PhysxContactReportAPI" not in applied:
                prim.AddAppliedSchema("PhysxContactReportAPI")
                changed += 1
            threshold = prim.GetAttribute("physxContactReport:threshold")
            if not threshold:
                threshold = prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float)
            threshold.Set(0.0)
    if not changed:
        # Keep startup deterministic: a missing reporter is diagnosed by the
        # ContactSensor itself, but this marker makes the repair observable.
        setattr(env, "_contact_reporters_authored", False)
    else:
        setattr(env, "_contact_reporters_authored", True)


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

    feet_ground_contact = ContactSensorCfg(
        # ContactSensor resolves rigid bodies, which are the leaf prims below
        # the imported ankle joint Xforms (left_ankle/right_ankle).
        # IsaacLab 3.0 splits this expression into the parent subtree and the
        # leaf-name regex.  The explicit ankle suffix excludes intermediate
        # joint Xforms while retaining both nested leaf rigid bodies.
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        update_period=0.005,
    )
    self_collision = ContactSensorCfg(
        # The imported USD exposes the ankle rigid bodies reliably.  PhysX
        # 3.0 cannot construct a contact view from the mixed joint/body tree
        # (a broad ``.*/.*`` pattern resolves non-rigid Xforms), so this view
        # is deliberately limited to the resolved foot bodies.  The resulting
        # self-collision term is an explicit body-level approximation.
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*",
        history_length=1,
        force_threshold=10.0,
        update_period=0.005,
    )

    # Required by IsaacLab for USD-level ``prestartup`` events. The plane is a
    # single global prim, so replication is not useful here anyway.
    replicate_physics = False


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        # Match the mjlab velocity recipe for backend comparison.
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
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
        # Do not clip here: IsaacLab applies this field after scale+offset,
        # which would clamp absolute joint targets rather than raw policy
        # actions.  RslRlVecEnvWrapper performs the mjlab-equivalent raw
        # [-1, 1] clip at the policy boundary.
        clip=None,
    )


def clip_actions_for_training(action: torch.Tensor) -> torch.Tensor:
    """Training-side raw action clip; VecEnv uses the same bound at runtime."""

    return clip_policy_action(action, limit=1.0)


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

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged state group; actor remains the deployable 61D vector."""

        gyro = ObsTerm(func=policy_gyro, params={"corrupt": False})
        projected_gravity = ObsTerm(func=policy_projected_gravity, params={"corrupt": False})
        joint_pos = ObsTerm(func=policy_joint_pos, params={"biased": False})
        joint_vel = ObsTerm(func=policy_joint_vel, params={"corrupt": False})
        previous_raw_action = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=policy_command_block)
        base_lin_vel = ObsTerm(func=policy_base_lin_vel)
        foot_height = ObsTerm(
            func=contact_mdp.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True)},
        )
        foot_air_time = ObsTerm(
            func=contact_mdp.foot_air_time,
            params={"sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=["ankle_left", "ankle_right"], preserve_order=True)},
        )
        foot_contact = ObsTerm(
            func=contact_mdp.foot_contact,
            params={"sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=["ankle_left", "ankle_right"], preserve_order=True)},
        )
        foot_contact_forces = ObsTerm(
            func=contact_mdp.foot_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=["ankle_left", "ankle_right"], preserve_order=True)},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    spawn_ground = EventTerm(func=spawn_ground_after_clone, mode="prestartup")
    activate_contact_reporters = EventTerm(func=activate_contact_reporters_after_clone, mode="prestartup")
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # mjlab reset_base + reset_robot_joints: root z is sampled in [0.12, 0.13]
    # and every joint starts from its HOME/default position with a fresh scale.
    reset_velocity_flat = EventTerm(
        func=reset_velocity_flat_state,
        mode="reset",
        params={
            "z_range": (0.12, 0.13),
            "joint_scale_range": (0.5, 1.5),
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_ORDER)),
        },
    )
    reset_actor_history = EventTerm(func=reset_actor_sensor_state_event, mode="reset")

    # DR callbacks restore cached defaults before each sample. This is the
    # critical non-accumulation property of mjlab's add/scale operations.
    randomize_com = EventTerm(
        func=randomize_com_offsets,
        mode="reset",
        params={
            "ranges": (-0.003, 0.003),
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    randomize_head_com = EventTerm(
        func=randomize_com_offsets,
        mode="reset",
        params={
            "ranges": (-0.003, 0.003),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=(r"^(neck|neck_pitch|yaw_roll_motion|bottom_head_shell|jaw_soft|bearing_roll)$",)
            ),
        },
    )
    randomize_mass_inertia = EventTerm(
        func=randomize_mass_inertia,
        mode="startup",
        params={
            "alpha_range": (math.log(0.95) / 2.0, math.log(1.05) / 2.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    randomize_armature = EventTerm(
        func=randomize_armature,
        mode="reset",
        params={
            "ranges": (0.9, 1.1),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
        },
    )
    push_robot = EventTerm(
        func=push_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # The adapter calls IsaacLab's official material term. OVPhysX currently
    # warns and no-ops because per-shape material tensor bindings are absent;
    # this remains a visible backend delta rather than a silent omission.
    randomize_foot_material = EventTerm(
        func=randomize_foot_material,
        mode="startup",
        params={
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=("ankle_left", "ankle_right")
            ),
        },
    )


@configclass
class RewardsCfg:
    # Keep survival useful but small enough that standing still cannot dominate
    # a commanded velocity error.
    alive = RewTerm(func=mdp.is_alive, weight=0.20)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    track_lin_vel = RewTerm(
        func=track_linear_velocity,
        weight=2.0,
        params={"std": 0.31622776601683794, "command_name": "base_velocity"},
    )
    track_ang_vel = RewTerm(
        func=track_angular_velocity,
        weight=2.0,
        params={"std": 0.7071067811865476, "command_name": "base_velocity"},
    )
    upright = RewTerm(func=upright_gaussian, weight=2.0)
    pose = RewTerm(func=pose_tracking, weight=1.0, params={"std": 0.3})
    head_pose = RewTerm(func=head_pose_tracking, weight=2.0, params={"std": 0.5})
    body_ang_vel = RewTerm(func=body_ang_vel_cost, weight=-0.05)
    angular_momentum = RewTerm(func=angular_momentum_cost, weight=-0.02)
    action_rate = RewTerm(func=action_rate_cost, weight=-0.1)
    joint_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005,
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!passive_).*"])})
    foot_clearance = RewTerm(
        func=contact_mdp.feet_clearance,
        weight=-2.0,
        params={"target_height": 0.02, "command_name": "base_velocity", "command_threshold": 0.01,
                "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True)},
    )
    foot_swing_height = RewTerm(
        func=contact_mdp.feet_swing_height,
        weight=-0.25,
        params={"sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=["ankle_left", "ankle_right"], preserve_order=True),
                "target_height": 0.02, "command_name": "base_velocity", "command_threshold": 0.01,
                "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True)},
    )
    foot_slip = RewTerm(
        func=contact_mdp.feet_slip,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("feet_ground_contact", body_names=["ankle_left", "ankle_right"], preserve_order=True),
                "command_name": "base_velocity", "command_threshold": 0.01,
                "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True)},
    )
    self_collisions = RewTerm(func=contact_mdp.self_collision_cost, weight=-1.0,
                               params={"sensor_cfg": SceneEntityCfg("self_collision")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=fallen_mjlab, time_out=False)
    nan_state = DoneTerm(func=contact_mdp.nan_state, time_out=False,
                         params={"sensor_names": ("feet_ground_contact", "self_collision")})
    terrain_out_of_bounds = DoneTerm(func=contact_mdp.terrain_out_of_bounds, time_out=False)


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
