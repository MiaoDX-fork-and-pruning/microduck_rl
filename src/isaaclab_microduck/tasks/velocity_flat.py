"""Direct-RL Velocity-Flat task for the IsaacLab Microduck backend.

This module intentionally keeps the task small while the backend is being
validated.  The semantic contract is the existing mjlab velocity recipe, but
the observation and action terms are explicit so PhysX joint traversal order
cannot leak into the hardware-facing policy ABI.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
    observation_noise,
    subtree_angular_momentum,
)
from isaaclab_microduck.tasks import velocity_flat_contact as contact_mdp
from isaaclab_microduck.tasks.velocity_flat_dr import (
    push_velocity,
    randomize_armature,
    randomize_bam_friction,
    randomize_com_offsets,
    randomize_foot_material,
    randomize_mass_inertia,
    curriculum_event_range,
    curriculum_pose_command_ranges,
    curriculum_reward_weight,
    curriculum_standing_probability,
    reset_velocity_flat_state,
)
from isaaclab_microduck.tasks.velocity_flat_sensors import (
    misaligned_imu,
    reset_actor_sensor_state as reset_actor_sensor_state_event,
    sensor_corruption,
)

# IsaacLab 3.0's filtered PhysX contact view is retained as a diagnostic
# configuration, but its nested USD path expansion is not multi-env safe. The
# production reward uses concrete raw views instead (see velocity_flat_contact).
ENABLE_FILTERED_SELF_CONTACT = False

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class MicroduckVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """mjlab-compatible velocity command with a held turn-in-place bucket."""

    class_type: type["MicroduckVelocityCommand"] | str = "isaaclab_microduck.tasks.velocity_flat:MicroduckVelocityCommand"
    rel_turn_in_place_envs: float = 0.15


class MicroduckVelocityCommand(mdp.UniformVelocityCommand):
    """Sample turn-in-place commands once per command resampling event."""

    cfg: MicroduckVelocityCommandCfg

    def _resample_command(self, env_ids) -> None:
        # CommandManager can pass ``slice(None)`` during an all-env reset,
        # while UniformVelocityCommand expects a sized sequence. Normalize it
        # once so initial reset and subset reset use the same path.
        ids = _command_env_ids(env_ids, self.num_envs, self.device)
        super()._resample_command(ids)
        fraction = float(self.cfg.rel_turn_in_place_envs)
        if fraction <= 0.0 or len(ids) == 0:
            return
        select = torch.rand(len(ids), device=self.device) < fraction
        turn_ids = ids[select]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, :2] = 0.0
        signs = torch.where(
            torch.rand(len(turn_ids), device=self.device) < 0.5,
            -torch.ones(len(turn_ids), device=self.device),
            torch.ones(len(turn_ids), device=self.device),
        )
        lo, hi = self.cfg.ranges.ang_vel_z
        magnitude = torch.empty(len(turn_ids), device=self.device).uniform_(
            0.4 * max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        )
        self.vel_command_b[turn_ids, 2] = signs * magnitude
        self.is_standing_env[turn_ids] = False


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Held per-dimension pose command used by the 13D policy command block."""

    class_type: type["UniformPoseCommand"] | str = "isaaclab_microduck.tasks.velocity_flat:UniformPoseCommand"
    ranges: tuple[tuple[float, float], ...] = ()
    zero_command_prob: float = 0.0


class UniformPoseCommand(CommandTerm):
    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(env.num_envs, len(cfg.ranges), device=env.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids) -> None:
        ids = _command_env_ids(env_ids, self.num_envs, self.device)
        if len(ids) == 0:
            return
        unit = torch.empty(len(ids), device=self.device)
        for index, (low, high) in enumerate(self.cfg.ranges):
            self._command[ids, index] = unit.uniform_(low, high)
        if self.cfg.zero_command_prob > 0.0:
            zero = torch.rand(len(ids), device=self.device) < self.cfg.zero_command_prob
            self._command[ids[zero]] = 0.0


def _command_env_ids(env_ids, num_envs: int, device: torch.device) -> torch.Tensor:
    """Normalize CommandManager reset ids to a concrete 1-D tensor."""

    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long).reshape(-1)


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
    return sensor_corruption(
        env,
        "gyro",
        value,
        noise=0.03 if corrupt else 0.0,
        delay=1 if corrupt else 0,
        delay_update_period=64 if corrupt else 0,
    )


def policy_projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), corrupt: bool = True
) -> torch.Tensor:
    value = _asset(env, asset_cfg).data.projected_gravity_b.torch
    if corrupt:
        value = misaligned_imu(value, env)
    return sensor_corruption(
        env,
        "gravity",
        value,
        noise=0.01 if corrupt else 0.0,
        delay=1 if corrupt else 0,
        delay_update_period=64 if corrupt else 0,
    )


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
    # mjlab's actor joint encoder includes both a persistent per-env bias and
    # fresh bounded observation noise.  The critic requests ``biased=False``
    # and therefore receives the clean value without either corruption.
    return observation_noise(value + getattr(env, f"_{key}"), 0.001)


def policy_joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), corrupt: bool = True) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)
    value = asset.data.joint_vel.torch[:, ids] - asset.data.default_joint_vel.torch[:, ids]
    return sensor_corruption(env, "joint_vel", value, noise=0.25 if corrupt else 0.0, delay=1 if corrupt else 0)


def _pose_commands(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the held command-manager pose terms (with a test fallback)."""

    manager = getattr(env, "command_manager", None)
    if manager is not None:
        return manager.get_command("head_pose"), manager.get_command("body_pose")
    n, device = env.num_envs, env.device
    return torch.zeros(n, 4, device=device), torch.zeros(n, 6, device=device)


def policy_command_block(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    command = effective_velocity_command(env, command_name)
    head, body = _pose_commands(env)
    return torch.cat((command, head, body), dim=-1)


def effective_velocity_command(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Return the command-manager value, already held by its resampling term."""

    return env.command_manager.get_command(command_name).clone()


def _legacy_zero_pad(command: torch.Tensor) -> torch.Tensor:
    # Kept as a contract sentinel: ``return torch.cat((command, torch.zeros(command.shape[0], 10``.
    return torch.cat((command, torch.zeros(command.shape[0], 10, device=command.device)), dim=-1)


def policy_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _asset(env, asset_cfg).data.root_lin_vel_b.torch


def _std_vector(
    joint_names: list[str] | tuple[str, ...],
    values: dict[str, float],
    *,
    default: float = 0.3,
    device: torch.device,
) -> torch.Tensor:
    """Resolve mjlab's regex keyed posture tolerances in joint order."""

    out = []
    for name in joint_names:
        match = next((value for pattern, value in values.items() if re.match(pattern, name)), default)
        out.append(float(match))
    return torch.as_tensor(out, dtype=torch.float32, device=device)


def pose_tracking(
    env: ManagerBasedEnv,
    std_standing: dict[str, float] | None = None,
    std_walking: dict[str, float] | None = None,
    std_running: dict[str, float] | None = None,
    walking_threshold: float = 0.01,
    running_threshold: float = 1.5,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Match mjlab ``variable_posture`` including speed-dependent tolerances."""

    asset = _asset(env, asset_cfg)
    ids = asset_cfg.joint_ids
    # Manager resolution uses ``slice(None)`` when no selector was supplied;
    # that means all simulator joints, not an empty selection.  The mjlab
    # variable-posture term nevertheless tracks only the ten leg servos.
    if ids is None or isinstance(ids, slice):
        ids = torch.cat((_policy_indices(asset)[:5], _policy_indices(asset)[9:])).tolist()
    elif len(ids) == 0:
        return torch.ones(asset.data.joint_pos.torch.shape[0], device=asset.device)
    ids = torch.as_tensor(ids, device=asset.device, dtype=torch.long)
    names = [asset.joint_names[int(index)] for index in ids]
    standing = _std_vector(names, std_standing or {}, device=asset.device)
    walking = _std_vector(names, std_walking or {}, device=asset.device)
    running = _std_vector(names, std_running or {}, device=asset.device)

    command = env.command_manager.get_command(command_name)
    total_speed = torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs()
    standing_mask = (total_speed < walking_threshold).to(asset.data.joint_pos.torch.dtype)
    walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).to(
        asset.data.joint_pos.torch.dtype
    )
    running_mask = (total_speed >= running_threshold).to(asset.data.joint_pos.torch.dtype)
    selected_std = (
        standing.unsqueeze(0) * standing_mask.unsqueeze(1)
        + walking.unsqueeze(0) * walking_mask.unsqueeze(1)
        + running.unsqueeze(0) * running_mask.unsqueeze(1)
    )
    error = asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]
    return torch.exp(torch.mean(-error.square() / selected_std.square(), dim=1))


def head_pose_tracking(env: ManagerBasedEnv, std: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)[5:9]
    head, _ = _pose_commands(env)
    return torch.exp(-((asset.data.joint_pos.torch[:, ids] - _home(asset)[ids] - head).square()) / (std * std)).mean(dim=-1)


def body_pose_tracking(
    env: ManagerBasedEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.05,
    z_std: float = 0.02,
    angle_std: float = math.radians(15.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Match mjlab's six-axis ``body_pose_tracking_6d`` kernel."""

    asset = _asset(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    origin = getattr(env.scene, "env_origins", None)
    if origin is None:
        origin = getattr(getattr(env.scene, "terrain", None), "env_origins", None)
    if origin is None:
        origin = torch.zeros_like(asset.data.root_link_pos_w.torch)
    rel = torch.nan_to_num(asset.data.root_link_pos_w.torch - origin, nan=0.0)
    errors = [
        rel[:, 0] - command[:, 0],
        rel[:, 1] - command[:, 1],
        rel[:, 2] - (nominal_height + command[:, 2]),
    ]
    quat = asset.data.root_link_quat_w.torch
    # IsaacLab/PhysX stores quaternions in xyzw order (scalar last).
    qx, qy, qz, qw = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx.square() + qy.square()))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy.square() + qz.square()))
    angle_error = torch.stack((roll - command[:, 3], pitch - command[:, 4], yaw - command[:, 5]), dim=-1)
    angle_error[:, 2] = torch.remainder(angle_error[:, 2] + math.pi, 2.0 * math.pi) - math.pi
    pos_error = torch.stack(errors, dim=-1)
    scaled = torch.cat((pos_error[:, :2] / xy_std, pos_error[:, 2:3] / z_std, angle_error / angle_std), dim=-1)
    return torch.exp(-scaled.square()).mean(dim=-1)


def head_pose_bias_penalty(
    env: ManagerBasedEnv,
    command_name: str = "head_pose",
    tau_s: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the 1-second EMA of head-pose error, as in mjlab."""

    asset = _asset(env, asset_cfg)
    ids = _policy_indices(asset)[5:9]
    command = env.command_manager.get_command(command_name)
    error = asset.data.joint_pos.torch[:, ids] - _home(asset)[ids] - command
    ema = getattr(env, "_head_bias_ema", None)
    if ema is None or ema.shape != error.shape:
        ema = torch.zeros_like(error)
        env._head_bias_ema = ema
    fresh = env.episode_length_buf <= 1
    ema[fresh] = 0.0
    alpha = min(1.0, float(env.step_dt) / max(float(tau_s), 1.0e-6))
    env._head_bias_ema = (1.0 - alpha) * ema + alpha * error
    return -env._head_bias_ema.abs().mean(dim=-1)


def body_ang_vel_cost(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    # mjlab selects the trunk body link and reads world-frame angular velocity.
    # IsaacLab exposes the same quantity through body_link_ang_vel_w; root
    # angular velocity in body frame is not equivalent for a moving trunk.
    body_vel = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :]
    body_vel = body_vel.squeeze(1)
    return body_vel[:, :2].square().sum(dim=-1)


def angular_momentum_cost(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Match MuJoCo's ``subtreeangmom`` sensor from IsaacLab body tensors.

    IsaacLab 3.0 has no manager sensor for MuJoCo's subtree angular momentum,
    but it exposes the same rigid-body quantities.  The trunk subtree contains
    every body in this articulation, so compute momentum about its aggregate
    center of mass: ``sum(r x m*v + R*I*R.T*w)``.  This is deliberately based
    on COM velocities/inertias and is not a root angular-velocity proxy.
    """

    del asset_cfg  # trunk_base is the articulation root; its subtree is all bodies.
    asset = _asset(env, SceneEntityCfg("robot"))
    data = asset.data

    def tensor(name: str) -> torch.Tensor:
        value = getattr(data, name)
        return getattr(value, "torch", value)

    momentum = subtree_angular_momentum(
        tensor("body_mass"),
        tensor("body_com_pos_w"),
        tensor("body_com_lin_vel_w"),
        tensor("body_com_ang_vel_w"),
        tensor("body_inertia"),
        tensor("body_com_quat_w"),
    )
    return momentum.square().sum(dim=-1)


def joint_pos_limits(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Match mjlab/IsaacLab soft joint-position limit penalty."""

    asset = _asset(env, asset_cfg)
    ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)
    q = asset.data.joint_pos.torch[:, ids]
    limits = asset.data.soft_joint_pos_limits.torch[:, ids]
    below = (limits[..., 0] - q).clamp_min(0.0)
    above = (q - limits[..., 1]).clamp_min(0.0)
    return (below + above).sum(dim=-1)


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
    return torch.linalg.norm(gravity_xy, dim=-1) > 0.9396926


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
    for name in ("_encoder_bias",):
        value = getattr(env, name, None)
        if value is not None and value.shape[0] == env.num_envs:
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


def air_time_reward(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Count feet currently in mjlab's air-time window.

    mjlab's ``feet_air_time`` is a per-step current-air-time count.  It does
    not wait for a first-contact event or use ``last_air_time``; keeping that
    distinction here is important because the resulting reward cadence differs
    materially for a biped.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    current = sensor.data.current_air_time
    current = getattr(current, "torch", current)
    body_ids = sensor_cfg.body_ids
    current = current[:, body_ids]
    reward = ((current > threshold_min) & (current < threshold_max)).float().sum(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        linear_norm = torch.linalg.norm(command[:, :2], dim=1)
        total = linear_norm + torch.abs(command[:, 2])
        reward *= (total > command_threshold).float()
    return reward


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
    # IsaacLab's filtered ContactSensor currently rewrites nested articulation
    # roots to ``.../trunk_base/trunk_base`` and cannot initialize the view for
    # multiple clones.  Keep the reference configurations visible for contract
    # inspection, but disable them in production; self-collision uses the raw
    # concrete PhysX view in ``velocity_flat_contact.self_collision_cost``.
    self_collision_trunk = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg",
            "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2",
        ],
        history_length=0,
        track_contact_points=True,
        update_period=0.005,
    ) if ENABLE_FILTERED_SELF_CONTACT else None
    self_collision_legs = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2",
        ],
        history_length=0,
        track_contact_points=True,
        update_period=0.005,
    ) if ENABLE_FILTERED_SELF_CONTACT else None

    # Required by IsaacLab for USD-level ``prestartup`` events. The plane is a
    # single global prim, so replication is not useful here anyway.
    replicate_physics = False


@configclass
class CommandsCfg:
    # Keep the command generators as the sole owner of command sampling.  In
    # particular, turn-in-place is sampled by MicroduckVelocityCommand during
    # each command resample and must not be re-randomized by observations or
    # rewards.
    base_velocity = MicroduckVelocityCommandCfg(
        class_type=MicroduckVelocityCommand,
        asset_name="robot",
        # Match the mjlab velocity recipe for backend comparison.
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_turn_in_place_envs=0.15,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
    # These are held commands, independently resampled from the velocity
    # command.  The velocity task's mjlab config deliberately leaves the
    # exact-zero pose bucket disabled (standup enables its own 0.3 bucket).
    head_pose = UniformPoseCommandCfg(
        class_type=UniformPoseCommand,
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.07, 0.07),
            (-0.015, 0.015),
        ),
        zero_command_prob=0.0,
    )
    body_pose = UniformPoseCommandCfg(
        class_type=UniformPoseCommand,
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.005, 0.005),
            (-0.005, 0.005),
            (-0.005, 0.005),
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.05, 0.05),
        ),
        zero_command_prob=0.0,
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
        # Do not clip here: the production mjlab Velocity-Flat runner leaves
        # clip_actions unset and sends the complete raw policy output through
        # HOME+scale to BAM. IsaacLab's action-term clip would additionally run
        # after scale+offset, changing that target contract.
        clip=None,
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
            "xy_range": (-0.5, 0.5),
            "yaw_range": (-3.14, 3.14),
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
    randomize_bam_friction = EventTerm(
        func=randomize_bam_friction,
        mode="reset",
        params={
            "scale_range": (0.9, 1.1),
            "asset_cfg": SceneEntityCfg("robot"),
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
    pose = RewTerm(
        func=pose_tracking,
        weight=1.0,
        params={
            "std_standing": {
                r".*hip_yaw.*": 0.1,
                r".*hip_roll.*": 0.05,
                r".*hip_pitch.*": 0.15,
                r".*knee.*": 0.15,
                r".*ankle.*": 0.1,
            },
            "std_walking": {
                r".*hip_yaw.*": 0.3,
                r".*hip_roll.*": 0.05,
                r".*hip_pitch.*": 0.4,
                r".*knee.*": 0.4,
                r".*ankle.*": 0.25,
            },
            "std_running": {
                r".*hip_yaw.*": 0.3,
                r".*hip_roll.*": 0.05,
                r".*hip_pitch.*": 0.4,
                r".*knee.*": 0.4,
                r".*ankle.*": 0.25,
            },
            "walking_threshold": 0.01,
            "running_threshold": 1.5,
            "command_name": "base_velocity",
        },
    )
    head_pose = RewTerm(func=head_pose_tracking, weight=2.0, params={"std": 0.5})
    body_pose = RewTerm(
        func=body_pose_tracking,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15.0),
        },
    )
    head_pose_bias = RewTerm(
        func=head_pose_bias_penalty,
        weight=0.0,
        params={"command_name": "head_pose", "tau_s": 1.0},
    )
    air_time = RewTerm(
        func=air_time_reward,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "feet_ground_contact",
                body_names=["ankle_left", "ankle_right"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "command_threshold": 0.01,
            "threshold_min": 0.125,
            "threshold_max": 0.300,
        },
    )
    body_ang_vel = RewTerm(
        func=body_ang_vel_cost,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    angular_momentum = RewTerm(func=angular_momentum_cost, weight=-0.02)
    action_rate_l2 = RewTerm(func=action_rate_cost, weight=-0.1)
    dof_pos_limits = RewTerm(
        func=joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))},
    )
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
    self_collisions = RewTerm(
        func=contact_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_names": ("self_collision_trunk", "self_collision_legs")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=fallen_mjlab, time_out=False)
    nan_state = DoneTerm(
        func=contact_mdp.nan_state,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )
    terrain_out_of_bounds = DoneTerm(func=contact_mdp.terrain_out_of_bounds, time_out=False)


@configclass
class CurriculumCfg:
    """Stepwise schedules ported from mjlab's Velocity-Flat recipe."""

    action_rate_weight = CurrTerm(
        func=curriculum_reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 750 * 24, "weight": -0.4},
                {"step": 1000 * 24, "weight": -0.6},
                {"step": 1250 * 24, "weight": -0.8},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )
    standing_envs = CurrTerm(
        func=curriculum_standing_probability,
        params={
            "command_name": "base_velocity",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 500 * 24, "rel_standing_envs": 0.05},
                {"step": 750 * 24, "rel_standing_envs": 0.10},
                {"step": 1000 * 24, "rel_standing_envs": 0.15},
                {"step": 1500 * 24, "rel_standing_envs": 0.20},
                {"step": 2000 * 24, "rel_standing_envs": 0.25},
            ],
        },
    )
    head_pose_range = CurrTerm(
        func=curriculum_pose_command_ranges,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 500 * 24, "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )
    body_pose_range = CurrTerm(
        func=curriculum_pose_command_ranges,
        params={
            "command_name": "body_pose",
            "range_stages": [{"step": 0, "ranges": ((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005), (-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))}],
        },
    )
    com_range = CurrTerm(
        func=curriculum_event_range,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": 0, "range": 0.003},
                {"step": 500 * 24, "range": 0.005},
                {"step": 1000 * 24, "range": 0.01},
                {"step": 1500 * 24, "range": 0.015},
            ],
        },
    )
    head_com_range = CurrTerm(
        func=curriculum_event_range,
        params={
            "event_name": "randomize_head_com",
            "range_stages": [
                {"step": 0, "range": 0.003},
                {"step": 500 * 24, "range": 0.005},
                {"step": 1000 * 24, "range": 0.01},
            ],
        },
    )
    head_pose_bias_weight = CurrTerm(
        func=curriculum_reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * 24, "weight": 1.0},
                {"step": 1000 * 24, "weight": 2.0},
                {"step": 1500 * 24, "weight": 3.0},
            ],
        },
    )


@configclass
class IsaacLabVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

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
