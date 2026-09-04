"""Contact-backed Velocity-Flat MDP terms.

The mjlab recipe reads contact sensors and terrain-height sensors for all foot
terms.  IsaacLab's PhysX contact sensor is body based (USD shape filtering is
not available in this backend), so this module keeps the same tensors and
mathematical kernels while resolving the imported USD's ankle bodies.  The
functions deliberately return finite tensors: contact sensors can contain a
non-finite impulse for one frame after a solver failure, and ``nan_state`` then
terminates that environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:  # keep deterministic CPU tests importable outside the IsaacLab container
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers.manager_term_cfg import RewardTermCfg
except ModuleNotFoundError:  # pragma: no cover - exercised by host-side tests
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class SceneEntityCfg:  # type: ignore[no-redef]
        name: str
        body_names: str | list[str] | None = None
        body_ids: list[int] | slice = field(default_factory=lambda: slice(None))
        preserve_order: bool = False

    RewardTermCfg = Any  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _tensor(value: object) -> torch.Tensor:
    """Unwrap IsaacLab's backend proxy while also accepting plain tensors in tests."""

    return getattr(value, "torch", value)


def _scene_sensor(env: ManagerBasedEnv, cfg: SceneEntityCfg):
    return env.scene.sensors[cfg.name]


def _body_ids(cfg: SceneEntityCfg, count: int) -> torch.Tensor | slice:
    ids = cfg.body_ids
    if isinstance(ids, slice):
        return ids
    return torch.as_tensor(ids, dtype=torch.long)


def _command_active(env: ManagerBasedEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = _tensor(env.command_manager.get_command(command_name))
    speed = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return speed > threshold


def _contact_force(sensor, *, filtered: bool = False) -> torch.Tensor:
    data = sensor.data
    if filtered:
        matrix = getattr(data, "force_matrix_w", None)
        if matrix is not None:
            matrix = _tensor(matrix)
            # A filtered contact matrix has one axis per partner body.  Sum
            # those partners so the remaining shape matches net_forces_w.
            return matrix.sum(dim=-2)
    value = getattr(data, "net_forces_w", None)
    if value is None:
        value = getattr(data, "force", None)
    if value is None:
        raise RuntimeError("contact sensor does not expose net_forces_w/force")
    return _tensor(value)


def _contact_mask(sensor, *, threshold: float = 1.0) -> torch.Tensor:
    force = _contact_force(sensor)
    return torch.linalg.norm(force, dim=-1) > threshold


def _select(value: torch.Tensor, cfg: SceneEntityCfg) -> torch.Tensor:
    ids = _body_ids(cfg, value.shape[1])
    return value[:, ids]


def foot_contact_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
) -> torch.Tensor:
    """Privileged log-compressed per-foot contact force, matching mjlab."""

    forces = _select(_contact_force(_scene_sensor(env, sensor_cfg)), sensor_cfg)
    forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sign(forces).mul(torch.log1p(torch.abs(forces))).flatten(start_dim=1)


def foot_air_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
) -> torch.Tensor:
    """Privileged current air time for each selected foot."""

    value = getattr(_scene_sensor(env, sensor_cfg).data, "current_air_time", None)
    if value is None:
        raise RuntimeError("feet contact sensor must set track_air_time=True")
    return torch.nan_to_num(_select(_tensor(value).unsqueeze(-1), sensor_cfg).squeeze(-1), nan=0.0)


def foot_contact(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Privileged binary contact state for each selected foot."""

    return _select(_contact_mask(_scene_sensor(env, sensor_cfg), threshold=threshold), sensor_cfg).float()


def foot_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(".*ankle_left", ".*ankle_right")),
) -> torch.Tensor:
    """Foot clearance above the flat task plane in metres.

    The Velocity-Flat scene is an infinite z=0 plane.  Reading the resolved
    ankle body position is therefore equivalent to mjlab's terrain-height ray
    sensor and remains valid for every PhysX scene clone.
    """

    asset = env.scene[asset_cfg.name]
    pos = _tensor(asset.data.body_pos_w)
    value = _select(pos, asset_cfg)[..., 2]
    origins = _tensor(getattr(env.scene, "env_origins", torch.zeros_like(pos[:, 0])))
    if origins.ndim == 2:
        value = value - origins[:, 2].unsqueeze(-1)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold_min: float = 0.125,
    threshold_max: float = 0.300,
    command_name: str = "base_velocity",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Reward completed swing phases in the mjlab target time window."""

    sensor = _scene_sensor(env, sensor_cfg)
    first = _tensor(sensor.compute_first_contact(env.step_dt))
    air = getattr(sensor.data, "last_air_time", None)
    if air is None:
        raise RuntimeError("feet contact sensor must set track_air_time=True")
    first = _select(first, sensor_cfg)
    air = _select(_tensor(air), sensor_cfg)
    reward = ((air > threshold_min) & (air < threshold_max)).float() * first
    return reward.sum(dim=1) * _command_active(env, command_name, command_threshold).float()


def feet_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(".*ankle_left", ".*ankle_right")),
) -> torch.Tensor:
    """Velocity-weighted clearance error from mjlab's sensor-backed term."""

    heights = foot_height(env, asset_cfg)
    asset = env.scene[asset_cfg.name]
    velocity = _select(_tensor(asset.data.body_lin_vel_w), asset_cfg)[..., :2]
    cost = torch.abs(heights - target_height).mul(torch.linalg.norm(velocity, dim=-1)).sum(dim=1)
    return cost * _command_active(env, command_name, command_threshold).float()


class feet_swing_height:
    """Stateful peak swing-height error, charged exactly on first contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        self.peak_heights = torch.zeros((env.num_envs, 2), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(".*ankle_left", ".*ankle_right")),
    ) -> torch.Tensor:
        fresh = _tensor(getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))) <= 1
        if fresh.any():
            self.peak_heights[fresh] = 0.0
        sensor = _scene_sensor(env, sensor_cfg)
        heights = foot_height(env, asset_cfg)
        in_air = ~_select(_contact_mask(sensor), sensor_cfg)
        self.peak_heights = torch.where(in_air, torch.maximum(self.peak_heights, heights), self.peak_heights)
        first = _select(_tensor(sensor.compute_first_contact(env.step_dt)), sensor_cfg)
        error = torch.square(self.peak_heights / max(target_height, 1.0e-6) - 1.0)
        cost = (error * first.float()).sum(dim=1) * _command_active(env, command_name, command_threshold).float()
        self.peak_heights = torch.where(first, torch.zeros_like(self.peak_heights), self.peak_heights)
        return cost


def feet_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(".*ankle_left", ".*ankle_right")),
) -> torch.Tensor:
    """Penalize squared XY foot velocity while the foot is in contact."""

    sensor = _scene_sensor(env, sensor_cfg)
    contact = _select(_contact_mask(sensor), sensor_cfg).float()
    asset = env.scene[asset_cfg.name]
    vel = _select(_tensor(asset.data.body_lin_vel_w), asset_cfg)[..., :2]
    return torch.square(torch.linalg.norm(vel, dim=-1)).mul(contact).sum(dim=1) * _command_active(
        env, command_name, command_threshold
    ).float()


def self_collision_cost(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Count selected-body contact-force violations (the PhysX body sensor view)."""

    forces = _select(_contact_force(_scene_sensor(env, sensor_cfg), filtered=True), sensor_cfg)
    return (torch.linalg.norm(forces, dim=-1) > force_threshold).sum(dim=1).float()


def nan_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_names: tuple[str, ...] = ("feet_ground_contact",),
) -> torch.Tensor:
    """Terminate on non-finite articulation or contact-sensor state."""

    data = env.scene[asset_cfg.name].data
    bad = ~torch.isfinite(_tensor(data.joint_pos)).all(dim=1)
    bad |= ~torch.isfinite(_tensor(data.joint_vel)).all(dim=1)
    bad |= ~torch.isfinite(_tensor(data.root_pos_w)).all(dim=1)
    bad |= ~torch.isfinite(_tensor(data.root_quat_w)).all(dim=1)
    for name in sensor_names:
        sensor = env.scene.sensors.get(name)
        if sensor is None:
            continue
        force = getattr(sensor.data, "net_forces_w", None)
        if force is not None:
            bad |= ~torch.isfinite(_tensor(force)).all(dim=tuple(range(1, _tensor(force).ndim)))
        air = getattr(sensor.data, "current_air_time", None)
        if air is not None:
            bad |= ~torch.isfinite(_tensor(air)).all(dim=1)
    return bad


def terrain_out_of_bounds(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
) -> torch.Tensor:
    """Match IsaacLab's terrain-bound termination; infinite flat planes never terminate."""

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or getattr(terrain.cfg, "terrain_type", "plane") == "plane":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    gen = terrain.cfg.terrain_generator
    width, length = gen.size
    map_width = gen.num_rows * width + 2 * gen.border_width
    map_length = gen.num_cols * length + 2 * gen.border_width
    pos = _tensor(env.scene[asset_cfg.name].data.root_pos_w)
    return (torch.abs(pos[:, 0]) > 0.5 * map_width - distance_buffer) | (
        torch.abs(pos[:, 1]) > 0.5 * map_length - distance_buffer
    )


__all__ = [
    "feet_air_time",
    "feet_clearance",
    "feet_slip",
    "feet_swing_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
    "foot_height",
    "nan_state",
    "self_collision_cost",
    "terrain_out_of_bounds",
]
