"""Contact-backed Velocity-Flat MDP terms.

The walk asset has exactly one collidable shape on each ankle body, so PhysX's
body-level ankle contact view is equivalent to mjlab's named foot-geom view.
Foot height and velocity are evaluated at the original MJCF sites, not at the
ankle body origins. Self-collision uses separate one-to-many filtered views for
the three bodies carrying ``self_collision_only`` shapes, which excludes ground
contacts and avoids PhysX's unsupported many-to-many filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:  # keep deterministic CPU tests importable outside the IsaacLab container
    from isaaclab.managers import ManagerTermBase, SceneEntityCfg
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

    class ManagerTermBase:  # type: ignore[no-redef]
        """Small host-test fallback for IsaacLab's stateful term base."""

        def __init__(self, cfg: Any, env: Any) -> None:
            self.cfg = cfg
            self._env = env

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


FOOT_SITE_OFFSETS_B = (
    (0.0, -0.0238146, -0.0140852),
    (0.0, -0.0238146, -0.0140852),
)


def _tensor(value: object) -> torch.Tensor:
    """Unwrap IsaacLab's backend proxy while also accepting plain tensors in tests."""

    if isinstance(value, torch.Tensor):
        return value
    proxy_tensor = getattr(value, "torch", None)
    if proxy_tensor is not None:
        return proxy_tensor
    import warp as wp

    return wp.to_torch(value)


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


def _quat_apply_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    imaginary = quat[..., :3]
    real = quat[..., 3:4]
    return vector + 2.0 * (
        real * torch.linalg.cross(imaginary, vector)
        + torch.linalg.cross(imaginary, torch.linalg.cross(imaginary, vector))
    )


def _foot_site_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    site_offsets_b: tuple[tuple[float, float, float], ...] = FOOT_SITE_OFFSETS_B,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world position and linear velocity at the canonical MJCF foot sites."""

    asset = env.scene[asset_cfg.name]
    body_pos = _select(_tensor(asset.data.body_pos_w), asset_cfg)
    body_quat = _select(_tensor(asset.data.body_quat_w), asset_cfg)
    body_lin_vel = _select(_tensor(asset.data.body_lin_vel_w), asset_cfg)
    body_ang_vel = _select(_tensor(asset.data.body_ang_vel_w), asset_cfg)
    offsets = torch.as_tensor(site_offsets_b, dtype=body_pos.dtype, device=body_pos.device)
    if offsets.shape != body_pos.shape[1:]:
        raise ValueError(f"expected one foot-site offset per selected body, got {offsets.shape}")
    offset_w = _quat_apply_xyzw(body_quat, offsets.unsqueeze(0).expand_as(body_pos))
    site_pos = body_pos + offset_w
    site_lin_vel = body_lin_vel + torch.linalg.cross(body_ang_vel, offset_w)
    return site_pos, site_lin_vel


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

    site_pos, _ = _foot_site_state(env, asset_cfg)
    value = site_pos[..., 2]
    origins = _tensor(getattr(env.scene, "env_origins", torch.zeros_like(site_pos[:, 0])))
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
    first = _tensor(sensor.compute_first_contact(env.step_dt)).to(dtype=torch.bool)
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
    _, velocity = _foot_site_state(env, asset_cfg)
    velocity = velocity[..., :2]
    cost = torch.abs(heights - target_height).mul(torch.linalg.norm(velocity, dim=-1)).sum(dim=1)
    return cost * _command_active(env, command_name, command_threshold).float()


class feet_swing_height(ManagerTermBase):
    """Stateful peak swing-height error, charged exactly on first contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
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
        first = _select(_tensor(sensor.compute_first_contact(env.step_dt)).to(dtype=torch.bool), sensor_cfg)
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
    _, vel = _foot_site_state(env, asset_cfg)
    vel = vel[..., :2]
    return torch.square(torch.linalg.norm(vel, dim=-1)).mul(contact).sum(dim=1) * _command_active(
        env, command_name, command_threshold
    ).float()


def self_collision_cost(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
) -> torch.Tensor:
    """Count self-contact points without including external contacts.

    The normal path uses the two filtered ContactSensors retained by the host
    tests.  IsaacLab 3.0 cannot construct those filtered views for this nested
    USD articulation across multiple clones, so production scenes leave them
    disabled and lazily create concrete per-environment ``RigidContactView``
    objects from the already-resolved foot sensor body paths.  This keeps the
    count source at PhysX contact-point granularity and excludes ground by
    construction.
    """

    available = getattr(env.scene, "sensors", {})
    if all(name in available for name in sensor_names):
        return _filtered_sensor_self_collision_cost(env, sensor_names)
    return _raw_physx_self_collision_cost(env)


def _filtered_sensor_self_collision_cost(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
) -> torch.Tensor:
    """Compatibility path for explicit filtered ContactSensor instances."""

    cost = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        matrix = getattr(sensor.data, "force_matrix_w", None)
        if matrix is None:
            raise RuntimeError(f"self-collision sensor {name!r} must use filtered partner paths")
        counts = getattr(sensor, "_contact_counts", None)
        if counts is None:
            raise RuntimeError(f"self-collision sensor {name!r} must track contact points")
        counts = _tensor(counts)
        if counts.numel() % env.num_envs != 0:
            raise RuntimeError(f"unexpected self-collision count shape for {name!r}: {counts.shape}")
        cost += counts.reshape(env.num_envs, -1).sum(dim=1).float()
    return cost


def _concrete_env_path(path: str, env_name: str) -> str:
    """Replace the standard IsaacLab environment glob with one concrete env."""

    marker = "/World/envs/env_*/"
    if marker in path:
        return path.replace(marker, f"/World/envs/{env_name}/", 1)
    return path


def _raw_physx_self_collision_cost(env: ManagerBasedEnv) -> torch.Tensor:
    """Read multi-env raw PhysX contact counts for trunk/leg self contacts.

    The tensor API accepts concrete paths with one sensor body per environment
    and one-to-many filters.  Passing the paths explicitly avoids its nested
    USD body-name expansion (which otherwise produces ``body/body`` paths).
    """

    cache = getattr(env, "_velocity_flat_raw_self_contact_views", None)
    if cache is None:
        sensors = getattr(env.scene, "sensors", {})
        foot_sensor = sensors.get("feet_ground_contact")
        if foot_sensor is None or getattr(foot_sensor, "body_physx_view", None) is None:
            raise RuntimeError("raw self-collision view requires feet_ground_contact body paths")
        sim_view = getattr(foot_sensor, "_physics_sim_view", None)
        if sim_view is None:
            raise RuntimeError("raw self-collision view requires an initialized PhysX simulation view")
        body_paths = [str(path) for path in foot_sensor.body_physx_view.prim_paths]
        selected: dict[str, list[str]] = {"trunk": [], "left_leg": [], "right_leg": []}
        for path in body_paths:
            if path.endswith("/trunk_base"):
                selected["trunk"].append(path)
            elif path.endswith("/upper_leg_left/leg"):
                selected["left_leg"].append(path)
            elif path.endswith("/upper_leg_right/leg_2"):
                selected["right_leg"].append(path)
        if any(len(paths) != env.num_envs for paths in selected.values()):
            raise RuntimeError(
                "raw self-collision body paths did not resolve one trunk and two legs per environment: "
                f"{ {name: len(paths) for name, paths in selected.items()} }"
            )
        trunk_filters = [
            [
                _concrete_env_path(selected["left_leg"][env_id], selected["trunk"][env_id].split("/")[3]),
                _concrete_env_path(selected["right_leg"][env_id], selected["trunk"][env_id].split("/")[3]),
            ]
            for env_id in range(env.num_envs)
        ]
        leg_filters = [
            [_concrete_env_path(selected["right_leg"][env_id], selected["left_leg"][env_id].split("/")[3])]
            for env_id in range(env.num_envs)
        ]
        try:
            trunk_view = sim_view.create_rigid_contact_view(
                selected["trunk"],
                filter_patterns=trunk_filters,
                max_contact_data_count=64 * env.num_envs,
            )
            leg_view = sim_view.create_rigid_contact_view(
                selected["left_leg"],
                filter_patterns=leg_filters,
                max_contact_data_count=32 * env.num_envs,
            )
        except Exception as exc:
            raise RuntimeError("failed to initialize concrete raw self-collision PhysX views") from exc
        cache = (trunk_view, leg_view)
        setattr(env, "_velocity_flat_raw_self_contact_views", cache)

    counts_total = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    sim_cfg = getattr(getattr(env, "sim", None), "cfg", None)
    dt = float(getattr(sim_cfg, "dt", getattr(env, "step_dt", 0.005)))
    for view in cache:
        _, _, _, _, counts, _ = view.get_contact_data(dt=dt)
        counts_total += torch.as_tensor(counts, device=env.device).reshape(env.num_envs, -1).sum(dim=1).float()
    # Keep the production observation available to deterministic probes after
    # ManagerBasedEnv.step().  Isaac Sim 6.0.1 may invalidate a direct tensor
    # view read after the manager has completed its step, while the same read
    # during reward evaluation is stable.
    setattr(env, "_velocity_flat_last_raw_self_contact_counts", counts_total.detach().clone())
    return counts_total


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
