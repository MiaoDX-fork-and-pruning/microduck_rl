"""IsaacLab Velocity-Flat reset and domain-randomization adapters.

The mjlab recipe samples all mutable physics fields from their compile-time
defaults.  These adapters deliberately cache the first clean tensor and use
it on every invocation before applying a fresh sample; reset order therefore
cannot accumulate offsets or scales across episodes.

The functions are written against IsaacLab's duck-typed articulation API so
the pure sampling helpers can be tested without importing Isaac Sim.
"""

from __future__ import annotations

from typing import Any

import torch


def _device(env: Any, asset: Any | None = None) -> torch.device:
    return torch.device(getattr(env, "device", getattr(asset, "device", "cpu")))


def _ids(env: Any, env_ids: torch.Tensor | None) -> torch.Tensor:
    device = _device(env)
    if env_ids is None:
        return torch.arange(env.num_envs, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def sample_uniform(shape: tuple[int, ...], low: float, high: float, *, device=None) -> torch.Tensor:
    """Sample a fresh uniform tensor (kept as a testable reference primitive)."""

    return torch.empty(shape, device=device).uniform_(float(low), float(high))


def _data_tensor(data: Any, name: str) -> torch.Tensor:
    value = getattr(data, name)
    return getattr(value, "torch", value)


def _body_ids(asset: Any, asset_cfg: Any | None) -> torch.Tensor:
    ids = getattr(asset_cfg, "body_ids", slice(None)) if asset_cfg is not None else slice(None)
    if isinstance(ids, slice):
        return torch.arange(asset.num_bodies, device=asset.device, dtype=torch.long)[ids]
    return torch.as_tensor(ids, device=asset.device, dtype=torch.long)


def _joint_ids(asset: Any, asset_cfg: Any | None) -> torch.Tensor:
    ids = getattr(asset_cfg, "joint_ids", slice(None)) if asset_cfg is not None else slice(None)
    if isinstance(ids, slice):
        count = int(getattr(asset, "num_joints", _data_tensor(asset.data, "joint_pos").shape[-1]))
        return torch.arange(count, device=asset.device, dtype=torch.long)[ids]
    return torch.as_tensor(ids, device=asset.device, dtype=torch.long)


def reset_velocity_flat_state(
    env: Any,
    env_ids: torch.Tensor | None,
    *,
    z_range: tuple[float, float] = (0.12, 0.13),
    xy_range: tuple[float, float] = (-0.5, 0.5),
    yaw_range: tuple[float, float] = (-3.14, 3.14),
    asset_cfg: Any | None = None,
) -> None:
    """Reset root z and joints like mjlab's velocity recipe.

    The base pose is restored first, then x/y/yaw are sampled around the
    environment origin and z is sampled as an absolute height relative to it.
    Joint positions are reset to the articulation defaults (mjlab's
    ``position_range=(0, 0)``), not multiplied by a random scale. Velocities
    are reset to the default zero state.
    """

    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    ids = _ids(env, env_ids)
    n = ids.numel()
    device = _device(env, asset)
    default_pose = _data_tensor(asset.data, "default_root_pose")[ids].clone()
    origins = getattr(env.scene, "env_origins", None)
    if origins is None:
        origins = getattr(getattr(env.scene, "terrain", None), "env_origins", None)
    if origins is None:
        origins = torch.zeros((env.num_envs, 3), device=device)
    pose = default_pose
    pose[:, :3] = origins[ids].to(device) + default_pose[:, :3]
    pose[:, 0] += sample_uniform((n,), *xy_range, device=device)
    pose[:, 1] += sample_uniform((n,), *xy_range, device=device)
    pose[:, 2] = origins[ids, 2].to(device) + sample_uniform((n,), *z_range, device=device)
    yaw = sample_uniform((n,), *yaw_range, device=device)
    half_yaw = 0.5 * yaw
    yaw_quat = torch.zeros((n, 4), device=device, dtype=pose.dtype)
    # IsaacLab and USD use xyzw quaternions. Compose the authored default
    # orientation with the sampled world-yaw delta, matching mjlab's
    # ``quat_mul(default_root_state.quat, orientations_delta)``.
    yaw_quat[:, 2] = torch.sin(half_yaw)
    yaw_quat[:, 3] = torch.cos(half_yaw)
    base_quat = default_pose[:, 3:]
    yaw_sin, yaw_cos = yaw_quat[:, 2], yaw_quat[:, 3]
    pose[:, 3:] = torch.stack(
        (
            yaw_cos * base_quat[:, 0] + yaw_sin * base_quat[:, 1],
            -yaw_sin * base_quat[:, 0] + yaw_cos * base_quat[:, 1],
            yaw_sin * base_quat[:, 3] + yaw_cos * base_quat[:, 2],
            yaw_cos * base_quat[:, 3] - yaw_sin * base_quat[:, 2],
        ),
        dim=-1,
    )
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    default_vel = _data_tensor(asset.data, "default_root_vel")[ids]
    asset.write_root_velocity_to_sim_index(root_velocity=default_vel.clone(), env_ids=ids)

    joints = _joint_ids(asset, asset_cfg)
    default_q = _data_tensor(asset.data, "default_joint_pos")[ids[:, None], joints].clone()
    default_dq = _data_tensor(asset.data, "default_joint_vel")[ids[:, None], joints].clone()
    asset.write_joint_position_to_sim_index(position=default_q, joint_ids=joints, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=default_dq.clone(), joint_ids=joints, env_ids=ids)


def _cache(env: Any, name: str, value: torch.Tensor) -> torch.Tensor:
    key = f"_velocity_flat_default_{name}"
    if not hasattr(env, key):
        setattr(env, key, value.detach().clone())
    return getattr(env, key)


def randomize_com_offsets(
    env: Any,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: Any,
) -> None:
    """Apply fresh body-frame CoM offsets, restoring defaults first."""

    asset = env.scene[asset_cfg.name]
    ids = _ids(env, env_ids)
    bodies = _body_ids(asset, asset_cfg)
    default = _cache(env, "body_com_pose_b", _data_tensor(asset.data, "body_com_pose_b"))
    coms = default[ids[:, None], bodies].clone()
    if isinstance(ranges, dict):
        bounds = [ranges.get(k, (0.0, 0.0)) for k in ("x", "y", "z")]
    else:
        bounds = [(ranges[0], ranges[1])] * 3
    span = torch.tensor(bounds, device=asset.device, dtype=coms.dtype)
    coms[..., :3] += torch.empty((ids.numel(), bodies.numel(), 3), device=asset.device).uniform_(
        0.0, 1.0
    ) * (span[:, 1] - span[:, 0]) + span[:, 0]
    asset.set_coms_index(coms=coms, body_ids=bodies, env_ids=ids)


def randomize_mass_inertia(
    env: Any,
    env_ids: torch.Tensor | None,
    alpha_range: tuple[float, float],
    asset_cfg: Any,
) -> None:
    """Scale mass and inertia consistently using mjlab pseudo-inertia alpha."""

    asset = env.scene[asset_cfg.name]
    ids = _ids(env, env_ids)
    bodies = _body_ids(asset, asset_cfg)
    default_mass = _cache(env, "body_mass", _data_tensor(asset.data, "body_mass"))
    default_inertia = _cache(env, "body_inertia", _data_tensor(asset.data, "body_inertia"))
    alpha = torch.empty((ids.numel(), bodies.numel(), 1), device=asset.device).uniform_(*alpha_range)
    scale = torch.exp(2.0 * alpha)
    mass = default_mass[ids[:, None], bodies] * scale[..., 0]
    inertia = default_inertia[ids[:, None], bodies] * scale
    asset.set_masses_index(masses=mass, body_ids=bodies, env_ids=ids)
    asset.set_inertias_index(inertias=inertia, body_ids=bodies, env_ids=ids)


def randomize_armature(
    env: Any,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float] = (0.9, 1.1),
    asset_cfg: Any | None = None,
) -> None:
    """Scale joint armature from defaults (mjlab ``dr.joint_armature``)."""

    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    ids = _ids(env, env_ids)
    joints = _joint_ids(asset, asset_cfg)
    default = _cache(env, "joint_armature", _data_tensor(asset.data, "default_joint_armature"))
    values = default[ids[:, None], joints] * torch.empty(
        (ids.numel(), joints.numel()), device=asset.device
    ).uniform_(*ranges)
    writer = getattr(asset, "write_joint_armature_to_sim_index", None)
    if writer is None:
        raise AttributeError("IsaacLab articulation has no joint armature writer")
    writer(armature=values, joint_ids=joints, env_ids=ids)


def push_velocity(
    env: Any,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: Any | None = None,
) -> None:
    """mjlab-compatible interval push using root velocity writes."""

    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    ids = _ids(env, env_ids)
    current = _data_tensor(asset.data, "root_vel_w")[ids].clone()
    bounds = [velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    span = torch.tensor(bounds, device=asset.device, dtype=current.dtype)
    current += torch.empty_like(current).uniform_(0.0, 1.0) * (span[:, 1] - span[:, 0]) + span[:, 0]
    asset.write_root_velocity_to_sim_index(root_velocity=current, env_ids=ids)


def randomize_foot_material(
    env: Any,
    env_ids: torch.Tensor | None,
    *,
    static_friction_range: tuple[float, float] = (0.7, 1.3),
    dynamic_friction_range: tuple[float, float] = (0.7, 1.3),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    num_buckets: int = 64,
    asset_cfg: Any,
) -> None:
    """Use IsaacLab's material randomizer for the two foot collision geoms.

    OVPhysX 3.0 currently reports this operation as a no-op.  Keeping this
    adapter explicit means the ledger can distinguish that backend gap from a
    missing event in the task configuration.
    """

    # ``randomize_rigid_body_material`` is a ManagerTermBase class in
    # IsaacLab 3.0, rather than a plain function.  Build it once so its
    # backend-specific material buckets/defaults persist for the run, then
    # invoke the term with the same arguments the event manager would pass.
    term = getattr(env, "_velocity_flat_foot_material_term", None)
    if term is None:
        from isaaclab.envs import mdp as isaac_mdp
        from isaaclab.managers import EventTermCfg

        cfg = EventTermCfg(
            func=isaac_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets,
            },
        )
        term = isaac_mdp.randomize_rigid_body_material(cfg, env)
        env._velocity_flat_foot_material_term = term
    term(
        env,
        env_ids,
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        asset_cfg,
    )


__all__ = [
    "push_velocity",
    "randomize_armature",
    "randomize_com_offsets",
    "randomize_foot_material",
    "randomize_mass_inertia",
    "reset_velocity_flat_state",
    "sample_uniform",
]
