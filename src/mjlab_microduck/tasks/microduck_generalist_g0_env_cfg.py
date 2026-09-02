"""Training-only conditioned PPO environment for the bounded G0 experiment.

This composes the validated all-collisions VelStand recipe and appends the
frozen ten-dimensional conditioning block to its 61D specialist observation.
The production specialist ABI is untouched; this task is intentionally a
separate training task.
"""
from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab_microduck.tasks import mdp as _mdp
from mjlab_microduck.generalist_transition_graph import LEGAL_EDGES, route_transition
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

G0_BEHAVIORS = ("VELSTAND", "VELOCITY", "SITSTAND")
G0_OBS_DIM = 71
G0_ACTION_DIM = 14


def _behavior_id(env) -> torch.Tensor:
    value = getattr(env, "g0_behavior_id", None)
    if value is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    return value.to(device=env.device, dtype=torch.long)


def g0_behavior_one_hot(env) -> torch.Tensor:
    return torch.nn.functional.one_hot(_behavior_id(env), num_classes=6).to(torch.float32)


def g0_phase(env) -> torch.Tensor:
    value = getattr(env, "g0_phase", None)
    if value is None:
        return torch.zeros((env.num_envs, 2), device=env.device)
    return value.to(device=env.device, dtype=torch.float32)


def g0_posture(env) -> torch.Tensor:
    value = getattr(env, "g0_posture", None)
    if value is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    return value.to(device=env.device, dtype=torch.float32).reshape(-1, 1)


def g0_side(env) -> torch.Tensor:
    value = getattr(env, "g0_side", None)
    if value is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    return value.to(device=env.device, dtype=torch.float32).reshape(-1, 1)


def behavior_mask(env, behavior: str) -> torch.Tensor:
    """Per-environment mask for behavior-conditioned reward terms."""
    if behavior not in G0_BEHAVIORS:
        raise ValueError(f"unknown G0 behavior {behavior!r}")
    return (_behavior_id(env) == G0_BEHAVIORS.index(behavior)).to(torch.float32)


def _masked(func, behavior):
    # Manager reward terms may be callable classes with constructor state;
    # wrapping those changes their invocation contract. Leave such terms in
    # their proven form and mask only ordinary per-step functions.
    if isinstance(func, type):
        return func
    def wrapped(env, **params):
        return func(env, **params) * behavior_mask(env, behavior)
    wrapped.__name__ = f"g0_{behavior.lower()}_{func.__name__}"
    return wrapped


def initialize_g0_state(env, env_ids):
    if not hasattr(env, "g0_behavior_id"):
        env.g0_behavior_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env.g0_phase = torch.zeros((env.num_envs, 2), device=env.device)
        env.g0_posture = torch.zeros(env.num_envs, device=env.device)
        env.g0_side = torch.zeros(env.num_envs, device=env.device)
    env.g0_behavior_id[env_ids] = 0
    env.g0_phase[env_ids] = 0
    env.g0_posture[env_ids] = 0
    env.g0_side[env_ids] = 0
    return None


def sample_g0_transition(env, env_ids):
    """Sample only frozen graph edges; callers may override destination explicitly."""
    if not hasattr(env, "g0_behavior_id"):
        initialize_g0_state(env, env_ids)
    current = env.g0_behavior_id[env_ids]
    choices = {0: (1, 2), 1: (0,), 2: (0,)}
    for row, eid in enumerate(env_ids.tolist()):
        destination = choices[int(current[row])][int(torch.randint(len(choices[int(current[row])]), (1,), device=env.device))]
        env.g0_behavior_id[eid] = destination
    return None


def make_microduck_generalist_g0_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velstand_env_cfg(play=play, rough=rough)
    cfg.g0_behaviors = G0_BEHAVIORS
    cfg.g0_transition_graph = "docs/generalist_g0_transition_graph.json"
    cfg.g0_observation_dim = G0_OBS_DIM
    cfg.g0_action_dim = G0_ACTION_DIM
    cfg.events["g0_state"] = EventTermCfg(func=initialize_g0_state, mode="reset")
    cfg.events["g0_transition"] = EventTermCfg(
        func=sample_g0_transition, mode="interval", interval_range_s=(4.0, 8.0)
    )
    # Existing VelStand terms are retained but task-specific terms are active
    # only under their corresponding condition.
    for name, behavior in (("track_linear_velocity", "VELOCITY"), ("track_angular_velocity", "VELOCITY"),
                           ("pose", "VELOCITY"), ("upright_progress", "VELSTAND"),
                           ("height_progress", "VELSTAND"), ("recovery_success", "VELSTAND"),
                           ("fallen_tax", "VELSTAND")):
        if name in cfg.rewards:
            term = cfg.rewards[name]
            cfg.rewards[name] = RewardTermCfg(func=_masked(term.func, behavior), weight=term.weight, params=term.params)
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["g0_behavior"] = ObservationTermCfg(func=g0_behavior_one_hot)
        terms["g0_phase"] = ObservationTermCfg(func=g0_phase)
        terms["g0_posture"] = ObservationTermCfg(func=g0_posture)
        terms["g0_side"] = ObservationTermCfg(func=g0_side)
    return cfg


GeneralistG0RlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="generalist_g0",
    run_name="generalist_g0",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
