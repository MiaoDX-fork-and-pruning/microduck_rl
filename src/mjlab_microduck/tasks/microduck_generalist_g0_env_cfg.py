"""Training-only conditioned PPO environment for the bounded G0 experiment.

This composes the validated all-collisions VelStand recipe and appends the
frozen ten-dimensional conditioning block to its 61D specialist observation.
The production specialist ABI is untouched; this task is intentionally a
separate training task.
"""
from __future__ import annotations

import copy
import math
import torch
from dataclasses import dataclass
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab_microduck.tasks import mdp as _mdp
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

G0_BEHAVIORS = ("VELSTAND", "VELOCITY", "SITSTAND")
G0_OBS_DIM = 71
G0_ACTION_DIM = 14


@dataclass
class GeneralistG0HybridPpoAlgorithmCfg(PpoWithSymmetryCfg):
    anchor_weights: float = 0.1


@dataclass(frozen=True)
class G0TransitionContract:
    source: int
    destination: int
    dwell_s: float
    command: tuple[float, float, float]
    handoff_at_end: bool = False


# Exact G0 subset of docs/specialist_demo_scenario.json.  SITSTAND ->
# VELSTAND first issues the stand posture command to the sit/stand policy; only
# after its proven 6 s rise dwell does ownership pass back to VELSTAND.
G0_INITIAL_STAND_DWELL_S = 8.0
G0_TRANSITION_CONTRACTS = (
    G0TransitionContract(0, 1, 14.0, (0.15, 0.0, 0.0)),
    G0TransitionContract(1, 0, 8.0, (0.0, 0.0, 0.0)),
    G0TransitionContract(0, 2, 6.0, (1.0, 0.0, 0.0)),
    G0TransitionContract(2, 0, 6.0, (0.0, 0.0, 0.0), handoff_at_end=True),
)
_G0_CONTRACT_BY_EDGE = {(item.source, item.destination): item for item in G0_TRANSITION_CONTRACTS}
_G0_OUTGOING = {0: (1, 2), 1: (0,), 2: (0,)}


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
        # Reset coverage labels: 0=upright, 1=seated, 2=recovery/prone.
        env.g0_reset_bucket = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env.g0_reset_transition_phase = torch.zeros(env.num_envs, device=env.device)
        env.g0_transition_source = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env.g0_transition_destination = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
        env.g0_transition_elapsed_s = torch.zeros(env.num_envs, device=env.device)
        env.g0_transition_dwell_s = torch.full(
            (env.num_envs,), G0_INITIAL_STAND_DWELL_S, device=env.device
        )
    # Direct PPO must see every node before it can survive a full Track A
    # dwell. Starting every reset in VELSTAND starves VELOCITY/SITSTAND when a
    # fresh policy falls early; node sampling does not invent a graph edge.
    initial = torch.randint(len(G0_BEHAVIORS), (len(env_ids),), device=env.device)
    env.g0_behavior_id[env_ids] = initial
    env.g0_phase[env_ids] = 0
    env.g0_posture[env_ids] = 0
    env.g0_side[env_ids] = 0
    env.g0_reset_bucket[env_ids] = torch.where(
        initial == G0_BEHAVIORS.index("VELSTAND"),
        torch.full_like(initial, 2),
        torch.zeros_like(initial),
    )
    env.g0_reset_transition_phase[env_ids] = 0.0
    env.g0_transition_source[env_ids] = initial
    env.g0_transition_destination[env_ids] = -1
    env.g0_transition_elapsed_s[env_ids] = 0
    dwell = torch.tensor((G0_INITIAL_STAND_DWELL_S, 14.0, 6.0), device=env.device)
    env.g0_transition_dwell_s[env_ids] = dwell[initial]
    for behavior_id, command in enumerate(((0.0, 0.0, 0.0), (0.15, 0.0, 0.0))):
        selected = env_ids[initial == behavior_id]
        if len(selected):
            _write_g0_command(env, selected, command)
    sitstand_ids = env_ids[initial == G0_BEHAVIORS.index("SITSTAND")]
    if len(sitstand_ids):
        posture = torch.randint(2, (len(sitstand_ids),), device=env.device).to(torch.float32)
        env.g0_posture[sitstand_ids] = posture
        for target in (0, 1):
            selected = sitstand_ids[posture == target]
            if len(selected):
                _write_g0_command(env, selected, (float(target), 0.0, 0.0))
    return None


def reset_g0_sitstand_state(env, env_ids):
    """Apply the specialist's independent seated/standing reset buckets."""
    if not hasattr(env, "g0_behavior_id"):
        raise RuntimeError("g0_state must run before reset_g0_sitstand_state")
    sitstand_ids = env_ids[
        env.g0_behavior_id[env_ids] == G0_BEHAVIORS.index("SITSTAND")
    ]
    if not len(sitstand_ids):
        return None

    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES

    seated = torch.rand(len(sitstand_ids), device=env.device) < 0.5
    reset_params = {
        "face_down_prob": 0.0,
        "face_up_prob": 0.0,
        "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
        "sitting_joint_noise_std": 0.10,
        "sitting_tilt_max": math.radians(8.0),
        "sitting_z_min": 0.06,
        "sitting_z_max": 0.075,
        "standing_z_min": 0.11,
        "standing_z_max": 0.12,
    }
    if seated.any():
        env.g0_reset_bucket[sitstand_ids[seated]] = 1
        _mdp.set_random_ground_state(
            env,
            sitstand_ids[seated],
            sitting_prob=1.0,
            standing_prob=0.0,
            **reset_params,
        )
    if (~seated).any():
        env.g0_reset_bucket[sitstand_ids[~seated]] = 0
        _mdp.set_random_ground_state(
            env,
            sitstand_ids[~seated],
            sitting_prob=0.0,
            standing_prob=1.0,
            **reset_params,
        )
    return None


def _write_g0_command(env, env_ids, command: tuple[float, float, float]) -> None:
    """Make the frozen command authoritative over command-term resampling."""
    manager = getattr(env, "command_manager", None)
    if manager is None or not hasattr(manager, "get_term"):
        return
    term = manager.get_term("twist")
    buffer = getattr(term, "vel_command_b", None)
    if buffer is None:
        buffer = term.command
    buffer[env_ids, :3] = torch.as_tensor(command, device=buffer.device, dtype=buffer.dtype)
    world_buffer = getattr(term, "vel_command_w", None)
    if world_buffer is not None:
        world_buffer[env_ids, :3] = buffer[env_ids, :3]


def _activate_g0_edge(env, env_id: int, destination: int) -> None:
    source = int(env.g0_behavior_id[env_id])
    contract = _G0_CONTRACT_BY_EDGE[(source, destination)]
    env.g0_transition_source[env_id] = source
    env.g0_transition_destination[env_id] = destination
    env.g0_transition_elapsed_s[env_id] = 0.0
    env.g0_transition_dwell_s[env_id] = contract.dwell_s
    if not contract.handoff_at_end:
        env.g0_behavior_id[env_id] = destination
    env.g0_posture[env_id] = contract.command[0] if destination == 2 or source == 2 else 0.0
    _write_g0_command(env, torch.as_tensor([env_id], device=env.device), contract.command)


def sample_g0_transition(env, env_ids):
    """Advance Track A dwell timers and sample exclusively from its four edges."""
    if not hasattr(env, "g0_behavior_id"):
        initialize_g0_state(env, env_ids)
    dt = float(getattr(env, "step_dt", 0.02))
    env.g0_transition_elapsed_s[env_ids] += dt
    for eid in env_ids.tolist():
        destination = int(env.g0_transition_destination[eid])
        elapsed = float(env.g0_transition_elapsed_s[eid])
        dwell = float(env.g0_transition_dwell_s[eid])
        if destination >= 0:
            contract = _G0_CONTRACT_BY_EDGE[(int(env.g0_transition_source[eid]), destination)]
            _write_g0_command(env, torch.as_tensor([eid], device=env.device), contract.command)
            if elapsed < dwell:
                continue
            if contract.handoff_at_end:
                env.g0_behavior_id[eid] = destination
                env.g0_posture[eid] = 0.0
                env.g0_transition_destination[eid] = -1
                env.g0_transition_elapsed_s[eid] = 0.0
                env.g0_transition_dwell_s[eid] = G0_INITIAL_STAND_DWELL_S
                continue
        elif elapsed < dwell:
            _write_g0_command(env, torch.as_tensor([eid], device=env.device), (0.0, 0.0, 0.0))
            continue

        source = int(env.g0_behavior_id[eid])
        choices = _G0_OUTGOING[source]
        choice = int(torch.randint(len(choices), (1,), device=env.device))
        _activate_g0_edge(env, eid, choices[choice])

    active = env.g0_transition_destination[env_ids] >= 0
    progress = env.g0_transition_elapsed_s[env_ids] / env.g0_transition_dwell_s[env_ids].clamp_min(1e-6)
    env.g0_reset_transition_phase[env_ids] = progress.clamp(0.0, 1.0)
    env.g0_phase[env_ids, 0] = progress.clamp(0.0, 1.0)
    env.g0_phase[env_ids, 1] = active.to(env.g0_phase.dtype)
    return None


def make_microduck_generalist_g0_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velstand_env_cfg(play=play, rough=rough)
    # Reuse the validated commanded posture stack from SITSTAND. The
    # velstand template has no height/pose target for this behavior, so merely
    # masking its generic `pose` term leaves the sit task under-specified.
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg
    sit_cfg = make_microduck_sitstand_env_cfg(play=play, rough=rough)
    for name in ("posture_pose_legs", "posture_pose_l1", "posture_height", "posture_composite"):
        if name in sit_cfg.rewards:
            term = sit_cfg.rewards[name]
            cfg.rewards[f"sitstand_{name}"] = RewardTermCfg(
                func=_masked(term.func, "SITSTAND"), weight=term.weight, params=term.params
            )
    cfg.g0_behaviors = G0_BEHAVIORS
    cfg.g0_transition_graph = "docs/generalist_g0_transition_graph.json"
    cfg.g0_observation_dim = G0_OBS_DIM
    cfg.g0_action_dim = G0_ACTION_DIM
    cfg.events["g0_state"] = EventTermCfg(func=initialize_g0_state, mode="reset")
    # Dict insertion order is the reset order: route physical state only after
    # the behavior ID exists and after inherited base/joint/prone reset terms.
    cfg.events["g0_sitstand_state"] = EventTermCfg(
        func=reset_g0_sitstand_state, mode="reset"
    )
    cfg.events["g0_transition"] = EventTermCfg(
        func=sample_g0_transition, mode="interval", interval_range_s=(0.02, 0.02)
    )
    # Existing VelStand terms are retained but task-specific terms are active
    # only under their corresponding condition.
    pose_source = copy.copy(cfg.rewards.get("pose")) if "pose" in cfg.rewards else None
    for name, behavior in (("track_linear_velocity", "VELOCITY"), ("track_angular_velocity", "VELOCITY"),
                           ("pose", "VELOCITY"), ("upright_progress", "VELSTAND"),
                           ("height_progress", "VELSTAND"), ("recovery_success", "VELSTAND"),
                           ("fallen_tax", "VELSTAND")):
        if name in cfg.rewards:
            term = cfg.rewards[name]
            cfg.rewards[name] = RewardTermCfg(func=_masked(term.func, behavior), weight=term.weight, params=term.params)
    # Posture tracking is also the sit/stand task objective. Keep a separate
    # masked term so VELOCITY and SITSTAND each receive the same validated
    # pose-tracking signal without allowing either behavior to farm the other.
    if pose_source is not None:
        cfg.rewards["sitstand_pose"] = RewardTermCfg(
            func=_masked(pose_source.func, "SITSTAND"), weight=pose_source.weight, params=pose_source.params
        )
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        # The frozen G0 ABI is [48D proprio, 6D behavior, 13D commands,
        # phase/posture/side].  The velocity template appends command terms
        # after proprioception, so insert conditioning before those terms.
        command_terms = {
            name: terms.pop(name)
            for name in tuple(terms)
            if name in {"command", "velocity_commands", "twist", "head_command", "body_command"}
        }
        terms["g0_behavior"] = ObservationTermCfg(func=g0_behavior_one_hot)
        terms.update(command_terms)
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

# Keep baseline identities distinct in experiment tracking.  The environment
# and algorithm budget are intentionally identical; only initialization differs
# (the launcher below enforces that distinction at the command boundary).
GeneralistG0DirectPpoRlCfg = copy.deepcopy(GeneralistG0RlCfg)
GeneralistG0DirectPpoRlCfg.experiment_name = "generalist_g0_direct_ppo"
GeneralistG0DirectPpoRlCfg.run_name = "direct_ppo"

GeneralistG0HybridPpoRlCfg = copy.deepcopy(GeneralistG0RlCfg)
GeneralistG0HybridPpoRlCfg.experiment_name = "generalist_g0_hybrid_ppo"
GeneralistG0HybridPpoRlCfg.run_name = "hybrid_ppo"
GeneralistG0HybridPpoRlCfg.algorithm = GeneralistG0HybridPpoAlgorithmCfg(
    **vars(GeneralistG0HybridPpoRlCfg.algorithm)
)
GeneralistG0HybridPpoRlCfg.algorithm.class_name = "mjlab_microduck.generalist_hybrid_ppo.GeneralistHybridPPO"
