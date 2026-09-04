from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "src/isaaclab_microduck/tasks/velocity_flat.py"
SMOKE = ROOT / "scripts/isaaclab/velocity_flat_smoke.py"


def _source(path: Path) -> str:
    return path.read_text()


def test_velocity_flat_declares_shared_policy_contract() -> None:
    source = _source(TASK)
    assert "class IsaacLabVelocityFlatEnvCfg(ManagerBasedRLEnvCfg)" in source
    assert "joint_names=list(POLICY_JOINT_ORDER)" in source
    assert "preserve_order=True" in source
    assert "return torch.cat((command, torch.zeros(command.shape[0], 10" in source
    assert "return asset.data.joint_pos.torch[:, ids] - _home(asset)" in source
    assert "def track_linear_velocity(" in source
    assert "def track_angular_velocity(" in source
    assert "def upright_gaussian(" in source


def test_velocity_flat_command_profile_matches_mjlab_recipe() -> None:
    source = _source(TASK)
    assert "resampling_time_range=(3.0, 8.0)" in source
    assert "rel_standing_envs=0.02" in source
    assert "lin_vel_x=(-0.4, 0.4)" in source
    assert "lin_vel_y=(-0.3, 0.3)" in source
    assert "ang_vel_z=(-1.0, 1.0)" in source


def test_velocity_flat_uses_bam_asset_and_home_override() -> None:
    source = _source(TASK)
    assert "from isaaclab_microduck.assets.microduck import MICRODUCK_CFG" in source
    assert "return MICRODUCK_CFG.replace(" in source
    assert "joint_pos=home" in source
    assert 'home["right_hip_yaw"] = 0.436' in source
    assert "policy observations and action offsets stay canonical" in source
    assert "offset={name: float(value)" in source


def test_velocity_flat_keeps_raw_clip_at_rl_wrapper_boundary() -> None:
    source = _source(TASK)
    assert "clip=None" in source
    assert "def clip_actions_for_training(" in source
    assert "return clip_policy_action(action, limit=1.0)" in source


def test_velocity_flat_smoke_checks_observation_and_action_dimensions() -> None:
    tree = ast.parse(_source(SMOKE))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert {"OBSERVATION_SIZE", "ACTION_SIZE"} <= names
    source = _source(SMOKE)
    assert '"IsaacLab-Velocity-Flat-MicroDuck"' in source
    assert "torch.isfinite" in source
    assert "uniform_(-1.0, 1.0)" in source
    assert "ground_event_ready" in source
    assert "contact_reporter_count" in source
    assert "contact_body_count" in source
    assert "contact_forces_finite" in source
    assert "contact_body_names" in source


def test_ground_is_injected_at_the_prestartup_hook() -> None:
    source = _source(TASK)
    assert "def spawn_ground_after_clone" in source
    assert 'mode="prestartup"' in source
    assert 'cfg.func("/World/ground", cfg)' in source
    assert "def activate_contact_reporters_after_clone" in source
    assert "activate_contact_reporters = EventTerm" in source


def test_lazy_task_registry_points_at_velocity_cfg() -> None:
    source = _source(ROOT / "src/isaaclab_microduck/tasks/__init__.py")
    assert "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatEnvCfg" in source
    assert "rsl_rl_cfg_entry_point" in source


def test_ppo_runner_matches_mjlab_budget_and_policy_shape() -> None:
    source = _source(ROOT / "src/isaaclab_microduck/tasks/agents/rsl_rl_ppo_cfg.py")
    assert "MicroduckVelocityFlatPPORunnerCfg" in source
    assert "max_iterations = 6000" in source
    assert "num_steps_per_env = 24" in source
    assert "hidden_dims=[512, 256, 128]" in source
    assert 'obs_groups = {"actor": ["policy"], "critic": ["critic"]}' in source
    assert "save_interval = 250" in source
    assert "clip_actions = 1.0" in source


def test_ppo_runner_keeps_observation_normalization_enabled() -> None:
    source = _source(ROOT / "src/isaaclab_microduck/tasks/agents/rsl_rl_ppo_cfg.py")
    assert source.count("obs_normalization=True") == 2
    assert "obs_normalization=False" not in source


def test_ppo_runner_matches_mjlab_optimization_defaults() -> None:
    source = _source(ROOT / "src/isaaclab_microduck/tasks/agents/rsl_rl_ppo_cfg.py")
    assert "entropy_coef=0.01" in source
    assert "num_learning_epochs=5" in source
    assert "num_mini_batches=4" in source
    assert "learning_rate=1.0e-3" in source
