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


def test_velocity_flat_uses_bam_asset_and_home_override() -> None:
    source = _source(TASK)
    assert "from isaaclab_microduck.assets.microduck import MICRODUCK_CFG" in source
    assert "return MICRODUCK_CFG.replace(" in source
    assert "joint_pos=home" in source
    assert 'home["right_hip_yaw"] = 0.436' in source
    assert "policy observations and action offsets stay canonical" in source
    assert "offset={name: float(value)" in source


def test_velocity_flat_smoke_checks_observation_and_action_dimensions() -> None:
    tree = ast.parse(_source(SMOKE))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert {"OBSERVATION_SIZE", "ACTION_SIZE"} <= names
    source = _source(SMOKE)
    assert '"IsaacLab-Velocity-Flat-MicroDuck"' in source
    assert "torch.isfinite" in source
    assert "uniform_(-1.0, 1.0)" in source
    assert "GroundPlaneCfg" in source


def test_lazy_task_registry_points_at_velocity_cfg() -> None:
    source = _source(ROOT / "src/isaaclab_microduck/tasks/__init__.py")
    assert "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatEnvCfg" in source
    assert "rsl_rl_cfg_entry_point" in source


def test_ppo_runner_is_smoke_sized_and_keeps_policy_shape() -> None:
    source = _source(ROOT / "src/isaaclab_microduck/tasks/agents/rsl_rl_ppo_cfg.py")
    assert "MicroduckVelocityFlatPPORunnerCfg" in source
    assert "max_iterations = 5" in source
    assert "num_steps_per_env = 24" in source
    assert "hidden_dims=[128, 128, 128]" in source
