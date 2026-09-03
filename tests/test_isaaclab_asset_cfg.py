from __future__ import annotations

import ast
from pathlib import Path


def test_asset_cfg_declares_policy_joint_order_without_importing_isaaclab() -> None:
    source = (Path(__file__).parents[1] / "src/isaaclab_microduck/assets/microduck.py").read_text()
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "ArticulationCfg" in names
    assert "BamActuatorCfg" in names
    assert "DCMotorCfg" not in names
    assert "ACTUATED_ORDER" in names
    assert "MICRODUCK_CFG" in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "policy_target_to_sim" in {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
