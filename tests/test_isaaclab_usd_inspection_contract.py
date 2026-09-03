from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/isaaclab/inspect_usd.py"


def test_usd_inspection_reports_joint_and_drive_parity() -> None:
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "SimulationApp" in names
    assert "PhysicsRevoluteJoint" in source
    assert "drive:angular:physics:stiffness" in source
    assert "missing_actuated_joints" in source
    assert "collision_geometry_count" in source
    assert "--output" in source


def test_usd_inspection_uses_policy_order_as_the_comparison_key() -> None:
    source = SCRIPT.read_text()
    assert "for name in ACTUATED_ORDER" in source
    assert "actuated_joint_order" in source
