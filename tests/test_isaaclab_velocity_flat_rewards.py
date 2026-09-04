from __future__ import annotations

import ast
from pathlib import Path


TASK = Path(__file__).parents[1] / "src/isaaclab_microduck/tasks/velocity_flat.py"


def _rewards_source() -> ast.ClassDef:
    tree = ast.parse(TASK.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg"
    )


def _term(name: str) -> ast.Assign:
    for node in _rewards_source().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node
    raise AssertionError(f"missing reward term: {name}")


def _keyword(term: ast.Assign, name: str) -> ast.AST:
    call = term.value
    assert isinstance(call, ast.Call)
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _literal(term: ast.Assign, keyword: str, key: str) -> float:
    params = _keyword(term, keyword)
    assert isinstance(params, ast.Dict)
    for dict_key, value in zip(params.keys, params.values):
        if isinstance(dict_key, ast.Constant) and dict_key.value == key:
            assert isinstance(value, ast.Constant)
            return float(value.value)
    raise AssertionError(f"missing {key!r} in {keyword}")


def _weight(name: str) -> float:
    value = _keyword(_term(name), "weight")
    assert isinstance(value, ast.Constant)
    return float(value.value)


def test_velocity_tracking_is_strong_enough_to_displace_standing_basin() -> None:
    assert _weight("track_lin_vel") == 2.0
    assert _literal(_term("track_lin_vel"), "params", "std") == 0.31622776601683794
    source = TASK.read_text()
    assert "alive = RewTerm" not in source
    assert "terminating = RewTerm" not in source
    assert "joint_vel = RewTerm" not in source


def test_yaw_tracking_is_tight_enough_for_commanded_turns() -> None:
    assert _weight("track_ang_vel") == 2.0
    assert _literal(_term("track_ang_vel"), "params", "std") == 0.7071067811865476


def test_core_reward_terms_use_mjlab_equivalent_functions() -> None:
    source = TASK.read_text()
    assert "func=track_linear_velocity" in source
    assert "func=track_angular_velocity" in source
    assert "upright = RewTerm(func=upright_gaussian, weight=2.0)" in source
    assert "flat_orientation = RewTerm" not in source


def test_pose_reward_uses_mjlab_variable_posture_stages() -> None:
    source = TASK.read_text()
    assert "def pose_tracking(" in source
    assert "std_standing" in source
    assert "std_walking" in source
    assert "std_running" in source
    assert "walking_threshold" in source
    assert "running_threshold" in source
    assert "torch.mean(-error.square() / selected_std.square(), dim=1)" in source


def test_velocity_flat_keeps_zero_weight_pose_terms_and_limit_penalty() -> None:
    source = TASK.read_text()
    assert "body_pose = RewTerm(" in source
    assert "func=body_pose_tracking" in source
    assert "head_pose_bias = RewTerm(" in source
    assert "func=head_pose_bias_penalty" in source
    assert "dof_pos_limits = RewTerm(" in source
    assert "func=joint_pos_limits" in source
    assert "action_rate_l2 = RewTerm(" in source


def test_angular_momentum_does_not_fallback_to_root_angular_velocity() -> None:
    source = TASK.read_text()
    start = source.index("def angular_momentum_cost(")
    end = source.index("\ndef joint_pos_limits(", start)
    function = source[start:end]
    assert "root_ang_vel_b" not in function
    assert "subtree-angmom" in function
