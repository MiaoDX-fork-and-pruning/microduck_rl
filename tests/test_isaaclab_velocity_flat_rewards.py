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
    assert _weight("alive") <= 0.20
    assert _weight("track_lin_vel") >= 3.5
    assert _literal(_term("track_lin_vel"), "params", "std") <= 0.25


def test_yaw_tracking_is_tight_enough_for_commanded_turns() -> None:
    assert _weight("track_ang_vel") >= 1.5
    assert _literal(_term("track_ang_vel"), "params", "std") <= 0.50
