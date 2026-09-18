from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from mjlab_microduck.tasks.adaptive_curriculum import AxisConfig, CapabilityGate
from mjlab_microduck.tasks.adaptive_runner import AdaptiveMicroduckOnPolicyRunner


class _EventCfg:
    def __init__(self) -> None:
        self.params = {"ranges": (-0.003, 0.003)}


class _EventManager:
    def __init__(self) -> None:
        self.cfg = _EventCfg()

    def get_term_cfg(self, name: str):
        assert name == "randomize_com"
        return self.cfg


class _Env:
    def __init__(self) -> None:
        self.event_manager = _EventManager()
        self.cfg = type("Cfg", (), {"adaptive_evaluator_schema_version": 2, "task_id": "fake"})()


class _Alg:
    def __init__(self) -> None:
        self.state = {"weight": torch.tensor([1.0])}


def _gate() -> CapabilityGate:
    return CapabilityGate(
        (AxisConfig("com_range", (0.003, 0.010), 0.8, 0.6, pass_windows=2),),
        critical_buckets=("zero", "forward", "yaw"),
        axis_mode="com",
        ema_alpha=1.0,
    )


def _runner() -> AdaptiveMicroduckOnPolicyRunner:
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.env = _Env()
    runner.cfg = {"num_steps_per_env": 24, "upload_model": False}
    runner.alg = _Alg()
    runner.current_learning_iteration = 0
    runner.evaluation_events = []
    runner.last_known_good_checkpoint = None
    runner.capability_gate = _gate()
    return runner


@pytest.fixture
def fake_parent_io(monkeypatch):
    """Keep the test on CPU while retaining the runner's real metadata path."""

    def save(self, path, infos=None):
        torch.save(
            {
                "iter": self.current_learning_iteration,
                "infos": infos,
                "fake_alg_state": copy.deepcopy(self.alg.state),
            },
            path,
        )

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        self.current_learning_iteration = payload["iter"]
        self.alg.state = copy.deepcopy(payload["fake_alg_state"])
        return payload["infos"]

    monkeypatch.setattr("mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.save", save)
    monkeypatch.setattr("mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.load", load)


def _good_metrics() -> dict[str, float]:
    return {"zero": 0.95, "forward": 0.90, "yaw": 0.88}


def test_save_load_replays_transition_live_range_and_rng(tmp_path, fake_parent_io):
    runner = _runner()
    checkpoint = tmp_path / "model_0.pt"

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    runner.save(str(checkpoint))
    expected_random = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    # Advance the original branch and record the deterministic expected trace.
    runner.current_learning_iteration = 1
    runner.record_capability_metrics(_good_metrics(), step=1, checkpoint=str(checkpoint), seed=7)
    runner.current_learning_iteration = 2
    runner.record_capability_metrics(_good_metrics(), step=2, checkpoint=str(checkpoint), seed=7)
    expected_state = runner.capability_gate.state_dict()
    expected_range = runner.env.event_manager.cfg.params["ranges"]
    expected_alg = copy.deepcopy(runner.alg.state)

    # Corrupt every state source, then restore the one explicit checkpoint.
    runner.current_learning_iteration = 99
    runner.alg.state["weight"] = torch.tensor([99.0])
    runner.capability_gate = _gate()
    runner.env.event_manager.cfg.params["ranges"] = (-0.99, 0.99)
    random.random()
    np.random.rand()
    torch.rand(1)
    runner.load(str(checkpoint))

    assert runner.current_learning_iteration == 0
    assert runner.capability_gate.state_dict() == _gate().state_dict()
    assert runner.env.event_manager.cfg.params["ranges"] == (-0.003, 0.003)
    assert torch.equal(runner.alg.state["weight"], expected_alg["weight"])
    replay_random = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert replay_random == pytest.approx(expected_random)

    # Replaying the same validated windows from the restored checkpoint gives
    # the same transition and live manager range as the original branch.
    runner.current_learning_iteration = 1
    runner.record_capability_metrics(_good_metrics(), step=1, checkpoint=str(checkpoint), seed=7)
    transition = runner.record_capability_metrics(_good_metrics(), step=2, checkpoint=str(checkpoint), seed=7)
    assert transition is not None
    assert runner.capability_gate.state_dict() == expected_state
    assert runner.env.event_manager.cfg.params["ranges"] == expected_range


def test_load_rejects_incompatible_axis_state_without_partial_apply(tmp_path, fake_parent_io):
    runner = _runner()
    checkpoint = tmp_path / "bad.pt"
    runner.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    payload["infos"]["adaptive_curriculum"]["axis_mode"] = "head_com"
    torch.save(payload, checkpoint)

    before = runner.capability_gate.state_dict()
    with pytest.raises(ValueError, match="axis mode/allowlist"):
        runner.load(str(checkpoint))
    assert runner.capability_gate.state_dict() == before
