from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from mjlab_microduck.tasks.adaptive_curriculum import AxisConfig, CapabilityGate
from mjlab_microduck.tasks.adaptive_runner import AdaptiveMicroduckOnPolicyRunner


class _Manager:
    def __init__(self) -> None:
        self.cfg = type("TermCfg", (), {"params": {"ranges": (-0.003, 0.003)}})()

    def get_term_cfg(self, name):
        assert name == "randomize_com"
        return self.cfg


class _Env:
    def __init__(self) -> None:
        self.event_manager = _Manager()
        self.cfg = type("Cfg", (), {"adaptive_evaluator_schema_version": 2})()


def _gate():
    return CapabilityGate(
        (AxisConfig("com_range", (0.003, 0.010), 0.8, 0.6, pass_windows=1),),
        critical_buckets=("zero", "forward", "yaw"),
        axis_mode="com",
        ema_alpha=1.0,
    )


def _runner():
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.env = _Env()
    runner.cfg = {"num_steps_per_env": 24, "upload_model": False}
    runner.current_learning_iteration = 4
    runner.alg = type("Alg", (), {"state": {"weight": torch.tensor([4.0])}})()
    runner.capability_gate = _gate()
    runner.evaluation_events = []
    runner.last_known_good_checkpoint = None
    return runner


@pytest.fixture
def fake_parent_io(monkeypatch):
    def save(self, path, infos=None):
        torch.save({"iter": self.current_learning_iteration, "infos": infos, "alg": copy.deepcopy(self.alg.state)}, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        self.current_learning_iteration = payload["iter"]
        self.alg.state = copy.deepcopy(payload["alg"])
        return payload["infos"]

    monkeypatch.setattr("mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.save", save)
    monkeypatch.setattr("mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.load", load)


def _metrics(score):
    return {"zero": score, "forward": score, "yaw": score}


def test_rollback_restores_policy_gate_ranges_rng_and_records_event(tmp_path, fake_parent_io):
    runner = _runner()
    checkpoint = tmp_path / "known-good.pt"
    runner.record_capability_metrics(_metrics(0.9), step=4, checkpoint=str(checkpoint), seed=3)
    runner.save(str(checkpoint))
    runner.last_known_good_checkpoint = str(checkpoint)
    expected_gate = runner.capability_gate.state_dict()
    expected_range = runner.env.event_manager.cfg.params["ranges"]
    expected_rng = runner._rng_state()

    runner.current_learning_iteration = 99
    runner.alg.state["weight"] = torch.tensor([99.0])
    runner.capability_gate = _gate()
    runner.env.event_manager.cfg.params["ranges"] = (-0.5, 0.5)
    random.random()
    np.random.rand()
    torch.rand(1)

    runner.rollback(str(checkpoint))
    assert runner.current_learning_iteration == 4
    assert runner.capability_gate.state_dict() == expected_gate
    assert runner.env.event_manager.cfg.params["ranges"] == expected_range
    assert torch.equal(runner.alg.state["weight"], torch.tensor([4.0]))
    assert runner.evaluation_events[-1]["kind"] == "rollback"

    # Loading restored RNG state means the next draws match the saved stream.
    expected_next = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    runner.rollback(str(checkpoint))
    actual_next = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert actual_next[0] == expected_next[0]
    assert np.array_equal(actual_next[1][1], expected_next[1][1])
    assert torch.equal(actual_next[2], expected_next[2])
    assert expected_rng["python"] is not None


def test_rollback_requires_recorded_known_good_state(tmp_path, fake_parent_io):
    runner = _runner()
    checkpoint = tmp_path / "candidate.pt"
    runner.save(str(checkpoint))
    with pytest.raises(ValueError, match="recorded known-good"):
        runner.rollback(str(checkpoint))
