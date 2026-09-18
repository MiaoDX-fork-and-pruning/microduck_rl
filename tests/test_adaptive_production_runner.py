"""Production-facing tests for the runner-owned adaptive evaluation boundary.

These tests deliberately use a tiny fake parent runner and CPU-only state.  The
point is to exercise the contract at the boundary where the production runner
hands a checkpoint to the frozen capability battery, records a gate decision,
and persists enough state to resume or roll back deterministically.
"""

from __future__ import annotations

import copy
import hashlib
import random
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report
from mjlab_microduck.tasks.adaptive_curriculum import AxisConfig, CapabilityGate
from mjlab_microduck.tasks.adaptive_runner import (
    AdaptiveMicroduckOnPolicyRunner,
    CommandCapabilityEvaluator,
)
from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    make_microduck_adaptive_velocity_env_cfg,
)


class _TermManager:
    def __init__(self) -> None:
        self.cfgs = {
            "randomize_com": SimpleNamespace(params={"ranges": (-0.003, 0.003)}),
            "randomize_head_com": SimpleNamespace(params={"ranges": (-0.003, 0.003)}),
        }

    def get_term_cfg(self, name: str):
        assert name in self.cfgs
        return self.cfgs[name]


class _Env:
    def __init__(self, *, mode: str = "composed") -> None:
        self.event_manager = _TermManager()
        self.cfg = SimpleNamespace(
            adaptive_axis_mode=mode,
            adaptive_evaluation_interval=2,
            adaptive_evaluation_seed=20260915,
            adaptive_seed_set_id="adaptive-default-20260915",
            adaptive_evaluation_timeout_s=30,
            adaptive_evaluator_schema_version=2,
            task_id="fake-adaptive-task",
        )


class _FakeAlg:
    def __init__(self) -> None:
        self.state = {"weight": torch.tensor([1.0])}


def _fake_parent_io(monkeypatch):
    """Persist only fake PPO state while retaining AdaptiveRunner metadata."""

    def save(self, path, infos=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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

    monkeypatch.setattr(
        "mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.save", save
    )
    monkeypatch.setattr(
        "mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.load", load
    )


def _gate(*, mode: str = "composed", stages=(0.003, 0.005), pass_windows: int = 1):
    axes = [
        AxisConfig(
            "com_range",
            tuple(stages),
            upper_threshold=0.80,
            lower_threshold=0.60,
            pass_windows=pass_windows,
            fail_windows=1,
        )
    ]
    if mode == "composed":
        axes.append(
            AxisConfig(
                "head_com_range",
                tuple(stages),
                upper_threshold=0.80,
                lower_threshold=0.60,
                pass_windows=pass_windows,
                fail_windows=1,
            )
        )
    return CapabilityGate(
        tuple(axes),
        critical_buckets=BUCKETS,
        axis_mode=mode,
        ema_alpha=1.0,
        preservation_tolerance=0.05,
    )


def _runner(*, mode: str = "composed", gate=None) -> AdaptiveMicroduckOnPolicyRunner:
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.env = _Env(mode=mode)
    runner.cfg = {"num_steps_per_env": 24, "upload_model": False}
    runner.alg = _FakeAlg()
    runner.current_learning_iteration = 0
    runner.completed_iterations = 0
    runner.evaluation_interval = 2
    runner.evaluation_seed = 20260915
    runner.evaluation_seed_set_id = "adaptive-default-20260915"
    runner.evaluation_events = []
    runner.last_evaluation_provenance = None
    runner.last_known_good_checkpoint = None
    runner.last_gate_outcome = None
    runner.capability_gate = gate or _gate(mode=mode)
    return runner


def _raw(*, low: bool = False):
    # Keep the report structurally valid even for a low-capability HOLD.  A
    # large tilt is enough to lower the score while all required fields remain
    # finite, so the gate sees a real capability result rather than an error.
    tilt = 1.0 if low else 0.05
    tracking = 1.0 if low else 0.0
    return {
        name: {
            "survival_fraction": 1.0,
            "tilt_p95_rad": tilt,
            (
                "zero_drift_m"
                if name == "zero"
                else "angular_tracking_error_rad_s"
                if name in ("yaw", "turn-left", "turn-right")
                else "tracking_error_m_s"
            ): tracking,
        }
        for name in BUCKETS
    }


def _report(checkpoint: Path, *, mode: str = "composed", low: bool = False):
    return _report_with_raw(checkpoint, _raw(low=low), mode=mode)


def _report_with_raw(checkpoint: Path, raw, *, mode: str = "composed"):
    return build_capability_report(
        raw,
        axis_mode=mode,
        metadata={
            "task_id": "fake-adaptive-task",
            "source_sha": "source",
            "evaluator_config_sha256": "config",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "policy_format": "pt",
            "seed_set_id": "adaptive-default-20260915",
            "generated_at": "test",
        },
    )


def test_constructor_wires_command_evaluator_from_environment(monkeypatch):
    base_init_calls = []

    def fake_base_init(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        base_init_calls.append((env, train_cfg, log_dir, device, kwargs))
        self.cfg = train_cfg

    monkeypatch.setattr(
        "mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.__init__",
        fake_base_init,
    )
    monkeypatch.setenv(
        "MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND",
        "python scripts/run_adaptive_checkpoint_battery.py {checkpoint}",
    )
    runner = AdaptiveMicroduckOnPolicyRunner(_Env(mode="com"), {"num_steps_per_env": 24})

    assert len(base_init_calls) == 1
    assert isinstance(runner.evaluator, CommandCapabilityEvaluator)
    assert runner.evaluator.command.startswith("python scripts/run_adaptive_checkpoint_battery.py")
    assert runner.evaluation_interval == 2
    assert runner.evaluation_seed == 20260915
    assert runner.evaluation_seed_set_id == "adaptive-default-20260915"


def test_command_evaluator_writes_checkpoint_scoped_report_with_provenance(tmp_path):
    checkpoint = tmp_path / "model 2.eval.pt"
    checkpoint.write_bytes(b"checkpoint with a space")
    script = tmp_path / "fake_evaluator.py"
    script.write_text(
        """
import argparse
import hashlib
import json
from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--task-id', required=True)
p.add_argument('--axis-mode', required=True)
p.add_argument('--seed-set-id', required=True)
p.add_argument('--evaluation-seed', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
raw = {}
for name in BUCKETS:
    key = 'zero_drift_m' if name == 'zero' else ('angular_tracking_error_rad_s' if name in ('yaw', 'turn-left', 'turn-right') else 'tracking_error_m_s')
    raw[name] = {'survival_fraction': 1.0, 'tilt_p95_rad': 0.05, key: 0.0}
report = build_capability_report(raw, axis_mode=a.axis_mode, metadata={
    'task_id': a.task_id, 'source_sha': 'source',
    'evaluator_config_sha256': 'config', 'checkpoint': a.checkpoint,
    'checkpoint_sha256': hashlib.sha256(open(a.checkpoint, 'rb').read()).hexdigest(),
    'policy_format': 'pt', 'seed_set_id': a.seed_set_id,
    'generated_at': 'test', 'evaluation_seed': int(a.evaluation_seed),
})
with open(a.output, 'w') as f:
    json.dump(report.payload, f)
""",
        encoding="utf-8",
    )
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} "
        "--checkpoint {checkpoint} --task-id {task_id} --axis-mode {axis_mode} "
        "--seed-set-id {seed_set_id} --evaluation-seed {evaluation_seed} --output {output}"
    )
    evaluator = CommandCapabilityEvaluator(command, timeout_s=30)
    report = evaluator.evaluate(
        checkpoint_path=checkpoint,
        task_id="fake-adaptive-task",
        axis_mode="composed",
        curriculum_state={},
        iteration=2,
        seed_set_id="heldout-v2",
        evaluation_seed=20260915,
    )

    report_path = checkpoint.parent / "adaptive_eval" / checkpoint.stem / "capability.json"
    assert report_path.exists()
    assert report.payload["metadata"]["checkpoint"] == str(checkpoint.resolve())
    assert report.payload["metadata"]["task_id"] == "fake-adaptive-task"
    assert report.payload["metadata"]["seed_set_id"] == "heldout-v2"
    assert report.payload["metadata"]["evaluation_seed"] == 20260915


@pytest.mark.parametrize(
    ("axis_mode", "task_id"),
    [
        ("all_static", "Mjlab-Velocity-Flat-Adaptive-Static-MicroDuck"),
        ("com", "Mjlab-Velocity-Flat-Adaptive-CoM-MicroDuck"),
        ("head_com", "Mjlab-Velocity-Flat-Adaptive-HeadCoM-MicroDuck"),
        ("composed", "Mjlab-Velocity-Flat-Adaptive-MicroDuck"),
    ],
)
def test_production_config_preserves_launch_metadata_and_task_mapping(
    monkeypatch, axis_mode, task_id
):
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_EVALUATION_INTERVAL", "250")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_EVALUATION_SEED", "20260915")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_SEED_SET_ID", "heldout-v2")
    monkeypatch.setenv("MICRODUCK_SOURCE_SHA", "source-sha")
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_EVALUATOR_CONFIG_SHA256", "eval-config-sha")
    cfg = make_microduck_adaptive_velocity_env_cfg(axis_mode=axis_mode)

    assert cfg.task_id == task_id
    assert cfg.adaptive_axis_mode == axis_mode
    assert cfg.adaptive_evaluation_interval == 250
    assert cfg.adaptive_evaluation_seed == 20260915
    assert cfg.adaptive_seed_set_id == "heldout-v2"
    assert cfg.adaptive_source_sha == "source-sha"
    assert cfg.adaptive_evaluator_config_sha256 == "eval-config-sha"


def test_evaluate_window_passes_numeric_seed_and_uses_env_step(monkeypatch, tmp_path):
    runner = _runner(mode="composed")
    runner.current_learning_iteration = 2
    # model_2.eval.pt is emitted after the third completed PPO update.
    runner.completed_iterations = 3
    checkpoint = tmp_path / "model_2.pt"
    checkpoint.write_bytes(b"evaluated checkpoint")
    calls = []

    class Evaluator:
        def evaluate(self, **kwargs):
            calls.append(kwargs)
            assert isinstance(kwargs["evaluation_seed"], int)
            assert kwargs["evaluation_seed"] == 20260915
            assert kwargs["seed_set_id"] == "adaptive-default-20260915"
            assert kwargs["task_id"] == "fake-adaptive-task"
            assert kwargs["axis_mode"] == "composed"
            return _report(checkpoint, mode="composed", low=True)

    runner.evaluator = Evaluator()
    _fake_parent_io(monkeypatch)
    runner._evaluate_window(str(checkpoint))

    assert len(calls) == 1
    assert runner.last_gate_outcome == "hold"
    assert runner.evaluation_events[-1]["kind"] == "hold"
    # CapabilityGate dwell and transitions are expressed in environment steps.
    assert runner.evaluation_events[-1]["step"] == 3 * 24


def test_valid_low_capability_hold_does_not_become_known_good(monkeypatch, tmp_path):
    runner = _runner()
    checkpoint = tmp_path / "model_2.pt"
    checkpoint.write_bytes(b"candidate")

    class Evaluator:
        def evaluate(self, **kwargs):
            return _report(checkpoint, low=True)

    runner.evaluator = Evaluator()
    _fake_parent_io(monkeypatch)
    runner.current_learning_iteration = 2
    runner._evaluate_window(str(checkpoint))

    assert runner.last_gate_outcome == "hold"
    assert runner.last_known_good_checkpoint is None


def test_evaluator_exception_is_audited_and_never_known_good(monkeypatch, tmp_path):
    runner = _runner()
    runner.current_learning_iteration = 2
    runner.completed_iterations = 3
    checkpoint = tmp_path / "model_2.pt"
    checkpoint.write_bytes(b"candidate")

    class Evaluator:
        def evaluate(self, **kwargs):
            raise RuntimeError("battery unavailable")

    runner.evaluator = Evaluator()
    _fake_parent_io(monkeypatch)
    runner._evaluate_window(str(checkpoint))

    assert runner.last_known_good_checkpoint is None
    assert runner.evaluation_events[0]["kind"] == "evaluation_error"
    assert runner.evaluation_events[0]["error"] == "RuntimeError('battery unavailable')"
    assert runner.evaluation_events[0]["iteration"] == 2
    assert runner.evaluation_events[0]["checkpoint"] == str(checkpoint.resolve())


def test_evaluated_checkpoint_stays_immutable_and_hold_saves_post_decision_state(
    monkeypatch, tmp_path
): 
    runner = _runner()
    runner.current_learning_iteration = 2
    runner.completed_iterations = 2
    checkpoint = tmp_path / "model_2.pt"
    original = b"immutable pre-decision checkpoint"
    checkpoint.write_bytes(original)

    class Evaluator:
        def evaluate(self, **kwargs):
            return _report(checkpoint, low=True)

    runner.evaluator = Evaluator()
    _fake_parent_io(monkeypatch)
    runner._evaluate_window(str(checkpoint))

    post = checkpoint.with_name("model_2.adaptive.pt")
    assert checkpoint.read_bytes() == original
    assert post.exists()
    payload = torch.load(post, weights_only=False)
    adaptive = payload["infos"]["adaptive_curriculum"]
    assert runner.completed_iterations == 2
    assert adaptive["evaluation_events"][-1]["kind"] == "hold"
    assert adaptive["evaluation_iteration"] == 2
    assert adaptive["env_step"] == 2 * 24


def test_preservation_failure_rolls_back_policy_rng_and_keeps_chronological_audit(
    monkeypatch, tmp_path
):
    _fake_parent_io(monkeypatch)
    runner = _runner(gate=_gate(mode="com", stages=(0.003, 0.005)))
    runner.current_learning_iteration = 1
    candidate = tmp_path / "model_1.pt"
    candidate.write_bytes(b"evaluated candidate")

    class Evaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _report(candidate, mode="com", low=False)
            # Preserve validity while making one frontier bucket regress by
            # more than the 5% tolerance used by the gate. Rebuild the report
            # so all derived evidence remains canonical.
            raw = _raw(low=False)
            raw["zero"]["zero_drift_m"] = 0.01
            return _report_with_raw(candidate, raw, mode="com")

    runner.evaluator = Evaluator()
    runner.current_learning_iteration = 1
    runner._evaluate_window(str(candidate))
    good = candidate.with_suffix(".adaptive.pt")
    assert runner.last_known_good_checkpoint == str(good)
    expected_policy = copy.deepcopy(runner.alg.state)
    expected_rng = runner._rng_state()

    runner.current_learning_iteration = 99
    runner.alg.state["weight"] = torch.tensor([99.0])
    random.seed(991)
    np.random.seed(991)
    torch.manual_seed(991)

    # The score still clears the pass threshold, but one frontier bucket fell
    # by more than the preservation tolerance.  This must fail closed and
    # restore the exact known-good policy and RNG stream.
    runner.current_learning_iteration = 2
    runner._evaluate_window(str(candidate))

    assert runner.last_gate_outcome == "preservation_failure"
    assert torch.equal(runner.alg.state["weight"], expected_policy["weight"])
    assert runner.current_learning_iteration == 1
    restored_rng = runner._rng_state()
    assert restored_rng["python"] == expected_rng["python"]
    assert np.array_equal(restored_rng["numpy"][1], expected_rng["numpy"][1])
    assert torch.equal(restored_rng["torch"], expected_rng["torch"])
    assert [event["kind"] for event in runner.evaluation_events] == [
        "advance",
        "preservation_failure",
        "rollback",
    ]


def test_disabled_evaluation_delegates_to_stock_learn(monkeypatch):
    calls = []

    def stock_learn(self, num_learning_iterations, init_at_random_ep_len=False):
        calls.append((num_learning_iterations, init_at_random_ep_len))

    monkeypatch.setattr(
        "mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.learn",
        stock_learn,
    )
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.evaluation_interval = 0
    runner.evaluator = None
    runner.learn(7, init_at_random_ep_len=True)

    assert calls == [(7, True)]
