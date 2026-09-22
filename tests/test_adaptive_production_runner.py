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
from mjlab.envs.mdp import dr

from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report
from mjlab_microduck.tasks.adaptive_curriculum import (
    AxisConfig,
    CapabilityGate,
    TransitionExposure,
)
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
            "randomize_com": SimpleNamespace(func=dr.body_ipos, params={"ranges": (-0.003, 0.003)}),
            "randomize_head_com": SimpleNamespace(func=dr.body_ipos, params={"ranges": (-0.003, 0.003)}),
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
    runner.last_known_good_buckets = ()
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


def test_command_evaluator_writes_checkpoint_scoped_report_with_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model 2.eval.pt"
    checkpoint.write_bytes(b"checkpoint with a space")
    script = tmp_path / "fake_evaluator.py"
    script.write_text(
        """
import argparse
import hashlib
import json
import os
from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report

for _name in (
    'MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY',
    'MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE',
    'MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE',
    'MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE',
):
    assert _name not in os.environ, _name

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
    for name in (
        "MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY",
        "MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE",
        "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE",
        "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE",
    ):
        monkeypatch.setenv(name, "test")
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


def test_partial_mastery_checkpoint_is_rollback_safe_before_aggregate_pass(
    monkeypatch, tmp_path
):
    """Acquisition must preserve early buckets before all six can pass."""
    _fake_parent_io(monkeypatch)
    runner = _runner()
    checkpoint = tmp_path / "model_2.pt"
    checkpoint.write_bytes(b"partial mastery candidate")
    raw = _raw(low=False)
    # Keep only zero and forward above the 0.8 capability threshold.
    raw["lateral"]["tracking_error_m_s"] = 1.0
    for name in ("yaw", "turn-left", "turn-right"):
        raw[name]["angular_tracking_error_rad_s"] = 1.0

    class Evaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, **kwargs):
            self.calls += 1
            return _report_with_raw(
                checkpoint, raw if self.calls == 1 else _raw(low=True)
            )

    runner.evaluator = Evaluator()
    runner.current_learning_iteration = 2
    runner.completed_iterations = 3
    runner._evaluate_window(str(checkpoint))

    good = checkpoint.with_suffix(".adaptive.pt")
    assert runner.last_gate_outcome == "hold"
    assert runner.last_known_good_checkpoint == str(good)
    assert runner.last_known_good_buckets == ("forward", "zero")
    assert good.exists()

    runner.current_learning_iteration = 3
    runner.completed_iterations = 4
    runner._evaluate_window(str(checkpoint))
    assert runner.last_gate_outcome == "preservation_failure"
    assert runner.evaluation_events[-1]["kind"] == "rollback"
    assert runner.last_known_good_checkpoint == str(good)


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
    assert runner.current_learning_iteration == 2
    restored_rng = runner._rng_state()
    assert restored_rng["python"] == expected_rng["python"]
    assert np.array_equal(restored_rng["numpy"][1], expected_rng["numpy"][1])
    assert torch.equal(restored_rng["torch"], expected_rng["torch"])
    assert [event["kind"] for event in runner.evaluation_events] == [
        "advance",
        "preservation_failure",
        "rollback",
    ]


def test_real_ppo_can_train_after_rollback_of_inference_normalizer(tmp_path):
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    torch.manual_seed(19)
    obs = TensorDict({"actor": torch.randn(4, 61)}, batch_size=[4])
    groups = {"actor": ["actor"], "critic": ["actor"]}
    actor = MLPModel(obs, groups, "actor", 14, hidden_dims=[16], obs_normalization=True,
                     distribution_cfg={"class_name": "rsl_rl.modules.distribution:GaussianDistribution"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16], obs_normalization=True)
    runner = _runner()
    runner.env.unwrapped = runner.env
    runner.env.common_step_counter = 24
    runner.alg = PPO(actor, critic, RolloutStorage("rl", 4, 2, obs, [14]),
                     num_learning_epochs=1, num_mini_batches=1, schedule="fixed")

    def update():
        with torch.inference_mode():
            for _ in range(2):
                runner.alg.act(obs)
                runner.alg.process_env_step(obs, torch.randn(4), torch.zeros(4, dtype=torch.bool), {})
            runner.alg.compute_returns(obs)
        return runner.alg.update()

    update()  # Allocate real Adam state and inference-created normalizer buffers.
    assert actor.obs_normalizer._std.is_inference()
    runner.alg.learning_rate = 0.003
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    checkpoint = tmp_path / "known-good.adaptive.pt"
    runner.last_known_good_checkpoint = str(checkpoint)
    runner.save(str(checkpoint))
    saved = torch.load(checkpoint, weights_only=False)
    update()
    runner.alg.learning_rate = 0.009

    runner.rollback(str(checkpoint))
    torch.testing.assert_close(actor.state_dict(), saved["actor_state_dict"])
    torch.testing.assert_close(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"])
    assert runner.alg.learning_rate == 0.003
    # Merely entering inference mode for the entire restore can poison Adam's
    # moments. The next gradient update is the observable rollback contract.
    losses = update()
    assert all(np.isfinite(value) for value in losses.values())
    assert any(not torch.equal(value, saved["actor_state_dict"][name])
               for name, value in actor.named_parameters())


def test_post_rollback_reset_runs_in_inference_mode_for_delay_buffers():
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.env = SimpleNamespace()
    observed = []

    def reset():
        observed.append(torch.is_inference_mode_enabled())
        return None, {}

    runner.env.reset = reset
    obs, _ = runner._reset_after_rollback()
    assert obs is None
    assert observed == [True]


@pytest.mark.parametrize("mode, interval", [("all_static", 0), ("all_static", 2), ("composed", 0), ("composed", 2)])
def test_resume_counts_completed_updates_and_publishes_explicit_result(monkeypatch, tmp_path, mode, interval):
    import json
    from tensordict import TensorDict

    _fake_parent_io(monkeypatch)
    runner = _runner(mode=mode)
    if mode == "all_static":
        runner.capability_gate = None
    runner.completed_iterations = 5
    runner.current_learning_iteration = 4
    runner.evaluation_interval = interval
    runner.cfg.update(num_steps_per_env=1, algorithm={}, save_interval=10)
    runner.is_distributed = False
    runner.device = "cpu"
    runner.env.device = "cpu"
    runner.env.num_envs = 2
    runner.env.get_observations = lambda: TensorDict({"actor": torch.zeros(2, 1)}, batch_size=[2])
    runner.env.step = lambda _: (runner.env.get_observations(), torch.zeros(2), torch.zeros(2), {})
    runner.alg.train_mode = lambda: None
    runner.alg.act = lambda _: torch.zeros(2, 1)
    runner.alg.process_env_step = lambda *a: None
    runner.alg.compute_returns = lambda _: None
    runner.alg.update = lambda: {}
    runner.alg.learning_rate = 0.001
    runner.alg.get_policy = lambda: SimpleNamespace(output_std=1.0)
    iterations, windows = [], []
    runner.logger = SimpleNamespace(
        writer=True, log_dir=str(tmp_path), init_logging_writer=lambda: None,
        stop_logging_writer=lambda: None, process_env_step=lambda *a: None,
        log=lambda **kw: iterations.append(kw["it"]),
    )
    runner._evaluate_window = lambda cp: windows.append(cp)
    runner.learn(2)
    assert iterations == [5, 6]
    assert runner.completed_iterations == 7
    result = json.loads((tmp_path / "training-result.json").read_text())
    assert result["completed_iterations"] == 7
    assert result["segment_iterations"] == 2
    assert result["start_completed_iterations"] == 5
    assert Path(result["checkpoint"]).name == "model_6.pt"
    assert len(windows) == (1 if interval and mode != "all_static" else 0)


def test_actor_only_load_does_not_restore_trainer_rng_gate_or_live_ranges(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    trained = _runner()
    trained.completed_iterations = 100
    checkpoint = tmp_path / "trained.pt"
    trained.save(str(checkpoint))
    evaluator = _runner()
    before_gate = evaluator.capability_gate.state_dict()
    before_rng = evaluator._rng_state()
    before_ranges = copy.deepcopy(evaluator.env.event_manager.cfgs)
    evaluator.load(str(checkpoint), load_cfg={"actor": True})
    assert evaluator.completed_iterations == 0
    assert evaluator.capability_gate.state_dict() == before_gate
    assert evaluator.env.event_manager.cfgs == before_ranges
    assert torch.equal(evaluator._rng_state()["torch"], before_rng["torch"])
    assert evaluator._rng_state()["python"] == before_rng["python"]


def _attach_exposure(runner):
    from mjlab_microduck.tasks.adaptive_curriculum import CommandExposure

    runner.command_exposure = CommandExposure()
    term = SimpleNamespace(cfg=SimpleNamespace(bucket_probabilities=()))
    runner.env.command_manager = SimpleNamespace(get_term=lambda _: term)
    runner.command_exposure.apply(runner.env)
    return term


def _attach_transition(runner, probability: float):
    old_term = None
    manager = getattr(runner.env, "command_manager", None)
    if manager is not None:
        try:
            old_term = manager.get_term("twist")
        except (AttributeError, KeyError):
            old_term = None
    term = SimpleNamespace(
        cfg=SimpleNamespace(
            transition_probability=probability,
            bucket_probabilities=(
                getattr(getattr(old_term, "cfg", None), "bucket_probabilities", ())
            ),
        )
    )
    runner.transition_exposure = TransitionExposure(probability)
    runner.env.command_manager = SimpleNamespace(get_term=lambda _: term)
    runner.transition_exposure.apply(runner.env)
    return term


def test_transition_checkpoint_load_is_exact_and_legacy_load_disables_live_state(
    monkeypatch, tmp_path
):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    term = _attach_transition(runner, 0.30)
    runner.transition_exposure.windows = 7
    runner.transition_exposure.repairs = 2
    runner.transition_exposure.last_reason = "test"
    checkpoint = tmp_path / "transition-state.pt"
    runner.save(str(checkpoint))

    runner.transition_exposure.probability = 0.05
    runner.transition_exposure.windows = 99
    runner.transition_exposure.apply(runner.env)
    runner.load(str(checkpoint))
    assert runner.transition_exposure.state_dict() == {
        "version": 1,
        "probability": 0.30,
        "windows": 7,
        "repairs": 2,
        "last_reason": "test",
    }

    assert term.cfg.transition_probability == pytest.approx(0.30)

    # An old checkpoint has no transition state. Explicit load must clear a
    # live opt-in; a launch override is applied separately by the constructor.
    legacy = tmp_path / "legacy.pt"
    runner.transition_exposure = None
    runner.save(str(legacy))
    runner.transition_exposure = TransitionExposure(0.40)
    runner.transition_exposure.apply(runner.env)
    runner.load(str(legacy))
    assert runner.transition_exposure.probability == 0.0
    assert runner.transition_exposure.windows == 0
    assert runner.transition_exposure.last_reason == "legacy_checkpoint_bootstrap"
    assert term.cfg.transition_probability == 0.0


def test_sensor_reset_fraction_is_checkpointed_and_must_match_environment(
    monkeypatch, tmp_path
):
    _fake_parent_io(monkeypatch)
    source = _runner()
    source.env.cfg.adaptive_sensor_reset_fraction = 1.0
    source.env.event_manager.cfgs["adaptive_sensor_resample"] = SimpleNamespace(
        params={"fraction": 1.0}
    )
    checkpoint = tmp_path / "sensor-reset.pt"
    source.save(str(checkpoint))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["infos"]["adaptive_curriculum"]["sensor_reset_fraction"] == 1.0

    resumed = _runner()
    with pytest.raises(ValueError, match="sensor reset fraction mismatch"):
        resumed.load(str(checkpoint))

    resumed.env.cfg.adaptive_sensor_reset_fraction = 1.0
    resumed.env.event_manager.cfgs["adaptive_sensor_resample"] = SimpleNamespace(
        params={"fraction": 1.0}
    )
    resumed.load(str(checkpoint))
    assert resumed.env.cfg.adaptive_sensor_reset_fraction == 1.0


def test_zero_transition_checkpoint_loads_into_legacy_command_exposure_runner(
    monkeypatch, tmp_path
):
    _fake_parent_io(monkeypatch)
    source = _runner()
    _attach_transition(source, 0.0)
    checkpoint = tmp_path / "disabled-transition.pt"
    source.save(str(checkpoint))

    destination = _runner()
    destination.load(str(checkpoint))
    assert destination.transition_exposure is None


def test_repeated_rollback_preserves_and_repairs_transition_exposure(
    monkeypatch, tmp_path
):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    _attach_exposure(runner)
    term = _attach_transition(runner, 0.20)
    runner.completed_iterations = 10
    known_candidate = tmp_path / "model_9.pt"
    known_candidate.write_bytes(b"known candidate")
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: _report(known_candidate))
    runner._evaluate_window(str(known_candidate))
    known_good = Path(runner.last_known_good_checkpoint)
    # A fully mastered yaw window releases one 0.025 slice.
    assert runner.transition_exposure.probability == pytest.approx(0.175)

    # Mutate live acquisition coverage before a preservation failure. The
    # rollback must restore policy state, then put this live teacher state back
    # and add one bounded repair slice for the regressed yaw frontier.
    runner.transition_exposure.probability = 0.30
    runner.transition_exposure.apply(runner.env)
    candidate = tmp_path / "model_10.pt"
    candidate.write_bytes(b"failing candidate")
    raw = _raw(low=False)
    raw["yaw"]["angular_tracking_error_rad_s"] = 0.4
    runner.evaluator = SimpleNamespace(
        evaluate=lambda **kw: _report_with_raw(kw["checkpoint_path"], raw)
    )
    runner.completed_iterations = 11
    runner._evaluate_window(str(candidate))

    assert runner.last_known_good_checkpoint == str(known_good)
    assert runner.transition_exposure.probability == pytest.approx(0.35)
    assert runner.transition_exposure.repairs == 1
    assert term.cfg.transition_probability == pytest.approx(0.35)


def test_runner_feedback_restores_live_mixture_and_invalid_report_is_noop(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    term = _attach_exposure(runner)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    raw = _raw(low=False)
    raw["lateral"]["tracking_error_m_s"] = 0.12
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: _report_with_raw(checkpoint, raw))
    runner._evaluate_window(str(checkpoint))
    saved = runner.command_exposure.state_dict()
    probabilities = term.cfg.bucket_probabilities
    assert saved["windows"] == 1
    assert saved["focus_bucket"] == "lateral"
    assert saved["probabilities"]["lateral"] > saved["probabilities"]["yaw"]
    assert saved["probabilities"]["zero"] >= 0.20
    assert min(value for name, value in saved["probabilities"].items() if name != "zero") >= 0.08
    gate = runner.capability_gate.state_dict()
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: {"schema_version": -1})
    runner._evaluate_window(str(checkpoint))
    assert runner.command_exposure.state_dict() == saved
    assert runner.capability_gate.state_dict() == gate
    assert term.cfg.bucket_probabilities == probabilities
    runner.command_exposure.update(dict.fromkeys(BUCKETS, 1.0))
    runner.load(str(checkpoint.with_suffix(".adaptive.pt")))
    assert runner.command_exposure.state_dict() == saved
    assert term.cfg.bucket_probabilities == probabilities


def test_low_score_preservation_failure_restores_sampling_without_rewinding_budget(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    term = _attach_exposure(runner)
    runner.completed_iterations = 10
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: _report(checkpoint))
    runner._evaluate_window(str(checkpoint))
    probabilities = term.cfg.bucket_probabilities
    saved_exposure = runner.command_exposure.state_dict()
    runner.completed_iterations = 20
    runner.command_exposure.update({name: 0.0 if name == "lateral" else 1.0 for name in BUCKETS})
    runner.command_exposure.apply(runner.env)
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: _report(checkpoint, low=True))
    runner._evaluate_window(str(checkpoint))
    assert runner.last_gate_outcome == "preservation_failure"
    repaired = runner.command_exposure.state_dict()
    assert repaired != saved_exposure
    assert repaired["retention_repairs"] == 1
    assert repaired["last_repair_buckets"] == ["forward", "lateral", "yaw", "turn-left", "turn-right"]
    assert term.cfg.bucket_probabilities != probabilities
    assert runner.completed_iterations == 20
    assert runner.env.common_step_counter == 20 * 24
    assert runner.evaluation_events[-1]["kind"] == "rollback"


@pytest.mark.parametrize("resume_after_first", [False, True])
@pytest.mark.parametrize("final_com_fraction", [0.0, 0.20])
def test_repeated_preservation_failures_accumulate_live_exposure_repairs(
    monkeypatch, tmp_path, resume_after_first, final_com_fraction
):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    _attach_exposure(runner)
    runner.completed_iterations = 10
    checkpoint = tmp_path / "known-good.eval.pt"
    checkpoint.write_bytes(b"candidate")
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: _report(checkpoint))
    runner._evaluate_window(str(checkpoint))
    known_good = Path(runner.last_known_good_checkpoint)
    known_good_bytes = known_good.read_bytes()
    # The rollback target predates the rehearsal experiment. Auto-rollback
    # must preserve the current distribution across restart, while an explicit
    # load of that target below must restore its old distribution exactly.
    runner._set_final_com_fraction(final_com_fraction)
    expected_policy = copy.deepcopy(runner.alg.state)
    expected_gate = runner.capability_gate.state_dict()
    yaw_probability = runner.command_exposure.probabilities["yaw"]
    windows = runner.command_exposure.windows
    raw = _raw(low=False)
    raw["yaw"]["angular_tracking_error_rad_s"] = 0.4

    for repair_index, completed in enumerate((20, 30), start=1):
        runner.completed_iterations = completed
        runner.current_learning_iteration = completed - 1
        runner.alg.state["weight"] = torch.tensor([float(completed)])
        candidate = tmp_path / f"model_{completed - 1}.eval.pt"
        runner.save(str(candidate))
        runner.evaluator = SimpleNamespace(
            evaluate=lambda **kw: _report_with_raw(kw["checkpoint_path"], raw)
        )
        runner._evaluate_window(str(candidate))
        state = runner.command_exposure.state_dict()
        assert runner.last_gate_outcome == "preservation_failure"
        assert runner.final_com_fraction == final_com_fraction
        for name in ("randomize_com", "randomize_head_com"):
            params = runner.env.event_manager.get_term_cfg(name).params
            assert params.get("final_fraction", 0.0) == final_com_fraction
        assert runner.last_known_good_checkpoint == str(known_good)
        assert known_good.read_bytes() == known_good_bytes
        torch.testing.assert_close(runner.alg.state, expected_policy)
        assert runner.capability_gate.state_dict() == expected_gate
        assert runner.completed_iterations == completed
        assert runner.env.common_step_counter == completed * 24
        assert state["retention_repairs"] == repair_index
        assert state["windows"] == windows + repair_index
        assert state["last_repair_buckets"] == ["yaw"]
        assert state["probabilities"]["yaw"] > yaw_probability
        yaw_probability = state["probabilities"]["yaw"]
        assert state["probabilities"]["zero"] == pytest.approx(0.20)
        assert sum(state["probabilities"].values()) == pytest.approx(0.80)
        live = runner.env.command_manager.get_term("twist").cfg.bucket_probabilities
        assert live == pytest.approx(tuple(state["probabilities"].values()))
        if repair_index == 1 and resume_after_first:
            runner = _runner()
            _attach_exposure(runner)
            runner.load(str(candidate.with_suffix(".adaptive.pt")))
            assert runner.command_exposure.state_dict() == state
            assert runner.final_com_fraction == final_com_fraction
    runner.load(str(known_good))
    assert runner.final_com_fraction == 0.0
    for name in ("randomize_com", "randomize_head_com"):
        term = runner.env.event_manager.get_term_cfg(name)
        assert term.func is dr.body_ipos
        assert "final_fraction" not in term.params


def test_launch_override_is_recorded_after_full_checkpoint_resume(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    checkpoint = tmp_path / "old.pt"
    runner.save(str(checkpoint))

    def init(self, env, cfg, *args, **kwargs):
        self.env = env
        self.cfg = cfg
        self.alg = _FakeAlg()
        self.current_learning_iteration = 0

    monkeypatch.setattr("mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.__init__", init)
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND", "unused")
    env = _Env()
    env.cfg.adaptive_final_com_fraction = 0.2
    new = AdaptiveMicroduckOnPolicyRunner(env, runner.cfg, log_dir=str(tmp_path))
    assert new.final_com_fraction == 0.2
    assert new.evaluation_events[-1] == {
        "kind": "com_rehearsal_override", "previous_fraction": 0.0,
        "final_com_fraction": 0.2, "completed_iterations": 0,
    }
    assert new._needs_reset


def test_transition_launch_override_is_applied_after_full_checkpoint_resume(
    monkeypatch, tmp_path
):
    _fake_parent_io(monkeypatch)
    source = _runner(mode="com")
    source_term = _attach_transition(source, 0.10)
    checkpoint = tmp_path / "old-transition.pt"
    source.save(str(checkpoint))

    def init(self, env, cfg, *args, **kwargs):
        self.env = env
        self.cfg = cfg
        self.alg = _FakeAlg()
        self.current_learning_iteration = 0

    monkeypatch.setattr(
        "mjlab_microduck.tasks.adaptive_runner.MicroduckOnPolicyRunner.__init__",
        init,
    )
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT", str(checkpoint))
    monkeypatch.delenv("MICRODUCK_ADAPTIVE_EVALUATOR_COMMAND", raising=False)
    env = _Env(mode="com")
    env.cfg.adaptive_evaluation_interval = 0
    env.cfg.adaptive_transition_acquisition = True
    env.cfg.adaptive_transition_probability = 0.10
    env.cfg.adaptive_transition_probability_override = 0.30
    env_term = SimpleNamespace(cfg=SimpleNamespace(transition_probability=0.10))
    env.command_manager = SimpleNamespace(get_term=lambda _: env_term)
    new = AdaptiveMicroduckOnPolicyRunner(
        env, source.cfg, log_dir=str(tmp_path)
    )

    assert new.transition_exposure.probability == pytest.approx(0.30)
    assert env_term.cfg.transition_probability == pytest.approx(0.30)
    assert new.evaluation_events[-1] == {
        "kind": "transition_acquisition_override",
        "previous_probability": 0.10,
        "transition_probability": 0.30,
        "completed_iterations": 0,
    }
    assert source_term.cfg.transition_probability == pytest.approx(0.10)


def test_legacy_single_seed_checkpoint_requires_explicit_cohort_migration(
    monkeypatch, tmp_path
):
    _fake_parent_io(monkeypatch)
    source = _runner()
    source.completed_iterations = 17
    source.current_learning_iteration = 16
    source.record_capability_metrics(
        {name: 0.9 for name in BUCKETS}, step=17 * 24, checkpoint="old.pt", seed=7
    )
    checkpoint = tmp_path / "legacy-single-seed.pt"
    source.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=False)
    payload["infos"]["adaptive_curriculum"].pop("evaluation_cohort_size")
    torch.save(payload, checkpoint)

    destination = _runner()
    destination.evaluation_cohort_size = 3
    destination.allow_legacy_cohort_migration = False
    with pytest.raises(ValueError, match="explicit cohort migration"):
        destination.load(str(checkpoint))

    destination = _runner()
    destination.evaluation_cohort_size = 3
    destination.allow_legacy_cohort_migration = True
    destination.load(str(checkpoint))
    assert destination.completed_iterations == 17
    assert torch.equal(destination.alg.state["weight"], source.alg.state["weight"])
    assert destination.capability_gate.best_metrics == {}
    assert all(
        state.pass_count == 0
        and state.fail_count == 0
        and state.ema_score is None
        and state.last_transition_step == -1
        for state in destination.capability_gate.states.values()
    )
    assert destination.last_known_good_checkpoint is None
    assert destination.evaluation_events[-1]["kind"] == "cohort_rebaseline"
    assert destination.evaluation_events[-1]["legacy_known_good"]["checkpoint"] is None
