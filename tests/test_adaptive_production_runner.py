"""Production-facing tests for the runner-owned adaptive evaluation boundary.

These tests deliberately use a tiny fake parent runner and CPU-only state.  The
point is to exercise the contract at the boundary where the production runner
hands a checkpoint to the frozen capability battery, records a gate decision,
and persists enough state to resume or roll back deterministically.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.envs.mdp import dr

from mjlab_microduck.evaluation.capability import BUCKETS, build_capability_report, canonical_sha256
from mjlab_microduck.tasks.adaptive_curriculum import (
    AdaptiveActionRateRelief,
    AxisConfig,
    CapabilityGate,
    TransitionExposure,
    evaluation_com_widths,
)
from mjlab_microduck.tasks.adaptive_runner import (
    AdaptiveMicroduckOnPolicyRunner,
    CommandCapabilityEvaluator,
)
from mjlab_microduck.tasks.adaptive_curriculum import EntropyConsolidation
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


def _consolidation_metrics():
    return dict(zip(BUCKETS, (0.95, 0.735, 0.796, 0.669, 0.830, 0.812), strict=True))


def _consolidation_report(runner, metrics):
    # Existing report-contract tests exercise hashes/provenance. This fixture
    # isolates the trainer's response to an already validated frozen report.
    runner._validate_report = lambda *args: None
    runner.evaluator = SimpleNamespace(evaluate=lambda **kwargs: SimpleNamespace(
        payload={"metadata": {}, "schema_version": 2},
        to_gate_metrics=lambda: metrics, sha256=lambda: "verified-test-report",
    ))


@pytest.mark.parametrize("outcome", ["retain", "no_gain", "preservation", "evaluation_error"])
def test_consolidation_boundary_retains_or_rolls_back_and_stops(monkeypatch, tmp_path, outcome):
    from mjlab_microduck.tasks.adaptive_curriculum import CommandExposure

    _fake_parent_io(monkeypatch)
    runner = _runner(gate=_gate(pass_windows=100))
    term = _attach_exposure(runner)
    runner.command_exposure = CommandExposure(initial_focus="lateral")
    runner.command_exposure.apply(runner.env)
    # An ordinary gate boundary first advances the existing teacher; the
    # consolidation backup owns that pre-treatment state. Bootstrap does not
    # run this normal boundary update and has separate coverage below.
    baseline_teacher = copy.deepcopy(runner.command_exposure)
    baseline_teacher.update(_consolidation_metrics())
    original_teacher = baseline_teacher.state_dict()
    original_probabilities = tuple(baseline_teacher.probabilities[name] for name in BUCKETS)
    runner.alg.entropy_coef = 0.01
    runner.entropy_consolidation = EntropyConsolidation()
    runner.evaluation_interval = 250
    runner.completed_iterations = 6500
    runner.current_learning_iteration = 6499
    checkpoint = tmp_path / "model_6499.eval.pt"
    runner.save(str(checkpoint))
    _consolidation_report(runner, _consolidation_metrics())
    runner._evaluate_window(str(checkpoint))
    assert runner.entropy_consolidation.phase == "active"
    assert runner.alg.entropy_coef == 0.0
    assert runner.entropy_consolidation.deadline == 6750
    assert runner.entropy_consolidation.focus_bucket == "yaw"
    assert runner.command_exposure.focus_bucket == "yaw"
    expected_probabilities = tuple(runner.command_exposure.probabilities[name] for name in BUCKETS)
    assert term.cfg.bucket_probabilities == expected_probabilities
    assert term.cfg.bucket_probabilities[BUCKETS.index("yaw")] > original_probabilities[BUCKETS.index("yaw")]
    baseline = Path(runner.entropy_consolidation.baseline_checkpoint)
    baseline_metadata = torch.load(baseline, weights_only=False)["infos"]["adaptive_curriculum"]
    assert baseline_metadata["entropy_coef"] == 0.01
    assert baseline_metadata["command_exposure"] == original_teacher

    # Restart midway: neither the original budget nor backup may be replaced.
    runner.completed_iterations = 6600
    active = tmp_path / "active.pt"
    runner.save(str(active))
    expected_controller = runner.entropy_consolidation.state_dict()
    runner.load(str(active))
    assert runner.entropy_consolidation.state_dict() == expected_controller
    assert term.cfg.bucket_probabilities == expected_probabilities

    runner.completed_iterations = 6750
    runner.current_learning_iteration = 6749
    runner.alg.state["weight"] = torch.tensor([2.0])
    candidate = tmp_path / "model_6749.eval.pt"
    runner.save(str(candidate))
    metrics = (dict.fromkeys(BUCKETS, 0.96) if outcome == "retain"
               else {**_consolidation_metrics(), "forward": 0.4, "zero": 0.5} if outcome == "preservation"
               else _consolidation_metrics())
    _consolidation_report(runner, metrics)
    if outcome == "evaluation_error":
        runner.evaluator.evaluate = lambda **kwargs: (_ for _ in ()).throw(ValueError("bad report"))
    runner._evaluate_window(str(candidate))
    assert runner.entropy_consolidation.terminal
    assert runner.completed_iterations == 6750
    if outcome == "retain":
        assert runner.entropy_consolidation.phase == "retained"
        assert runner.alg.entropy_coef == 0.0
        assert runner.alg.state["weight"].item() == 2.0
    else:
        assert runner.entropy_consolidation.phase == "rejected"
        assert runner.alg.entropy_coef == 0.01
        assert runner.alg.state["weight"].item() == 1.0
        assert runner.last_known_good_checkpoint == str(baseline)
        assert runner.command_exposure.state_dict() == original_teacher
        assert term.cfg.bucket_probabilities == original_probabilities
    with pytest.raises(ValueError, match="do not extend"):
        runner.learn(1)


@pytest.mark.parametrize("yaw,started", [(0.4, False), (0.669, True)])
@pytest.mark.parametrize("relative_log_dir", [False, True])
def test_legacy_consolidation_bootstrap_uses_current_report_without_double_teacher_update(
    monkeypatch, tmp_path, yaw, started, relative_log_dir
):
    _fake_parent_io(monkeypatch)
    runner = _runner(gate=_gate(pass_windows=100))
    runner.capability_gate.best_metrics = _consolidation_metrics()
    runner.alg.entropy_coef = 0.01
    runner.entropy_consolidation = EntropyConsolidation()
    runner._consolidation_needs_baseline = True
    runner.evaluation_interval = 250
    runner.completed_iterations = 6500
    runner.current_learning_iteration = 6499
    monkeypatch.chdir(tmp_path)
    runner.logger = SimpleNamespace(log_dir="logs" if relative_log_dir else str(tmp_path))
    _consolidation_report(runner, {**_consolidation_metrics(), "yaw": yaw})
    checked_paths = []

    def validate(report, checkpoint_path):
        assert Path(checkpoint_path).is_absolute()
        assert Path(checkpoint_path).is_file()
        checked_paths.append(checkpoint_path)

    runner._validate_report = validate
    before = runner.capability_gate.state_dict()
    runner._bootstrap_entropy_consolidation()
    assert (runner.entropy_consolidation.phase == "active") == started
    assert runner.capability_gate.state_dict() == before
    assert not runner._consolidation_needs_baseline
    assert not any(e["kind"] == "hold" for e in runner.evaluation_events)
    assert len(checked_paths) == 1
    if started:
        assert Path(runner.entropy_consolidation.baseline_checkpoint).is_absolute()


def test_learn_initializes_logger_before_bootstrap_can_save():
    runner = _runner()
    runner.device = "cpu"
    runner.is_distributed = False
    runner.env.get_observations = lambda: torch.zeros(1)
    runner.alg.train_mode = lambda: None
    runner.logger = SimpleNamespace()
    runner.logger.init_logging_writer = lambda: setattr(runner.logger, "writer", True)

    def bootstrap():
        assert runner.logger.writer is True
        raise RuntimeError("bootstrap reached after logger initialization")

    runner._bootstrap_entropy_consolidation = bootstrap
    with pytest.raises(RuntimeError, match="bootstrap reached"):
        runner.learn(1)


@pytest.mark.parametrize("resume_pending_evaluation", [False, True])
def test_consolidation_stops_learn_at_its_window_not_the_requested_budget(
    monkeypatch, tmp_path, resume_pending_evaluation
):
    from tensordict import TensorDict

    _fake_parent_io(monkeypatch)
    runner = _runner(gate=_gate(pass_windows=100))
    runner.alg.entropy_coef = 0.01
    runner.entropy_consolidation = EntropyConsolidation()
    runner.completed_iterations = 5
    runner.current_learning_iteration = 4
    runner.evaluation_interval = 2
    runner.cfg.update(num_steps_per_env=1, algorithm={}, save_interval=10)
    runner.is_distributed = False
    runner.device = runner.env.device = "cpu"
    runner.env.num_envs = 2
    runner.env.get_observations = lambda: TensorDict({"actor": torch.zeros(2, 1)}, batch_size=[2])
    runner.env.step = lambda _: (runner.env.get_observations(), torch.zeros(2), torch.zeros(2), {})
    runner.alg.train_mode = lambda: None
    runner.alg.act = lambda _: torch.zeros(2, 1)
    runner.alg.process_env_step = lambda *a: None
    runner.alg.compute_returns = lambda _: None
    runner.alg.update = dict
    runner.alg.learning_rate = 0.001
    runner.alg.get_policy = lambda: SimpleNamespace(output_std=1.0)
    iterations = []
    runner.logger = SimpleNamespace(
        writer=True, log_dir=str(tmp_path), init_logging_writer=lambda: None,
        stop_logging_writer=lambda: None, process_env_step=lambda *a: None,
        log=lambda **kw: iterations.append(kw["it"]),
    )
    runner._begin_entropy_consolidation(_consolidation_metrics(), str(tmp_path / "start.pt"))
    _consolidation_report(runner, dict.fromkeys(BUCKETS, 0.96))
    if resume_pending_evaluation:
        runner.completed_iterations = 7
        runner.current_learning_iteration = 6
    # Global boundary is 6, but the full window from 5 ends at 7. No extra
    # updates are permitted after that deadline, even with ten requested.
    runner.learn(10)
    assert iterations == ([] if resume_pending_evaluation else [5, 6])
    result = json.loads((tmp_path / "training-result.json").read_text())
    assert result["status"] == "stopped"
    assert result["requested_completed_iterations"] == (17 if resume_pending_evaluation else 15)
    assert result["completed_iterations"] == 7
    assert result["segment_iterations"] == (0 if resume_pending_evaluation else 2)
    assert result["stop_reason"] == "entropy_consolidation_terminal"


@pytest.mark.parametrize("legacy_scores", [{}, dict.fromkeys(BUCKETS, 0.9)])
def test_legacy_resume_clears_previous_live_relief(monkeypatch, tmp_path, legacy_scores):
    _fake_parent_io(monkeypatch)
    source = _runner()
    source.capability_gate.best_metrics = legacy_scores
    checkpoint = tmp_path / "legacy.pt"
    source.save(str(checkpoint))
    resumed = _runner()
    resumed.action_rate_relief = AdaptiveActionRateRelief()
    resumed.action_rate_relief.bootstrap(dict.fromkeys(BUCKETS, 0.0))
    resumed.action_rate_relief.apply(resumed.env)
    resumed.load(str(checkpoint))
    assert not resumed.action_rate_relief.active
    assert resumed.action_rate_relief.remaining_windows == 0
    assert resumed.action_rate_relief.triggers == 0
    assert resumed.env._adaptive_action_rate_weight is None


@pytest.mark.parametrize("scope", ["all", "pure_yaw"])
def test_relief_round_trip_and_expiry_survive_policy_rollback(monkeypatch, tmp_path, scope):
    _fake_parent_io(monkeypatch)
    runner = _runner(gate=_gate(pass_windows=100))
    runner.action_rate_relief = AdaptiveActionRateRelief(active_windows=2, scope=scope)
    known_metrics = {**dict.fromkeys(BUCKETS, 0.9), "yaw": 0.2}
    runner.record_capability_metrics(known_metrics)
    assert runner.env._adaptive_action_rate_weight == -0.2
    runner.completed_iterations = 10
    known = tmp_path / "known.pt"
    runner.last_known_good_checkpoint = str(known)
    runner.save(str(known))
    expected_policy = copy.deepcopy(runner.alg.state)
    # The rejected actor acquires yaw but loses forward. Its successful yaw
    # result must not release relief for the restored actor's missing yaw.
    failed = {**dict.fromkeys(BUCKETS, 0.9), "forward": 0.1}
    for completed in (20, 30):
        runner.completed_iterations = completed
        runner.alg.state["weight"] = torch.tensor([float(completed)])
        runner.record_capability_metrics(failed)
        assert runner.last_gate_outcome == "preservation_failure"
        assert runner.completed_iterations == completed
        torch.testing.assert_close(runner.alg.state, expected_policy)
        if completed == 20:
            assert runner.action_rate_relief.active
            assert runner.action_rate_relief.remaining_windows == 1
            repair = tmp_path / "repair.pt"
            runner.save(str(repair))
            state = runner.action_rate_relief.state_dict()
            runner = _runner(gate=_gate(pass_windows=100))
            runner.action_rate_relief = AdaptiveActionRateRelief(active_windows=2, scope=scope)
            runner.load(str(repair))
            assert runner.action_rate_relief.state_dict() == state
            assert runner.env._adaptive_action_rate_weight == -0.2
            assert runner.env._adaptive_action_rate_scope == scope
    assert not runner.action_rate_relief.active
    assert runner.action_rate_relief.remaining_windows == 0
    assert runner.env._adaptive_action_rate_weight is None


def test_full_resume_cannot_silently_drop_inactive_relief_controller(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    runner.action_rate_relief = AdaptiveActionRateRelief()
    checkpoint = tmp_path / "enabled-inactive.pt"
    runner.save(str(checkpoint))
    destination = _runner()
    with pytest.raises(ValueError, match="action-rate relief mismatch"):
        destination.load(str(checkpoint))


def test_full_resume_rejects_changed_action_rate_relief_scope(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    source = _runner()
    source.action_rate_relief = AdaptiveActionRateRelief(scope="pure_yaw")
    checkpoint = tmp_path / "pure-yaw.pt"
    source.save(str(checkpoint))
    destination = _runner()
    destination.action_rate_relief = AdaptiveActionRateRelief()
    with pytest.raises(ValueError, match="scope mismatch"):
        destination.load(str(checkpoint))


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


def test_final_range_finetune_loads_terminal_consolidation_as_new_window(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    source = _runner()
    source.entropy_consolidation = EntropyConsolidation()
    source.alg.entropy_coef = 0.0
    source.entropy_consolidation.begin(
        _consolidation_metrics(), entropy_coef=0.01, checkpoint="baseline.pt",
        completed_iterations=5, window_updates=2, focus_bucket="yaw",
    )
    source.entropy_consolidation.finish(
        _consolidation_metrics(), gate_retained=True, completed_iterations=7,
    )
    source.completed_iterations = 7
    source.current_learning_iteration = 6
    checkpoint = tmp_path / "terminal-consolidation.pt"
    source.save(str(checkpoint))

    destination = _runner()
    destination.final_range_finetune = True
    destination.evaluation_interval = 0
    destination.final_com_fraction = 0.0
    destination.entropy_consolidation = None
    destination.load(str(checkpoint))

    assert destination.entropy_consolidation is None
    assert destination.completed_iterations == 7
    assert destination.evaluation_distribution == "final"
    assert destination.capability_gate.stage_value("com_range") == 0.005
    assert destination.capability_gate.stage_value("head_com_range") == 0.005
    assert destination.evaluation_events[-1]["kind"] == "final_range_finetune_source"


def test_ordinary_adaptive_resume_rejects_final_range_checkpoint(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    source = _runner()
    source.final_range_finetune = True
    source.completed_iterations = 2
    source.current_learning_iteration = 1
    checkpoint = tmp_path / "final-range.pt"
    source.save(str(checkpoint))

    destination = _runner()
    with pytest.raises(ValueError, match="final-range fine-tuning checkpoint"):
        destination.load(str(checkpoint))


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


def _stage_report(checkpoint, runner, *, low=False):
    report = _report(checkpoint, low=low)
    values = {name: runner.capability_gate.stage_value(name) for name in runner.capability_gate.axis_order}
    report.payload["evaluator_config"].update({
        "distribution": "stage", "reference_env_step": 96000,
        "stage_values": values,
        "com_widths": evaluation_com_widths("stage", runner.capability_gate.axis_mode, values),
    })
    report.payload["metadata"]["evaluator_config_sha256"] = canonical_sha256(report.payload["evaluator_config"])
    return report


def test_stage_gate_requires_explicit_migration_and_resume_inherits_contract(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    source = _runner()
    source.completed_iterations = 17
    source.current_learning_iteration = 16
    source.capability_gate.best_metrics = {name: 0.9 for name in BUCKETS}
    source.capability_gate.states["com_range"].ema_score = 0.9
    source.last_known_good_checkpoint = "old-final-distribution.pt"
    source.last_known_good_buckets = BUCKETS
    checkpoint = tmp_path / "legacy-final.pt"
    source.save(str(checkpoint))
    saved = torch.load(checkpoint, weights_only=False)
    saved["infos"]["adaptive_curriculum"].pop("evaluation_distribution")
    torch.save(saved, checkpoint)

    destination = _runner()
    destination.env.cfg.adaptive_evaluation_distribution = "stage"
    with pytest.raises(ValueError, match="explicit migration"):
        destination.load(str(checkpoint))
    destination.env.cfg.adaptive_allow_distribution_migration = True
    destination.load(str(checkpoint))
    assert destination.evaluation_distribution == "stage"
    assert destination.completed_iterations == 17
    assert torch.equal(destination.alg.state["weight"], source.alg.state["weight"])
    assert destination.capability_gate.best_metrics == {}
    assert destination.capability_gate.states["com_range"].ema_score is None
    assert destination.last_known_good_checkpoint is None
    assert destination.last_known_good_buckets == ()
    assert destination.evaluation_events[-1]["kind"] == "distribution_rebaseline"
    stage_checkpoint = tmp_path / "stage.pt"
    destination.save(str(stage_checkpoint))

    resumed = _runner()
    resumed.load(str(stage_checkpoint))
    assert resumed.evaluation_distribution == "stage"
    assert resumed.adaptive_checkpoint_info()["adaptive_curriculum"]["evaluation_distribution"] == "stage"
    assert resumed.evaluation_events == destination.evaluation_events


def test_stage_advance_does_not_treat_old_difficulty_as_new_mastery(monkeypatch, tmp_path):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    runner.evaluation_distribution = "stage"
    first = tmp_path / "stage0.eval.pt"
    runner.save(str(first))
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: _stage_report(first, runner))
    runner._evaluate_window(str(first))
    assert runner.last_gate_outcome == "advance"
    assert runner.capability_gate.stage_value("com_range") == 0.005
    assert runner.capability_gate.best_metrics == {}
    assert runner.last_known_good_checkpoint is None
    assert runner.last_known_good_buckets == ()
    assert all(state.ema_score is None for state in runner.capability_gate.states.values())
    assert runner.evaluation_events[-1]["kind"] == "stage_evidence_rebaseline"

    resumed = _runner()
    resumed.load(str(first.with_suffix(".adaptive.pt")))
    assert resumed.evaluation_distribution == "stage"
    assert resumed.capability_gate.stage_value("com_range") == 0.005
    assert resumed.capability_gate.best_metrics == {}
    next_checkpoint = tmp_path / "stage1.eval.pt"
    resumed.save(str(next_checkpoint))
    resumed.evaluator = SimpleNamespace(evaluate=lambda **kw: _stage_report(next_checkpoint, resumed, low=True))
    resumed._evaluate_window(str(next_checkpoint))
    assert resumed.last_gate_outcome == "regress"
    assert not any(event["kind"] == "rollback" for event in resumed.evaluation_events)


@pytest.mark.parametrize("mutation", [
    lambda config: config.update(distribution="final"),
    lambda config: config["stage_values"].update(com_range=0.005),
    lambda config: config["com_widths"].update(com_range=0.015),
    lambda config: config.update(reference_env_step=0),
])
def test_stage_gate_rejects_wrong_distribution_without_mutating_gate(monkeypatch, tmp_path, mutation):
    _fake_parent_io(monkeypatch)
    runner = _runner()
    runner.evaluation_distribution = "stage"
    checkpoint = tmp_path / "candidate.pt"
    runner.save(str(checkpoint))
    before = runner.capability_gate.state_dict()
    report = _stage_report(checkpoint, runner)
    mutation(report.payload["evaluator_config"])
    report.payload["metadata"]["evaluator_config_sha256"] = canonical_sha256(report.payload["evaluator_config"])
    runner.evaluator = SimpleNamespace(evaluate=lambda **kw: report)
    runner._evaluate_window(str(checkpoint))
    assert runner.last_gate_outcome == "evaluation_error"
    assert runner.capability_gate.state_dict() == before
    assert runner.last_known_good_checkpoint is None
