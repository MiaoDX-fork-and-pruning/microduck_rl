# Preflight: G0 VELSTAND Causal Probe

Preflight status: ACTIVE
Task source: `docs/plans/generalist-g0-velstand-causal-probe.md`
Canonical source: `docs/plans/generalist-g0-velstand-causal-probe.md`
Route: durable `$intuitive-flow`

## Goal

Determine whether the G0 failure is caused by teacher/adapter incompatibility,
closed-loop imitation and state-distribution failure, or a later optimizer
objective issue, using a VELSTAND-only probe before resuming merged-policy work.

## Scope

- Validate native `velstand_flat` against `FrozenG0Teachers` through the 71D to
  61D reconstruction.
- Train and evaluate VELSTAND-only BC and bounded student-state DAgger.
- Preserve the existing 71D/14D contract, specialist artifact hashes, reset
  buckets, and corrected 50 Hz evaluator.
- Produce a diagnosis and, only if justified, a proposal for a separate PPO
  experiment.

## Non-goals

- No `VELOCITY`, `SITSTAND`, transition graph, measured unlock, or new G0 edge.
- No automatic PPO run, anchor sweep, broad reward rewrite, recurrent policy,
  MoE, progressive network, runtime, hardware, roller, or specialist change.
- No G0 P4 acceptance claim from this probe.

## Entity budget

- Reuse: existing `generalist_schema.py`, `generalist_teachers.py`, BC/DAgger
  tooling, specialist manifest, evaluator, focused tests, and artifact layout.
- Remove/merge: do not add another merged-PPO curriculum or duplicate P4 gate.
- New: one probe manifest and bounded compatibility/closed-loop report surfaces
  are necessary to preserve reproducible causal evidence; generated datasets,
  checkpoints, videos, and full reports remain outside Git.
- Expansion triggers: any new PPO objective, reward change, actor architecture,
  public ABI, runtime route, or hardware/provider run requires a revised plan
  and new approval.

## Context

Must read:

- `AGENTS.md`
- `docs/plans/generalist-g0-merged-policy-plan.md`
- `docs/plans/generalist-g0-velstand-causal-probe.md`
- `docs/status/active/generalist-g0-merged-policy.md`
- `docs/status/active/generalist-v0-p2.md`
- `docs/plans/generalist-g0-teacher-manifest.json`
- `cloudml/specialist-r2-velstand-acceptance-facd4f4.json`
- `scripts/evaluate_generalist_g0.py`
- `src/mjlab_microduck/generalist_teachers.py`

Useful evidence: existing DAgger traces, P2 rollout reports, and prior G0
diagnostic checkpoints under `artifacts/`, `/tmp/`, and `logs/`.

Avoid unless needed: unrelated specialist families, roller tasks, production
runtime code, and historical generated logs that do not affect the frozen
VELSTAND battery.

## Acceptance

- SUCCESS: Phase A meets the frozen action tolerances (`max_abs <= 1e-4`,
  `mean_abs <= 1e-5`), finite/termination/success outcomes agree, and Phase B
  produces a closed-loop student meeting the manifest-frozen VELSTAND battery
  (`>= 0.80` success over 32 seed-42 episodes and main-task metric `>= 18.0`),
  with the required reports, hashes, and diagnosis.
- BLOCKED_NEEDS_DECISION: a new PPO objective, reward redesign, actor
  architecture, public contract, or external cost class is needed after the
  probe; stop and request a separate approved plan.
- BLOCKED_NEEDS_LOCAL_VALIDATION: required simulator-backed 50 Hz rollout,
  ONNX parity, or manual video review cannot be run; no acceptance claim may
  be made until it is run.
- INTERMEDIATE_ONLY: failed or partial probe artifacts may be retained only as
  diagnostics, never as an accepted G0 candidate.
- No regressions: specialist 61D ABI, production runtime behavior, fallback
  artifacts, and `uv.lock` remain unchanged.

## Verification

Deterministic:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q \
  tests/test_generalist*.py \
  tests/test_prepare_generalist_hybrid.py \
  tests/test_rollout_generalist_bc.py
git diff --check
```

Integration/product-run:

```bash
uv run list-envs
uv run train <VELSTAND-ONLY-TASK> --env.scene.num-envs 64 --agent.max_iterations 5
uv run scripts/<teacher-compatibility-command> --manifest <probe-manifest>
uv run scripts/<velstand-bc-dagger-command> --manifest <probe-manifest>
uv run scripts/evaluate_generalist_g0.py --onnx <best-student.onnx> \
  --observation-reference-onnx artifacts/specialists/velstand_flat/policy.onnx
```

The placeholder script names are implementation deliverables of the fork;
execution must not silently substitute a different evaluator or task. Every
environment change requires the 64-env, 5-iteration smoke first.

Local/live/manual:

- corrected 50 Hz MuJoCo rollout over the frozen seed-42 battery;
- ONNX export and parity for the selected student;
- human review of representative teacher and student videos;
- GPU/provider runs only if explicitly authorized by the execution phase.

## Execution

Main: root supervisor owns route decisions, artifact review, integration, and
final completion/blocker judgment.

Worker: delegated read-only or bounded implementation workers only where file
ownership is explicit; no worker may change the parent G0 acceptance contract.

Worker goal: implement the smallest compatibility/BC/DAgger surfaces required
by the fork, with focused tests and external generated artifacts.

Current slice: Phase A compatibility tooling, adapter semantics, and tracked
probe manifest are implemented and committed (`8051dda`, `029937f`, plus the
normalizer repair below). Phase A passes against the frozen canonical trace.

Last proven evidence: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest
pytest -q tests/test_generalist_teacher_compat.py tests/test_generalist_teachers.py
tests/test_generalist_bc_eval.py tests/test_collect_generalist_dagger.py
tests/test_rollout_generalist_bc.py` -> 12 passed initially and 4 compatibility
tests pass after the repair. Phase A report from
`/tmp/g0-p0-repro/velstand_flat/canonical-compatibility.json` is finite with
`max_abs=3.84e-7`, `mean_abs=4.78e-8`, and `passed=true`. The prior mismatch was
caused by reconstructing with raw `_std` instead of the ONNX denominator
`_std + 0.01`.

Next action: retain the fixed-budget student battery as diagnosis-only and
separate the missing reward-term metric from the stability result. Do not start
PPO.

Phase B diagnostic evidence: the existing bounded candidate was evaluated with
`uv run scripts/rollout_generalist_bc.py` for 120 ticks. Its VELSTAND profile is
finite but fails closed-loop stability (`max_tilt=1.8503 rad` versus the
`1.1345 rad` safety threshold); the same candidate also fails the other legacy
profiles. This is diagnosis-only because it is not the manifest's 32-episode
seed-42 battery or main-task metric gate.

Canonical evaluator evidence: `scripts/evaluate_generalist_g0.py` confirms the
same failure over the 8-second VELSTAND segment: finite 400-step rollout,
`max_tilt=1.7143 rad`, final height `0.0460 m`, and `passed=false`. The bounded
candidate therefore does not justify Phase C or merged PPO. Machine-readable
diagnosis is `/tmp/g0-velstand-causal-diagnosis.json` and classifies the current
branch as `state_distribution_coverage_or_model_capacity`; the remaining
ambiguity requires the full frozen 32-episode recovery-bucket battery.

Battery evidence: `scripts/evaluate_velstand_student_battery.py` ran all 32
seed-42 episodes (the deterministic harness resets to the same canonical pose
for each case). The student is finite but has `success_rate=0.0`; every case
fails the 65-degree stability gate. `main_task_metric` is intentionally `null`
because this CPU student harness does not expose specialist reward terms, so
the acceptance report is fail-closed rather than inferred from tilt.

Nominal BC evidence: the trainer's new explicit `--behavior stand` filter was
used for the fixed 100-epoch fit at `/tmp/g0-velstand-bc-100` (300 VELSTAND
samples, validation MSE `1.38e-5`, model SHA-256
`d7c39458dfcecb73d08bf5c82bf0deba97a479332aeed38fd73cdee24bf28d18`). Its
32-episode battery remains finite but has `success_rate=0.0` and maximum tilt
`1.6816 rad`. This is a closed-loop failure despite low supervised error; the
remaining fixed budget is at most three DAgger rounds of 10,000 samples.

Stop condition: Phase A parity fails (repair semantics and stop), or Phase B
passes/fails at its fixed budget with an external report and root-cause class.

## To execute

`/goal execute docs/plans/generalist-g0-velstand-causal-probe.md with intuitive-flow`

Approval: `LGTM`, `approve`, or `go ahead` approves this contract; edits request
revision.
