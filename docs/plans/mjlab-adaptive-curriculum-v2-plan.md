# MJLab Adaptive Curriculum v2 Plan

Status: Phase 0/1 infrastructure complete; Phase 2A native usability remains
unproven. Runner consolidation, native MJLab/BAM evaluation, signed-EMA gate
calibration, and bounded command-exposure feedback are implemented, but the
single-seed acquisition continuation still fails the six-bucket gate across
native reset/DR seeds. See the active status capsule for current proof.
Date: 2026-09-22
Related:

- [`mjlab_adaptive_curriculum_proposal.md`](../mjlab_adaptive_curriculum_proposal.md)
- [`adaptive_curriculum_deep_research.md`](../adaptive_curriculum_deep_research.md)
- [`status/active/mjlab-adaptive-curriculum.md`](../status/active/mjlab-adaptive-curriculum.md)

## Active execution contract

Status: **ACTIVE**. Task control plane: thread
`01a0c2ae-6895-7700-accd-89a0e7a46e1b`, workspace `holy-ape`.
The user authorized sustained necessary changes, training and checks through
`intuitive-flow` until adaptive training produces usable policies. This includes
bounded reward/controller/evaluation repairs; the earlier exposure-only pilot
is not the completion boundary. The latest status request does not cancel this
authorization.

Acceptance remains behavioral: first establish a native six-capability policy,
then reproduce the automated procedure with at least three independent training
seeds (17, 23, 47), each with held-out native evidence, rollout/video inspection,
normalizer-baked export and CPU deployment rehearsal. These are policies in the
existing walking family, not a new collection of task IDs. Native usability,
CPU transfer, and superiority to fixed training remain separate verdicts; no
hardware success is inferred from simulation. Matched-budget comparison follows
acquisition; the optional advisor and stronger teachers remain deferred.

Feedback averages signed tracking error over 0.5 s before L1 magnitude, keeps a
20% exact-zero anchor, and uses a 0.80 focus mastery threshold. The calibrated
product metric keeps linear/angular tracking thresholds at `0.12 m/s` and
`0.6 rad/s`, with separate samplewise stability caps. The cumulative-3500
`lateral-drive` continuation from seed17 completed its planned budget, but
native held-out seed `20260915` still failed turn-left and additional native
seeds `20260921` and `20260922` failed yaw or linear buckets. Eight
preservation failures triggered explicit rollback; both adaptive CoM axes
remained at stage 0. Evidence:
`/tmp/microduck-adaptive-corrected-s17-3500/campaign-result.json`.
This is a valid orchestration result and a negative usability result, not a
reason to open the formal multi-seed matrix.

A separate 500-update strictification diagnostic tested lighter motion costs,
stronger tracking and 25% pure lateral sampling. Its native traces show mean
lateral velocity only 0.0028 m/s against a 0.12 m/s command; forward also
stalls. It survives all six cases but passes none. Evidence:
`/tmp/microduck-strictification-s17/native-499/capability.json`; this working
tree diagnostic has no held-out/transfer acceptance claim. Do not promote it
to the feedback recipe or continue its stricter stage without acquisition
evidence. Next isolate command-conditioned motion and reward mass before
another bounded training intervention. Samplewise MAE and product thresholds
remain fixed; any metric-semantic repair needs a separate calibration argument.

The bounded acquisition-feedback slice combined 0.5 s command-aligned feedback,
staged tracking width, and an initial lateral focus. It improved forward MAE at
500 updates but did not establish lateral acquisition. The checkpointed,
lateral-first diagnostic removed the frontier-order confounder, and the
`lateral-drive` continuation added a yaw-specific acquisition signal. The
pure-yaw exemption improved yaw on some windows, but the cumulative-1000,
cumulative-1500, and cumulative-3500 native reports still show unstable
lateral/yaw/turn retention. The next bounded experiment must therefore measure
command-conditioned exposure and reward mass directly before changing another
reward coefficient.

The final native traces support that diagnosis: held-out mean velocity bias is
small (`-0.015 / -0.0025 m/s` forward/lateral and `+0.014 rad/s` yaw), while
samplewise ripple remains large (`0.0217 / 0.1070 m/s` and `0.2227 rad/s`).
The extra reset/DR seeds move the failing bucket between yaw and linear
commands. The next contract must record bucket-conditioned sample counts and
weighted reward mass, rather than treating a single checkpoint's aggregate as
an acquisition explanation.

## Goal

Build an automated MJLab training system that can adjust bounded difficulty
parameters from measured policy capability and produce a policy that matches or
beats the canonical fixed-schedule policy on a held-out final distribution.

The system will use three layers with distinct responsibilities:

1. The untouched canonical task remains the reproducible baseline and final
   fine-tuning reference.
2. A deterministic adaptive runner owns evaluation cadence, stage transitions,
   checkpoint state, and rollback during a run.
3. An optional Codex advisor analyzes completed windows and proposes the next
   bounded experiment. It cannot directly mutate arbitrary training code or
   bypass controller gates.

The evaluator may run in-process, as a subprocess, or as a separate cluster
job. The adaptive runner remains the authoritative owner of the training and
curriculum state.

## Invariants

- Keep the canonical Velocity task and fixed schedule unchanged.
- Preserve the 61D actor observation ABI, 14D action ABI, command semantics,
  BAM M6 actuator model, reward signs, termination behavior, and mandatory
  normalizer-baked export path.
- Keep PPO architecture, optimizer, rollout length, and normalization fixed
  while testing curriculum behavior.
- Do not introduce action filtering, runtime adaptation, or M4/M6 interpolation.
- Adaptive changes use an explicit, checkpointed allowlist. The user's sustained
  execution authorization also covers necessary reward shaping, regularization
  pacing and evaluation-semantic repairs in the adaptive recipe, after an
  evidence-backed experiment contract. Canonical rewards remain unchanged.
- Every stage change is bounded, logged, reproducible, and reversible.
- Final claims require matched total environment steps, multiple seeds, and a
  fixed held-out battery. Training reward alone is not an acceptance signal.
- A held-out seed is evidence only when it changes a consumed reset, DR, or
  perturbation stream. Distinct labels on deterministic traces do not count as
  independent evaluation.
- A deployment-quality claim must first pass a native MJLab/BAM evaluator. The
  CPU MuJoCo/ONNX rehearsal is a separate transfer check and cannot diagnose
  native training quality by itself.

## Target architecture

```text
AdaptiveMicroduckOnPolicyRunner
  ├─ PPO rollout and update
  ├─ periodic checkpoint (policy + optimizer + RNG + gate state)
  ├─ FrozenCapabilityEvaluator interface
  ├─ CapabilityGate decision
  ├─ live EventManager stage application
  ├─ bounded command-bucket exposure feedback via live CommandManager
  ├─ explicit completed-update/result manifest for launch and resume
  └─ rollback to last-known-good checkpoint

FrozenCapabilityEvaluator
  ├─ fixed seeds and command buckets
  ├─ final-distribution anchor cases
  ├─ continuous tracking/survival/upright metrics
  └─ versioned JSON report

CodexAdvisor (optional outer loop)
  ├─ consumes immutable run manifest and reports
  ├─ returns a schema-validated proposal
  └─ chooses among bounded experiments; never bypasses runner safety rules
```

The shell and Executor scripts become thin launchers. They may submit jobs,
stage artifacts, and retry failed infrastructure, but they must not be the
source of truth for checkpoint selection, curriculum state, or rollback.

## Phase 0: Make the capability signal useful

### Work

- Replace the battery's binary pass score with continuous per-bucket values:
  command tracking error/ratio, survival fraction, episode length, root height,
  tilt percentile, angular tracking for yaw and turns, zero-command drift,
  action magnitude/rate, joint-limit proximity, and actuator saturation where
  available.
- Keep the six required buckets: `zero`, `forward`, `lateral`, `yaw`,
  `turn-left`, `turn-right`.
- Add fixed nominal seeds and a fixed final-distribution anchor set.
- Define a versioned aggregation schema: per-bucket gates plus lower-tail
  aggregate; do not allow a mean to hide a failed bucket.
- Fix axis isolation so `all_static`, `com`, `head_com`, and `composed` each
  construct exactly the intended adaptive axes. A CoM-only run must never
  advance head-CoM.
- Record the evaluation task/config/source SHA, checkpoint, seed, and metric
  schema in every report.

### Tests and evidence

- CPU tests for metric normalization, zero/yaw/turn-specific measures, missing
  or non-finite values, and per-axis isolation.
- A known good and known bad synthetic trace must produce different scores.
- Re-run the existing ONNX battery and confirm scores no longer saturate solely
  because the robot remains upright.

### Gate

Do not enable autonomous transitions until the battery distinguishes at least
one deliberately degraded policy or rollout from the known-good policy on a
continuous metric.

## Phase 1: Move control authority into the runner

### Work

- Add a typed evaluator protocol returning a report for a specific policy
  checkpoint and curriculum state.
- Extend `AdaptiveMicroduckOnPolicyRunner.learn()` with a configured evaluation
  interval. Keep normal PPO rollout/update behavior unchanged between windows.
- Snapshot policy, optimizer, iteration, RNG state, and adaptive state before
  evaluation. Use an explicit checkpoint path instead of globbing for the latest
  file.
- Feed the evaluator report into `CapabilityGate` and apply one legal stage
  transition through `EventManager.get_term_cfg(...)`.
- Store stage values, gate counters, best metrics, transition trace, evaluator
  report hash, and last-known-good checkpoint in the same checkpoint metadata.
- Make rollback restore the complete checkpoint and reapply all live manager
  terms. Record rollback as a first-class trace event.
- Keep the evaluator CLI as a standalone entry point for audit and cluster
  execution. A subprocess adapter may be used if evaluator simulation cannot
  share the training process safely.
- Move stage-file output to an audit artifact. It must not be the primary state
  used to resume training.

### Tests and evidence

- Runner unit tests with a fake evaluator: hold, advance, preservation failure,
  rollback, and resume.
- Save/load test that compares transition traces and live event ranges after
  restoration.
- 64-env, 5-iteration smoke using the adaptive task; verify 61D/14D and finite
  rewards.
- Failure injection test where the evaluator returns a failed bucket and the
  runner stays at or returns to the last good stage.

### Gate

The same checkpoint plus the same evaluator report must reproduce the same
decision and transition trace without relying on shell filename conventions.

## Phase 2A: Repair the adaptive experiment and establish a usable-policy signal

The r4 campaign is retained as historical evidence, but it does not satisfy the
experiment contract: the adaptive factory removed every canonical curriculum
term, the battery seed did not affect the executed trace, and the final battery
used XML position actuators while training used BAM M6. This phase resolves
those confounders before spending another matched-budget campaign.

### Work

- Change the adaptive factory so it removes only the canonical schedules for
  the axes owned by the adaptive controller (`com_range` and/or
  `head_com_range`). Preserve standing, action-rate, command, pose, and other
  canonical curricula exactly. Do not edit the canonical velocity factory.
- Add a native evaluator that runs the checkpoint in the MJLab environment with
  the same BAM M6 actuator, observation normalizer, command ABI, reset path,
  and six bucket definitions used for training. Keep the existing CPU
  MuJoCo/ONNX battery as a separately labelled deployment rehearsal.
- Make evaluation seeds affect an actual consumed source of variation. Use a
  fixed, recorded reset/DR/perturbation manifest so gate and held-out sets are
  disjoint and reproducible. Add a test that two seed sets produce different
  traces while rerunning one seed reproduces the same trace.
- Calibrate the curriculum thresholds, EMA, dwell, and preservation tolerance
  from native reports of canonical checkpoints. Keep the final product gate
  unchanged unless a written calibration report shows that its units or
  semantics are wrong; do not tune thresholds to rescue a failed run.
- Separate two decisions in every report: (a) whether a policy is usable under
  the deployment gate, and (b) whether adaptive matches or beats fixed. A
  usable adaptive policy remains valuable even when it does not win the
  research comparison.

### Tests and evidence

- Config test proving each adaptive mode preserves all non-owned canonical
  curriculum terms and owns only its declared axes.
- Native-vs-export observation/action parity test for one checkpoint, including
  the baked normalizer and 61D/14D ABI.
- Seed-consumption test proving gate and held-out seeds alter a consumed trace;
  same-seed replay remains byte-for-byte deterministic.
- One-checkpoint six-bucket report from native MJLab/BAM and the CPU rehearsal,
  with per-bucket raw metrics and an explicit transfer comparison.
- Single-seed checkpoint ladder at 500/1000/2000/4000 iterations for fixed and
  all-static, recording native capability, CPU rehearsal capability, reward
  terms, episode length, and videos/traces where available.

### Gate

Do not submit another 15-job campaign until the native evaluator has a valid
seed-consumption proof and the checkpoint ladder identifies whether the failure
is native learning, export/observation parity, or CPU actuator transfer. A
policy is **usable** only when its native six-bucket report passes the product
gate; adaptive superiority is a separate later claim.

## Phase 2: Establish the deterministic adaptive baseline

Phase 2 starts only after Phase 2A passes. The r4 matrix and artifacts remain
historical and must not be reused as proof of a valid held-out comparison.

### Work

- Calibrate thresholds, EMA, dwell, and preservation tolerance from measured
  canonical checkpoint distributions, rather than treating current values as
  universal constants.
- Run the following branches with identical PPO settings, total environment
  steps, environment count, and seeds:
  - canonical fixed schedule;
  - all-static initial adaptive slice;
  - CoM-only;
  - head-CoM-only;
  - composed best two-axis controller.
- Keep standing and action-rate diagnostics separate until their own
  controllers and metrics are validated.
- Use at least three training seeds for the final comparison, plus one fixed
  held-out battery seed set not used by the gate.
- Log learning curves, transition timing, per-bucket capability, anchor
  preservation, and rollback count.
- Finish adaptive acquisition by freezing canonical final ranges and run a
  canonical-distribution fine-tuning segment.
- Evaluate the final adaptive checkpoint with both the native product gate and
  the CPU deployment rehearsal, and report the two verdicts separately.

### Gate

Adaptive is accepted only if it reaches the canonical final distribution, has
reproducible transitions, and matches or improves the canonical held-out
battery without nominal or zero-command regression. Otherwise retain the fixed
schedule and record adaptive as negative or inconclusive evidence. A policy can
still be marked usable when it passes the product gate even if this comparison
gate is not won.

## Phase 3: Add the constrained Codex advisor

This phase is optional and starts only after Phase 2 produces informative
metrics and a reliable deterministic runner.

### Work

- Define an immutable run manifest containing source SHA, task ID, config hash,
  PPO settings, seed, checkpoint, curriculum state, battery schema, and all
  reports.
- Define a strict proposal schema with actions such as `hold`, `replicate`,
  `advance_allowed_axis`, `rollback`, `open_diagnostic`, and `stop_negative`.
- Allow only bounded values: approved axis, approved stage delta, maximum
  budget, maximum number of new seeds, and no arbitrary reward/termination/ABI
  changes.
- Require evidence references and a reason code for every proposal.
- Validate proposals deterministically before execution. Invalid or ambiguous
  proposals become `hold` or `replicate`, never an arbitrary code change.
- Run Codex at experiment-window boundaries or between jobs, not inside PPO
  rollout steps and not as the owner of live checkpoint state.
- Compare advisor-selected experiments against a deterministic scheduling
  baseline using the same experiment budget.

### Gate

The advisor is useful only if it improves experiment selection or time-to-good
policy without reducing reproducibility, violating invariants, or increasing
the false-transition rate. It is not required for the adaptive curriculum to
be considered successful.

## Phase 4: Consider stronger active teachers

Only consider ALP-GMM, PLR, DORAEMON-style constrained distribution expansion,
or SimOpt after the deterministic baseline is stable and the failure mode
requires more than a small number of bounded axes.

Each candidate must specify its state, sampling distribution, safety bound,
rollback behavior, and comparison budget before implementation. It must be
compared against the deterministic gate, not introduced as an unqualified
replacement. Real-robot data would be required before using SimOpt for BAM or
other sim-to-real parameter fitting.

## Executor experiment contract

- Use immutable JuiceFS source snapshots and versioned output prefixes.
- Keep one job manifest per branch and seed; never infer provenance from a
  mutable output directory.
- Run smoke64/5 before each new long-run source snapshot.
- Use the existing Executor/CloudML queue only for jobs whose YAML records task,
  source SHA, image, queue, seed, budget, evaluator schema, and output prefix.
- Treat infrastructure failures separately from policy failures.
- Download and verify reports before updating the active status document.

## Deliverables

- `FrozenCapabilityEvaluator` protocol and continuous battery schema.
- Runner-owned adaptive evaluation, checkpoint, rollback, and resume behavior.
- Tests for metrics, axis isolation, deterministic trace replay, and rollback.
- Matched-budget multi-seed comparison report.
- Optional validated Codex advisor schema, validator, and audit log.
- Final canonical fine-tuned ONNX, exported through `scripts/export.py`, followed
  by `scripts/infer_policy.py` deployment rehearsal.
- Updated proposal and active status record with measured results and explicit
  negative evidence where criteria are not met.

## Decision rule

The project does not need to adopt the most sophisticated method. The chosen
system is the smallest one that reliably produces a suitable policy under the
fixed held-out battery and preserves the repository's sim-to-real contracts.
The fixed schedule remains the production fallback until that result is shown.

## Executable planning loop output

The initial document is a roadmap, not an executor handoff. The planning loop
produced the two bounded, reviewable implementation plans below. They resolve
the report, axis, evaluator, checkpoint, rollback, and verification contracts
before any matched-budget compute is requested.

1. [`00-capability-signal-PLAN.md`](mjlab-adaptive-curriculum-v2/00-capability-signal-PLAN.md)
   defines the continuous capability report and end-to-end axis isolation.
2. [`01-runner-control-PLAN.md`](mjlab-adaptive-curriculum-v2/01-runner-control-PLAN.md)
   makes the adaptive runner the state authority after Phase 0 passes.
3. [`02a-diagnostic-repair-PLAN.md`](mjlab-adaptive-curriculum-v2/02a-diagnostic-repair-PLAN.md)
   repairs curriculum ownership, proves seed consumption, and establishes the
   native MJLab/BAM usability signal before new campaign compute.

Phase 2A (experiment repair and native usability diagnosis) is the next active
work item. Phase 2 (matched-budget multi-seed experiments) follows only after
its gate passes. Phase 3 (constrained Codex advisor) and Phase 4 (stronger
active teachers) remain parked until the deterministic baseline produces
informative, native-evaluated evidence.
