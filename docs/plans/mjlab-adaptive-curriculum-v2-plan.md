# MJLab Adaptive Curriculum v2 Plan

Status: Phase 0/1 infrastructure complete; Phase 2A usable-policy acquisition is
unproven. The slew acquisition experiment completed 5000→5500 updates but both
gate windows triggered retention rollback. Native cohort gate wiring is now
implemented and verified; formal seeds 17/23/47 remain gated.
Date: 2026-09-23
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
product metric retains linear/angular normalization thresholds of `0.12 m/s`
and `0.6 rad/s`, separate samplewise stability caps, and the existing pass
boundary. Canonical Velocity remains unchanged.

The command-conditioned contract records seven classes of samples and weighted
reward mass. Retention repair (`b2dc088`) redirects bounded sampling toward
regressed commands. The follow-up fix (`b3db958`) preserves live teacher
exposure at an automated preservation failure while restoring policy, optimizer,
gate, ranges and RNG from the known-good checkpoint. Explicit checkpoint load
and rollback still restore saved state. Immutable-checkpoint regression cases
(`873db35`) fail on the old implementation and pass continuously and across
restart. Native windows demonstrate repair counts increasing through 3, 4 and 5.

Current evidence is at `/tmp/microduck-adaptive-slew-s17-5500/` (read-only source
`b942d44`, 676 tracked hashes). The matched control is
`/tmp/microduck-adaptive-yaw-planar-s17-5500/clean-control/` at `e86a1f4`;
both start from the same 5000-update checkpoint and consume 500 updates at
4096 envs with the same 250-update cadence, transition probability and seeds.

`e86a1f4` removes training transition overrides from evaluators. The earlier
zero-bootstrap checkpoint's ten reset/DR seeds yielded yaw scores
`.809, .000, .776, .000, .000, .595, .555, .842, .000, .000`: five zero scores,
only two passes at 0.80. Scalar friction correlation (~0.13) is weak evidence,
not causal exclusion. The blocker is
`direct_command_yaw_under_reset_DR_robustness`.

`ac21204` rejects/reverts the yaw-planar reward treatment (`1f17e81`).
`b942d44` implements actual 1–2 s bootstrap→target command interpolation in the
opt-in transition sampler. Previously it held bootstrap, then jumped to target.
The bounded controller still caps exposure at 0.40, preserves anchors and exact
checkpoint/rollback state, and is disabled in native evaluation. Formal product
commands and thresholds are unchanged.

The slew candidate's 5250/5500 native gate minima are .71733/.72894. Forward
then forward/turn-left retention failures caused rollback; the retained actor
(including normalizer) equals the starting actor. Its held-out native and CPU/XML
minima are both zero. Candidate learning and retained-policy evidence must remain
separate: a reverted actor does not directly measure treatment learning.

The next evaluator slice adds `run_adaptive_native_cohort_battery.py`. It keeps
the single-seed native battery as a primitive, accepts a fixed cohort, selects
the lowest-scoring raw bucket evidence, records the member reports and selected
seed map, and emits a schema-v2 report accepted by the runner. The campaign
launcher defaults to `--gate-cohort-size 3`; held-out evaluation remains an
independent single-seed check. A native 3-seed proof was validated at
`/tmp/microduck-adaptive-cohort-proof/capability-rebuilt.json` with lower-tail
`.72200` and `passed=false`, correctly rejecting the current policy.

The subsequent sensor-reset campaign
`/tmp/microduck-adaptive-sensor-reset-s17-6000-r2/` completed 5,000→6,000
updates with source `07d31a5`. It resampled IMU and encoder realizations at
episode reset and passed the focused/config/smoke checks, but all four native
cohort gates held at CoM stage 0 and the final gate lower-tail was `.267`.
The held-out native lower-tail was `.653` and the CPU/XML transfer lower-tail
was `0.000`; neither passed. This confirms the reset augmentation is wired and
measurable, while leaving native reset/DR robustness and CPU actuator/observation
causality unresolved. Do not spend another matched budget on reset resampling
alone or start the formal 17/23/47 matrix from this result.

A bounded XML actuator probe at
`/tmp/microduck-adaptive-sensor-reset-s17-6000-r2/cpu-actuator-ab.jsonl`
changed the deployment recipe without changing the checkpoint. Current-limit
only traces were identical to the baseline; a 1–2 step action delay changed
turn metrics but left forward/lateral below the tracking gate and the lower-tail
at `0.0`. This is useful negative evidence for the CPU diagnosis, but it is
not a matched BAM-vs-XML experiment and does not authorize a product or
training conclusion by itself.

The cohort repair is now closed at both boundaries: aggregation checks every
member's provenance, evaluator config and trace hash, while the runner checks
the persisted member manifest and recomputes the worst-member selection before
feeding the gate. A legacy single-seed checkpoint can enter a larger cohort
only through an explicit migration flag; the migration retains PPO state and
step budget, clears incomparable mastery/rollback evidence, and records the
old evidence in the audit stream. Gate/held-out overlap checks include the
expanded cohort seed coverage.

Implementation proof: 207 adaptive/config tests, transition-enabled smoke64/5,
fresh cohort-size smoke64/5, and native 64-env command execution. All 26 sampled transitions completed under
the real environment clock within 52–96 steps, reaching exact targets with
matching actor command inputs. No timers were forced to completion. The source
manifest, runtime proof and post-training report paths live in the experiment
directory. Native ONNX parity for the candidate is part of the paired probe.

A fresh real three-member native cohort was run at
`/tmp/microduck-adaptive-native-cohort-smoke/capability.json` and passed the
existing trace validator; its lower-tail score was `0.0` from the intentionally
short 2-update smoke policy, so it validates the evaluator path rather than
policy usability. A separate 64-env runner smoke used a synthetic valid
evaluator to exercise two gate windows, checkpoint save, and resume; it also is
control-flow evidence only.

The 20% final-CoM rehearsal setting remains rejected for default use; its
implementation is opt-in and checkpointed (`da51889`). Routine resumes inherit
settings. Continue only with a changed, evidence-backed hypothesis when a
bounded attempt fails. Required acceptance remains retained native mastery,
fresh held-out/video/export/deployment proof, three independent training seeds,
and the matched-budget comparison; none is replaced by passing unit tests.

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
is native learning, export/observation parity, or CPU actuator transfer. The
sensor-reset run now supplies seed-consumption evidence but does not identify
the CPU/native causal split; the next bounded work must complete that A/B
diagnosis before another full campaign. A
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
