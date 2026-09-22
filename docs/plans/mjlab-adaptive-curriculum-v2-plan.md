# MJLab Adaptive Curriculum v2 Plan

Status: Phase 0/1 automation complete; Phase 2A usable-policy acquisition remains
unproven. Bounded adaptive smoothing relief improved native yaw in one matched
250-update window, but a later candidate was rolled back, the retained policy
fails native/CPU acceptance, and a 10-seed stage diagnostic exposes startup and
lateral-survival failures. Fixed sensor-bias ablations confirm calibration
sensitivity in yaw; matched push ablations locate the lateral recovery regression.
A bounded pure-yaw-only relief treatment (`610fbc5`) passed real smoke64/5
and live per-command cost checks; its 250-update comparison is running. Formal training seeds 17/23/47 remain gated.
Date: 2026-09-23
Related:

- [`mjlab_adaptive_curriculum_proposal.md`](../mjlab_adaptive_curriculum_proposal.md)
- [`adaptive_curriculum_deep_research.md`](../adaptive_curriculum_deep_research.md)
- [`status/active/mjlab-adaptive-curriculum.md`](../status/active/mjlab-adaptive-curriculum.md)

## Active execution contract

### Sustained objective and acceptance

Status: **ACTIVE**. Task control plane: thread
`01a0c2ae-6895-7700-accd-89a0e7a46e1b`, workspace `holy-ape`.
The user authorized sustained necessary changes, training and checks through
`intuitive-flow` until adaptive training produces usable policies, including
bounded reward/controller/evaluation repairs in the adaptive recipe. The earlier
exposure-only experiment's reward freeze is not the current scope boundary.
Canonical Velocity and the sim-to-real invariants below remain protected.

Acceptance is behavioral: retain all six native capability scores at or above
.80 under the final distribution, then reproduce the automated procedure across
training seeds 17/23/47 with fresh held-out native evidence, rollout/video
inspection, normalizer-baked ONNX and CPU rehearsal. Canonical final-range
fine-tuning and matched-budget fixed/axis comparisons remain required. Native
usability, CPU transfer and adaptive superiority are separate verdicts; no
hardware success follows from simulation. Advisor and stronger teachers remain
parked until acquisition is reliable.

Metrics use signed-error EMA over .5 s before magnitude, with linear/angular
normalization thresholds .12 m/s and .6 rad/s and separate samplewise stability
caps. Scores are normalized capability, not success probability. Keep the .80
pass threshold, 20% exact-zero anchor, nominal exposure and directional floors.

### Current retained policy and measured limits

Authoritative evidence:
`/tmp/microduck-adaptive-action-rate-s17-500/experiment-summary.json`.
The seed-17, 4096-env campaign consumed 6250→6750 updates after smoke64/5.
Its 6750 candidate failed turn preservation and was rolled back to the 6500
actor; all 13 actor-state tensors match the known-good snapshot and differ from
the rejected actor. A final filename of `model_6749.pt` records consumed budget,
not a newly retained 6750 policy. Both CoM axes remain at stage 0 (±3 mm).

The matched first 250 updates used identical starting checkpoint and gate seeds
20260815–17: native stage yaw .353 in the control versus .669 with relief.
Both failed mastery. The full relief campaign spent 500 updates and cannot be
reported as a matched 250-update total-budget success.

Retained final native scores (seed 20260915): zero .923, forward .879, lateral
.797, yaw .837, left .737, right .820. Corrected CPU: .940, .903, .727, .000,
.696, .849 in the same order. CPU yaw averages 1.390 rad/s for a .8 command;
native final yaw averages .843. The CPU samplewise error .655 exceeds the .6
cap. This is overspeed, not idle, and is not explained by export mismatch.

The completed stage diagnostic over seeds 20260915–24 has worst scores .914,
.696, .298, .146, .707, .767. No member passes all six; yaw and lateral each
pass 3/10. Lateral falls on seed 20260916. Worst yaw seed 20260919 starts slowly,
briefly reverses, then reaches .86 rad/s in the last two seconds. All 60
native/ONNX action comparisons pass (max absolute error 1.67e-6), with all
report/trace hashes verified. These diagnostics do not replace final-range
held-out acceptance. Their seed set is now diagnostic evidence and future
acceptance must use a fresh held-out set.

### Implemented controller and evidence contracts

`ef4734e` enables bounded action-rate relief only in the adaptive LateralDrive
recipe. Worst yaw/turn capability below .55 activates weight -.2; release
requires .80 yaw/turn mastery or four windows, followed by cooldown. The live
training weight was verified. `75f479f` ensures relief never strengthens an
earlier smaller penalty, clears stale state on legacy loads, prevents rejected
candidates from releasing relief and validates full-resume controller state.
83 focused tests plus the subsequent 6 relief tests pass; focused adaptive Ruff
and diff checks pass. Full `mdp.py` Ruff retains existing findings.

Training used `ef4734e`; the later fixes have not had long-run training proof.
The training snapshot contains 1001 verified copied-worktree files, including
the explicit pre-existing CPU seed overlay. The current diagnostic snapshot
`/tmp/microduck-adaptive-relief-source-75f479f` is `git archive 75f479f` plus
only that overlay (680 verified files). Exact manifests/hashes are in the
experiment summary. New-source training must first pass smoke64/5.

Previously proven infrastructure remains required: runner-owned gate/checkpoint/
rollback state; persisted command/transition/sensor settings; stage-matched
native gate cohorts with exact member provenance; clearing incomparable evidence
on distribution/stage changes; and final-distribution held-out checks kept
separate. `931e3c9` measures CPU velocities in the link frame/origin; old
principal-inertia-frame scores are superseded. A fresh locked non-editable
install/config/model compile passes; fresh-environment GPU and remote execution
remain unproven. Existing tests and smoke artifacts are linked from the capsule
and earlier experiment summaries.

### Current decision-changing diagnostic

Blocker: `reset_conditioned_acquisition_and_cpu_yaw_rate_transfer`.
Relief helped yaw but did not resolve startup/DR robustness or CPU transfer;
blindly extending global relief or direct-yaw/turn-proxy exposure is not the next
experiment. The old 6250 actor's additive-white-noise ablation did not rescue
its failed seed; persistent encoder bias and IMU misalignment were untouched.

The completed fixed-calibration diagnostic holds the actor and additive noise
fixed across nominal/zero-encoder/identity-IMU/both-nominal conditions for seeds
20260915/16/19. All 72 trace hashes pass, six recorded physical reset fields
match between conditions, and actual sensor realizations confirm the intended
interventions. Evidence:
`/tmp/microduck-relief-6500-sensor-bias/analysis-summary.json`.

Seed 20260919 yaw rises .146→.807 with zero encoder bias and mean rate
.360→.830 rad/s; identity IMU alone leaves .115. Seed 20260915 yaw rises
.744→.859 with identity IMU. Lateral still falls on seed 20260916 in all four
conditions. This confirms calibration sensitivity in the tested yaw cases,
while ruling out calibration removal as a sufficient repair for the lateral
failure. It is not product acceptance and does not explain CPU overspeed.

### Bounded pure-yaw relief comparison

The lateral fall is triggered by a seeded push at 3.9 s. Holding recorded
physical/sensor reset state and the pre-push trace fixed, half/no push changes
lateral survival from failure to full survival and scores .298→.801/.807. The
matched 6500 canonical-smoothing control also survives full push (.790), with
identical recorded initial state and actor observation. Actual qvel write
proof and the original cached-velocity logging limitation are recorded at
`/tmp/microduck-relief-6500-push-sensitivity/analysis-summary.json`.

Hypothesis: restricting relief to pure-yaw commands retains its acquisition
benefit while avoiding the lateral recovery regression caused during global
relief training. This remains a training hypothesis, not a proven repair.

Implementation is opt-in through
`MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE=pure_yaw`: positive action-rate
costs are scaled only for zero-planar/nonzero-yaw commands, with the canonical
manager weight and other command costs retained. Pure-yaw capability controls
the existing bounded window. Versioned state preserves the scope through resume
and rollback; legacy controllers remain global and mismatches fail closed.
Default LateralDrive stays global pending evidence. Product pushes, sensors,
commands, reward signs, ABI and gates are unchanged. Focused tests, smoke64/5
and a 64-env live weighted-cost probe pass. The 250-update treatment is running
at `/tmp/microduck-adaptive-pure-yaw-s17-250` on an immutable `610fbc5` source
plus the recorded CPU seed overlay; actor acceptance remains unproven.

Training contract: exact common 6250 checkpoint, seed 17, 4096 environments,
250 updates to 6500, gate seeds 20260815–17, stage distribution and existing
transition settings. The starting checkpoint has no relief controller and may
bootstrap this explicit treatment. Use a clean committed snapshot plus the
recorded CPU seed overlay and smoke64/5 before the budget. Verify actual
weighted action-rate costs in the live environment. Compare both existing
250-update controls, retained stage/final/CPU metrics and the seed-20260916
full-push rollout. Failure to preserve lateral recovery or yaw learning rejects
this treatment as sufficient. Do not extend its budget without a changed
hypothesis. Fresh held-out seeds, multi-training-seed proof and CPU transfer
remain required even if this diagnostic comparison improves.

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
