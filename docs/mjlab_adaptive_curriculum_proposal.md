# MJLab Adaptive Curriculum Proposal

Status: phase 1 implemented; adaptive experiment not yet run
Date: 2026-09-14

## Execution boundary

- **Owner:** MJLab curriculum experiment; this document is the working contract
  for the adaptive-curriculum implementation and evaluation.
- **Source evidence:** [`adaptive_curriculum_deep_research.md`](adaptive_curriculum_deep_research.md)
  and the repository invariants in [`AGENTS.md`](../AGENTS.md).
- **Implementation surface:** new adaptive task/controller/evaluation code and
  focused tests only; the canonical Velocity task remains read-only.
- **Proof required before declaring success:** deterministic controller unit
  tests, a 64-env five-iteration smoke test, reproducible transition traces,
  and the fixed held-out battery across multiple seeds.
- **Stop condition:** if the adaptive branch cannot match the canonical
  held-out battery or its trace cannot be reproduced after resume, retain the
  fixed schedule and record the result as negative evidence.

## Objective

Improve the current Microduck Velocity policy training recipe in MJLab by
replacing wall-clock advancement of selected curriculum axes with a measured,
reproducible capability controller. The experiment must answer whether the
policy can learn the same final canonical task more reliably when difficulty is
advanced at the policy's pace.

The experiment is intentionally MJLab-only. IsaacLab remains a separate
backend-alignment project and is not used as evidence for this policy result.

## Non-goals and invariants

- Do not modify the existing canonical Velocity task or its fixed schedule.
- Do not change the 61D actor observation layout, 14D action layout, command
  semantics, BAM M6 production model, reward sign conventions, termination
  semantics, or export path.
- Do not change PPO architecture, optimizer, rollout length, or normalization
  while testing the curriculum hypothesis.
- Do not introduce action filtering or a runtime adaptation module.
- Do not switch M4/M6 during one run.
- Do not make reward-weight adaptation part of the first adaptive experiment.

The adaptive run gets a distinct task/experiment name and remains comparable
to the canonical run through the same final command and DR distribution.

## Proposed controller

The controller owns explicit state for each difficulty axis:

```text
axis state = {current_stage, last_transition_step, pass_count, fail_count}
```

Candidate axes, in isolation order:

1. Trunk/head CoM DR range.
2. Action-rate penalty, only as a separately measured diagnostic branch.
3. Standing-command mixture, only as a separately measured diagnostic branch.
4. Head pose command range.
5. Head-pose-bias reward, deferred until environment-only axes are understood.

The first adaptive phase changes only environment/sampling difficulty. Reward weights
remain at their initial values or follow the untouched fixed recipe in a
separate control, so objective adaptation is not mixed with sampling
adaptation. Existing repo audits specifically recommend holding the standing
mix at 0.02 while isolating other overlapping stages. The first run therefore
uses a small matrix rather than assuming action-rate is guilty.

Every 10-20 PPO iterations, run a frozen evaluation battery using fixed seeds.
The curriculum term only updates at these windows, never every environment
step.

## Capability battery

Evaluate the following buckets separately:

```text
zero, forward, lateral, yaw, turn-left, turn-right
```

For each bucket record:

- command-conditioned tracking ratio or normalized velocity error;
- survival/no-fall rate and episode length;
- reset reason;
- root height and 95th-percentile tilt;
- mean action magnitude and action-rate;
- joint-limit proximity and actuator saturation where available.

Use fixed nominal seeds plus a small fixed final-distribution anchor set. Do not
use the training batch reward as the only signal.

Define a conservative aggregate capability as the lower-tail bucket score, not
the mean:

```text
C = min_or_p10(bucket_tracking, bucket_survival, bucket_upright)
```

Advancement requires every critical bucket to pass its own gate; the aggregate
`min_or_p10` is a compact summary and must not hide a failed turn or
zero-command bucket.

Zero-command uses drift/upright metrics rather than a displacement threshold.
Yaw and turn buckets require angular tracking metrics; planar distance alone is
insufficient.

## Transition policy

Initial controller values are hypotheses and must be calibrated against the
existing canonical checkpoints:

- evaluate every 10-20 iterations;
- use an EMA over 3-5 evaluation windows;
- advance only after 2-3 consecutive windows above an upper threshold for all
  critical buckets;
- hold inside a deadband;
- regress only after 2 consecutive windows below a lower threshold;
- enforce a minimum dwell time between transitions;
- change one axis by one stage per transition;
- cap each parameter change to roughly 5-10% of its final range;
- checkpoint policy, optimizer, and curriculum state at every transition.

The thresholds are not paper constants. They must be selected from measured
MJLab battery distributions and recorded in the run config. A transition must
also pass a preservation gate: the nominal/previously mastered buckets may not
drop beyond a predefined tolerance (initially 5%, to be calibrated). Initial
thresholds are calibration hypotheses, not acceptance claims.

On a failed preservation or frontier gate, restore the last successful
checkpoint and retry with the same or a smaller increment. This is the
checkpoint-and-rollback behavior recommended by recent adaptive DR work.

## Sampling policy

Adaptive difficulty must not remove the task's final support:

- retain 10-30% samples from the nominal/final MJLab distribution;
- use the current frontier for the remaining samples;
- keep exact-zero command sampling active;
- keep turn-in-place sampling explicit;
- evaluate final performance on the complete fixed distribution, not the
  adaptive training distribution.

The controller must log a deterministic transition trace:

```text
step, axis, old_stage, new_stage, metrics, threshold, checkpoint, seed
```

Resume must restore this trace state along with the PPO checkpoint.

## Experiment matrix

### A. Canonical control

Run the existing fixed schedule with the current production MJLab task. This is
the attribution baseline and must not be altered.

### B. All-static capability probe

Keep all adaptive axes at their initial values. Confirm that a fresh MJLab run
can develop a useful gait and non-zero command response without any stage
changes.

### C. One-axis gate runs

Run one branch per axis while all other axes remain at their initial values.
At minimum:

- standing-only gate;
- CoM-DR-only gate;
- action-rate-only gate.

The first adaptive phase starts with standing fixed at `0.02` and does not
adapt reward weights. Standing-only and action-rate-only are diagnostic
branches run with the other axes held fixed. Use the same seeds, environment
count, PPO settings, and total environment steps. This identifies whether a
particular stage is harmful or simply early.

### D. Composed adaptive run

Only after the one-axis runs are understood, enable the best two axes. Preserve
the same final values as the canonical recipe and use the same frozen battery.

### E. Final canonical fine-tuning

After the adaptive controller reaches the full target, freeze all ranges and
weights at canonical final values and fine-tune on the complete canonical
distribution. This separates curriculum acquisition from final policy quality.
The primary comparison is the fixed canonical held-out battery, not adaptive
training reward or the speed at which the controller reaches its frontier.

## BAM handling

Use the same M6 actuator and same initial stage for the first adaptive run.
There is no evidence that MJLab maintains an accepted independent M4 schedule.
For an M4/M6 comparison, first run identical task and schedule conditions and
compare actuator traces, saturation, tracking, and survival. Only then may a
model-specific controller delay advancement; it must not redefine the final
task or silently change model semantics.

If real actuator traces become available, prioritize system identification or a
SimOpt-style distribution update before widening BAM DR. Treat voltage,
voltage-drop, friction-scale, and delay as bounded continuous uncertainty axes.
Do not interpolate between categorical M4/M6 models during one policy run.

## Acceptance criteria

The adaptive proposal is successful only if, over multiple seeds:

1. It reaches the same final canonical distribution and ABI.
2. It improves or matches fixed-schedule held-out battery performance.
3. It does not regress nominal/zero-command behavior.
4. It reduces or eliminates the low-motion collapse seen in the clean IsaacLab
   comparison, when tested in MJLab.
5. Every transition is explainable from logged capability metrics.
6. A resumed run reproduces the same transition trace.

If adaptive runs do not beat the canonical baseline, retain the fixed schedule.
That is a valid result: adaptive curriculum is an optimization hypothesis, not
a required architectural change.

## Implementation sequence

1. **Implemented:** add a pure stateful capability-gate helper and unit tests;
   keep it independent of MJLab so state transitions are deterministic.
2. **Implemented:** add the runner adapter for frozen-battery evaluation
   results. `record_capability_metrics(...)` applies one gated live-manager
   transition at a time, and checkpoint `infos` carries the trace/state for
   resume. The battery process that produces the metrics remains an experiment
   harness, not an environment-side reward signal.
3. Add a distinct adaptive Velocity task factory that starts with all axes at
   initial values, with standing fixed at `0.02` and reward-weight adaptation
   disabled for the first phase.
4. Run the all-static and one-axis experiments.
5. Add checkpoint/rollback and final-distribution anchor sampling.
6. Run the composed adaptive experiment and final canonical fine-tune.
7. Compare against the untouched canonical baseline and update the handoff
   document with measured results.
