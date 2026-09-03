# G0 VELSTAND Causal Probe

Status: execution-ready-v2
Parent plan: `docs/plans/generalist-g0-merged-policy-plan.md`
Parent status: `docs/status/active/generalist-g0-merged-policy.md`

## Current Execution Outcome

The implemented probe has produced a preliminary, fail-closed diagnostic but
has not satisfied every evidence requirement in this contract. Teacher action
reconstruction matches the exported ONNX normalizer (`_std + 0.01`) on the
available 300-tick canonical trace (`max_abs=3.84e-7`, `mean_abs=4.78e-8`).
VELSTAND-only nominal BC and two bounded DAgger rounds remain finite but fail
closed loop with success `0.0` and maximum tilt `1.6762 rad` versus the
`1.1345 rad` gate. The provisional diagnosis is
`state_distribution_coverage_or_model_capacity`.

This is not a complete Phase A/B acceptance result. The previous student
harness repeated one upright reset, omitted the specialist main-task metric,
and did not run the declared recovery, frontier, or control comparisons. This
revision is the execution contract for the next context. No merged PPO run or
G0 acceptance claim is authorized until Phase A and Phase B pass.

## Purpose

Identify which layer blocks the G0 merged policy before spending another
multi-behavior PPO run. This is a diagnostic fork, not a replacement for the
G0 acceptance contract.

The probe establishes the smallest causal chain needed before another optimizer
experiment:

```text
teacher compatibility -> closed-loop imitation -> decision on PPO learnability
```

Only after this chain is understood may G0 resume incremental behavior merging.
This fork does not authorize a new PPO run by itself.

## Scope

Included:

- `VELSTAND` only;
- the existing 71D generalist input and 14D raw action contract;
- the manifest-frozen `velstand_flat` specialist teacher;
- the VELSTAND hold and recovery reset buckets already recorded by the parent
  collection/evaluation artifacts;
- native-vs-reconstructed teacher comparison;
- BC/DAgger closed-loop reproduction;
- a decision gate for whether a future single-behavior PPO experiment is
  justified.

Excluded:

- `VELOCITY`, `SITSTAND`, and all other behaviors;
- transition graph, transition rewards, and measured stage unlocks;
- broad reward redesign before an isolated learnability failure is observed;
- additional anchor-strength sweeps;
- recurrent policies, MoE, progressive networks, or runtime changes;
- production 61D ABI, scheduler, hardware, roller, and specialist changes.

## Phase A: evaluator and teacher validity

Rebuild the student evaluator from the exact seed list, episode count,
command buckets, reset-bucket labels, scene/model, termination classes,
reward terms, and corrected 50 Hz semantics recorded by the manifest-frozen
`velstand_flat` specialist acceptance report
(`cloudml/specialist-r2-velstand-acceptance-facd4f4.json`) for:

1. native `velstand_flat` inference;
2. `FrozenG0Teachers` inference through the 71D-to-61D reconstruction;
3. the same rollout battery and recovery cases, including per-bucket metrics.

The student and teacher paths must use the same 20-second horizon and the same
case initialization. A case is never re-used as a substitute for a different
reset bucket. The report must contain one native-teacher baseline row for every
bucket/command case before any student result is interpreted.

Record per-step action difference, action range, finite status, tilt, height,
fall/recovery outcome, and episode termination. Phase A passes only when all
of the following are true over the frozen battery: reconstructed-vs-native
action `max_abs <= 1e-4` and `mean_abs <= 1e-5`; both paths are finite; every
case has the same termination class and success/failure outcome; and no case
disagrees on the teacher report's primary height/tilt outcome. A syntactic
load/parity check alone is insufficient. Copy these thresholds and the exact
seed/battery identifiers into the probe manifest before execution.

Phase A also fails if the evaluator cannot compute the specialist main-task
metric, if a declared bucket is missing, or if the native teacher does not
reproduce its own frozen baseline. “Scripted teacher” is not an open-loop
action replay: the only upper-bound control is the native teacher evaluated
closed loop at the current simulator state.

Before any student data collection, freeze a case list containing exactly 32
seeded episodes, the reset bucket and command bucket for each episode, and the
per-bucket sample counts. Freeze the comparison tolerance at an absolute
success-rate delta of `0.10` from the native teacher, with a minimum bucket
floor of `0.50` when that formula would be lower. Freeze these values in the
manifest and hash the Phase-A baseline report; changing them after seeing a
student result invalidates the probe and requires a new version.

## Phase B: VELSTAND closed-loop imitation and learnability

Train no PPO and no transition router. Use the 71D wrapper with only the
`VELSTAND` condition active.

Freeze one student architecture, activation, output bound, normalization,
action scale, and 71D-to-61D initialization mapping for all arms. The
teacher-initialized arm means the accepted specialist MLP weights copied into
that exact student architecture, with the 48D proprioception and 13D command
blocks mapped explicitly and behavior/phase/posture/side weights initialized
as declared in the manifest. If this mapping cannot be made exact, omit the
arm and record that representation limitation; do not silently substitute a
random initialization.

Compare, on the identical Phase-A battery:

1. native frozen teacher (closed-loop reference);
2. random/no-op lower-bound policy;
3. teacher-initialized student;
4. trajectory-split BC student;
5. cumulative student-state DAgger student.

The native teacher and random/no-op controls validate the evaluator's upper and
lower bounds; they are not training targets. Do not add a separate scripted
action-replay arm.

Collect nominal and recovery buckets separately. Split by whole episode or
trajectory before balancing; oversampling is allowed only inside the training
partition, never before the split. A frontier sample is the first control tick
in a fixed-seed rollout where the declared upright/fall metric crosses its
threshold, plus exactly 8 control ticks before and after. Exclude states marked
physically unrecoverable by the bucket labels and retain those labels in every
dataset shard. DAgger rounds are cumulative and relabel student-induced states
with the native teacher at the same state, command, previous action, and timer.
Evaluate closed loop at fixed seeds, not only validation MSE.

### Phase B gate

The best student must reproduce the manifest-frozen specialist's VELSTAND
battery without weakening G0 P4. Use the frozen teacher baseline to define the
comparison before training: overall success must be at least the teacher rate
minus the manifest tolerance, and every declared recovery bucket must meet its
own minimum floor. The aggregate `0.80`/32-episode gate and main-task metric
`>=18.0` remain required, but aggregate success cannot hide a failed bucket.
Require finite actions, no ABI mismatch, no penalty-sign violation, and no
action-range violation. Max tilt, final height, fall rate, command tracking,
termination class, and recovery time are secondary only when a teacher baseline
exists; they may not be promoted to gates after results are observed.

Phase B budget is fixed before execution: one BC fit of at most 100 optimizer
epochs, followed by at most 3 cumulative DAgger aggregation rounds and 10,000
new relabeled student-state samples per round. Evaluate every arm after every
round on the same battery. Stop on gate pass or after two consecutive
evaluations with no improvement in the primary success metric. The report must
include native-teacher and random/no-op controls, per-bucket counts, and the
teacher-initialization mapping/hash.

If this gate fails, stop and classify the cause as evaluator/teacher semantics,
state-distribution coverage, action representation, initialization, or model
capacity. Do not start merged PPO or add a behavior.

## Phase C: explicit decision before any PPO

Phase C is a decision record, not an execution phase. There is no automatic PPO
run. Existing G0 evidence already contains repeated direct/hybrid PPO failures.
A new single-behavior PPO arm may be proposed only after Phase A/B produce a
passing closed-loop student and a separate approved plan names a materially
different reward, initialization, or actor-objective hypothesis. That plan must
pin the parent G0 VELSTAND slice's scene, termination, reset distribution,
reward terms, action scaling, evaluator, budget, and primary endpoint.

Interpretation of this probe is therefore:

| Result | Diagnosis and next owner |
| --- | --- |
| Phase A fails | Probe invalid; repair teacher semantics/adapter/evaluator before interpreting any student result. |
| Phase A passes; all student arms fail | Fix state coverage, action representation, initialization, or model capacity; do not start merged PPO. |
| Phase A passes; teacher-initialized arm passes but BC/DAgger fails | Treat optimization/data coverage as the blocker; preserve the initialized artifact and do not claim generic distillation success. |
| Phase A passes; BC/DAgger passes | Freeze and hash the student and propose incremental VELOCITY merging with a VELSTAND non-regression gate. |
| A later approved PPO arm fails | Investigate only the newly named reward/objective hypothesis in a separate experiment. |

## Stop conditions

- Stop immediately on teacher compatibility failure.
- Stop Phase B after the fixed data/DAgger budget if no closed-loop metric
  improves; retain the best artifact and diagnosis.
- Do not run Phase C without a separately approved causal experiment.
- Do not claim G0 acceptance from any probe result. Parent P4 remains the only
  acceptance gate.

## Deliverables

- tracked probe manifest with commit, dirty-tree state, commands, seeds, reset
  buckets, teacher hashes, schema, architecture, numeric thresholds, and
  budgets;
- native-vs-reconstructed teacher comparison report with native per-bucket
  baseline;
- control-arm, BC/DAgger dataset, and closed-loop report;
- a conditional PPO decision record (no PPO report is required unless a
  separately approved causal experiment is run);
- a short diagnosis stating which branch of the interpretation table applies.

Generated datasets, checkpoints, videos, and full reports remain outside Git.
The tracked probe manifest records their immutable paths and SHA-256 hashes;
this is the evidence boundary for the parent plan.

## Verification

Every environment change starts with the exact task's normal 64-env,
5-iteration smoke test. Run the focused generalist tests, the corrected 50 Hz
evaluator, and ONNX parity for the selected best student. No runtime or
hardware claim follows from this probe.

## Return to parent plan

- Every failure produces a diagnosis report naming one root-cause class,
  evidence, rejected alternatives, and one bounded parent-plan amendment.
  Re-entry requires parent-plan approval of that amendment plus rerunning Phase
  A and the affected gate; no merged PPO or new behavior starts from a failed
  probe.
- If VELSTAND passes, freeze and hash the passing student, evaluator, seeds,
  per-bucket metrics, and control-arm results. Resume the parent plan by adding
  `VELOCITY` incrementally with protected VELSTAND replay and a non-regression
  gate; only then consider `SITSTAND` and legal transitions.
