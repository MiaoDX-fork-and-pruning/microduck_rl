# G0 VELSTAND Causal Probe

Status: ready-for-execution
Parent plan: `docs/plans/generalist-g0-merged-policy-plan.md`
Parent status: `docs/status/active/generalist-g0-merged-policy.md`

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

## Phase A: teacher compatibility

Use the exact seed list, episode count, command buckets, reset-bucket labels,
scene/model, and corrected 50 Hz evaluator recorded by the manifest-frozen
`velstand_flat` specialist acceptance report
(`cloudml/specialist-r2-velstand-acceptance-facd4f4.json`) for:

1. native `velstand_flat` inference;
2. `FrozenG0Teachers` inference through the 71D-to-61D reconstruction;
3. the same rollout battery and recovery cases.

Record per-step action difference, action range, finite status, tilt, height,
fall/recovery outcome, and episode termination. Phase A passes only when all
of the following are true over the frozen battery: reconstructed-vs-native
action `max_abs <= 1e-4` and `mean_abs <= 1e-5`; both paths are finite; every
case has the same termination class and success/failure outcome; and no case
disagrees on the teacher report's primary height/tilt outcome. A syntactic
load/parity check alone is insufficient. Copy these thresholds and the exact
seed/battery identifiers into the probe manifest before execution.

## Phase B: VELSTAND closed-loop imitation

Train no PPO and no transition router. Use the 71D wrapper with only the
`VELSTAND` condition active.

Compare:

- teacher-initialized actor;
- BC on balanced nominal/recovery traces;
- student-state DAgger with teacher relabeling near pre-failure/frontier
  states.

Keep nominal and recovery buckets visible in the dataset. Define a frontier
sample as the first control tick in a fixed-seed rollout where the existing
upright/fall safety metric crosses its declared threshold, plus a fixed window
of 8 control ticks before and after that point. Exclude states already marked
physically unrecoverable by the existing bucket labels. Do not treat failed,
physically unrecoverable states as ordinary positive demonstrations. Evaluate
closed loop at fixed seeds, not only validation MSE.

### Phase B gate

The best student must reproduce the manifest-frozen specialist's VELSTAND
battery using the same primary thresholds: at least `0.80` success over the
32-episode seed-42 battery and main-task metric at least `18.0`, with finite
actions, no ABI mismatch, and no new penalty-sign or action-range violation.
Secondary diagnostics include max tilt, final height, fall rate, and command
tracking. They are gates only where the corresponding teacher baseline is
present in the frozen report; otherwise they are diagnosis-only and cannot be
selected post hoc as acceptance criteria. This diagnostic gate does not weaken
G0 P4.

Phase B budget is fixed before execution: one nominal BC fit of at most 100
optimizer epochs, followed by at most 3 DAgger aggregation rounds and 10,000
new relabeled student-state samples per round. Evaluate after each round and
stop early on gate pass or after two consecutive evaluations with no improvement
in the primary success metric. Keep a scripted teacher upper-bound and an
untrained/random lower-bound rollout in the same report.

If this gate fails, stop and classify the cause as adapter/teacher semantics,
state-distribution coverage, action representation, or model capacity. Do not
start merged PPO.

## Phase C: explicit decision before any PPO

There is no automatic Phase C training run. Existing G0 evidence already
contains repeated direct/hybrid PPO failures. A new single-behavior PPO arm may
be proposed only after Phase A/B produce a passing closed-loop student and a
separate approved plan names a materially different reward, initialization, or
actor-objective hypothesis. That plan must pin the parent G0 VELSTAND slice's
scene, termination, reset distribution, reward terms, action scaling, and
evaluator, and must define its own budget and primary endpoint.

Interpretation of this probe is therefore:

| Result | Diagnosis and next owner |
| --- | --- |
| Phase A fails | Probe invalid; repair teacher semantics/adapter/evaluator before interpreting any student result. |
| Phase A passes; BC/DAgger fails | Fix state coverage, action representation, initialization, or model capacity; do not start merged PPO. |
| Phase B passes | Freeze and hash the student, evaluator, seeds, and metrics; propose the next parent-plan increment with a VELSTAND non-regression gate. |
| Phase B passes but a later approved PPO arm fails | Investigate the newly named reward/objective hypothesis in a separate experiment. |

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
- native-vs-reconstructed teacher comparison report;
- BC/DAgger dataset and closed-loop report;
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
- If VELSTAND passes, freeze and hash the passing student, evaluator, seeds, and
  metrics. Resume the parent plan by adding `VELOCITY` incrementally with
  protected replay and a VELSTAND non-regression gate; only then consider
  `SITSTAND` and legal transitions.
