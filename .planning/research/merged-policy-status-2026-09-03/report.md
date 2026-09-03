# Merged Policy Plan Status And Root-Cause Research

Research date: 2026-09-03

Scope: repository evidence and implementation for the no-wheel/all-collisions
`generalist-v0` / G0 merged policy, plus foundational imitation and multi-task
learning literature. The question is whether the poor results primarily show a
bad plan or implementation drift. Roller, hardware rollout, and production
runtime release decisions are excluded.

## Executive Summary

The merged-policy plan is correctly stopped, but the current evidence does not
justify the provisional diagnosis `state_distribution_coverage_or_model_capacity`.
The dominant problem is an **evidence and implementation mismatch**: the
VELSTAND causal probe was declared as the prerequisite for another PPO run, yet
its evaluator and data path do not implement the declared battery. Therefore the
failed BC/DAgger result is not a valid causal test of coverage or capacity.

There is also a real plan-level weakness. The original G0 plan started with a
three-behavior PPO problem and added a substantial transition router, reward
masking, curriculum, and hybrid anchor before proving that one conditioned model
could reproduce even VELSTAND in closed loop. That reverses the cheapest causal
order. The observed stand/locomotion tradeoff is consistent with shared-model
interference, a known multi-task optimization failure mode ([PCGrad](https://arxiv.org/abs/2001.06782)),
but it is not yet separated from evaluator drift, state coverage, or action
semantics.

The practical decision is: **do not run another merged PPO experiment yet**.
First repair and execute a faithful VELSTAND-only benchmark with independent
teacher, scripted-teacher, random, teacher-initialized, BC, and cumulative DAgger
arms. Only a passing student should unlock an incremental merge.

## Current Status

- G0 status is `BLOCKED_PENDING_DIAGNOSTIC_PLAN`; no direct or hybrid PPO model
  passed P4. See [status](/home/mi/ws/gogo/microduck_rl/docs/status/active/generalist-g0-merged-policy.md).
- Teacher action reconstruction passes on one 300-tick canonical trace, with
  max absolute error `3.84e-7` and mean absolute error `4.78e-8`.
- PPO candidates are finite and pass ABI, ONNX parity, latency, and fallback
  infrastructure checks, but fail behavior and/or legal-edge gates.
- The latest VELSTAND-only BC/DAgger probe has zero closed-loop success and max
  tilt `1.6762 rad` against a `1.1345 rad` gate, but its report explicitly says
  the contract is incomplete.

## Findings

### 1. The strongest confirmed issue is implementation/evaluation drift

The probe manifest requires a 32-episode, seed-42 battery over nine reset
buckets, a 20-second horizon, termination/outcome parity, and the specialist
main-task metric. The actual student battery calls a simplified `run_case` with
one upright reset, 120 ticks, and `main_task_metric: None`, then marks acceptance
false by construction. The status document confirms the reported 32 episodes
repeated the same canonical reset. This makes the result useful as a smoke
failure, not as a diagnosis of recovery coverage or model capacity.

The DAgger collector has the same narrowing: the executable path selects one
behavior, initializes one nominal state, and collects 120 ticks per round. The
plan's frontier definition (first safety-threshold crossing plus eight ticks on
each side), recovery buckets, and cumulative aggregation are not present in that
path. DAgger is specifically intended to correct learner-induced distribution
shift ([DAgger](https://arxiv.org/abs/1011.0686)); collecting only a nominal
trajectory cannot test the stated hypothesis.

### 2. Reported BC MSE overstates what has been proven

The trainer's behavior-balanced sampler oversamples minority rows with
replacement and then splits the sampled indices into train and validation. The
same original row can therefore appear in both partitions. The reported
100-epoch validation MSE (`0.00017076`) is not an independent generalization
estimate. More importantly, low one-step MSE did not translate into stable
closed-loop behavior, which is expected in a sequential control problem and is
why the closed-loop gate is the right endpoint.

### 3. There is credible behavior interference, but it is not isolated

The P2 record shows opposite failures from different actor choices: bounded
DAgger/multi-head variants stabilize stand while locomotion remains outside the
gate, while another unbounded candidate shows the reverse. This is compatible
with shared-trunk/head interference. Multi-task literature independently
documents conflicting task gradients as a failure mechanism ([PCGrad](https://arxiv.org/abs/2001.06782)).
However, these runs also differ in capacity, output bounding, initialization, and
data composition, so they are not a clean ablation. A multi-head result does not
prove that the plan's single actor is impossible; it only says the tested shared
objective has a conflict.

### 4. The original plan has a sequencing flaw

The parent plan made direct PPO and hybrid PPO central P2 baselines while the
three-behavior distillation path was still only a small nominal dataset. It then
added reset routing, long-horizon termination, delayed taxes, transition
curriculum, measured unlocks, and action anchors in response to failures. This
created a large moving surface before a single-behavior upper/lower-bound test
was complete. The later causal probe is the correct direction, but it arrived
after substantial PPO spend and is itself incomplete.

The plan's abstraction is still reasonable: closed-set behavior conditioning,
teacher distillation, explicit scheduler ownership, and specialist fallback are
sound choices. Policy distillation is an established compression strategy
([Policy Distillation](https://arxiv.org/abs/1511.06295)), and retaining old
skills while adding new ones is consistent with continual-learning approaches
such as progressive networks ([Progressive Neural Networks](https://arxiv.org/abs/1606.04671)).
The failure is in validation order and experimental control, not proof that a
generalist is conceptually the wrong product architecture.

### 5. Acceptance and transition evidence are under-grounded

The 90% legal-edge gate is clear as a product requirement, but the source Track
A evidence is one long no-reset sequence rather than statistically independent
per-edge batteries. A single sequence demonstrates scheduler compatibility; it
does not establish a baseline success rate for each edge. The plan should first
measure each legal edge from fixed, reproducible boundary states, then define
the merged target relative to those measurements. Otherwise an edge failure can
mix transition learning failure with an unmeasured teacher baseline.

## Plan Problem Vs Implementation Deviation

### Implementation deviations that must be fixed first

1. Student and teacher batteries are not the same reset/horizon/termination/
   metric contract.
2. VELSTAND DAgger is nominal-only in the executable path and does not collect
   the declared frontier windows or cumulative rounds.
3. Recovery bucket labels are declared in the manifest but not exercised by the
   student harness.
4. The main student report cannot compute the required specialist task metric,
   so it cannot pass the gate even if stability succeeds.
5. BC validation can leak duplicated source examples across train/validation.
6. Control arms (scripted teacher upper bound, random lower bound, and
   teacher-initialized actor) were not completed, so failure localization is
   incomplete.

### Plan/design issues to correct

1. Make a one-behavior closed-loop imitation gate a hard prerequisite before
   any merged PPO or transition curriculum.
2. Separate “can imitate a stable specialist” from “can learn a new transition”;
   do not use PPO to answer both questions at once.
3. Define edge-specific teacher baselines before applying a 90% merged target.
4. Freeze one architecture, action bound, normalization, reset distribution,
   and evaluator for the causal comparison. Multi-head, bounded, and optimizer
   changes belong in separately named arms.
5. Treat the 71D flat actor as a hypothesis. If faithful VELSTAND imitation
   passes but adding the second behavior fails, then test explicit heads,
   gradient-conflict mitigation, or retained specialist adapters rather than
   repeatedly changing reward economics.

## Recommended Next Experiment

This is a bounded diagnostic, not authorization for merged PPO:

1. Rebuild the VELSTAND evaluator from the accepted specialist evaluator:
   same scene, 50 Hz stepping, 20-second horizon, termination classes,
   success definition, reward metric, seed semantics, and all nine reset
   buckets. Run native teacher first and publish its per-bucket baseline.
2. Run five arms on exactly that battery: scripted teacher actions, native
   teacher, random policy, teacher-initialized actor, and nominal BC.
3. Collect balanced nominal/recovery traces. Split by trajectory or episode
   before oversampling; never split duplicated rows.
4. Run cumulative DAgger for up to three rounds, with the declared 8+1+8
   frontier windows and explicit bucket labels. Evaluate after every round.
5. Require the student to pass the VELSTAND diagnostic gate, export ONNX, and
   review a representative video before adding VELOCITY.
6. Only then run a two-behavior merge with a protected VELSTAND non-regression
   gate. If that fails, compare a shared actor against explicit per-behavior
   heads under identical data and seeds.

## Contradictions And Uncertainty

- The implementation status says contracts are “complete,” while the causal
  probe manifest says the evidence contract is incomplete. These refer to
  different layers: plumbing/tests are present, but causal evidence is not.
- Teacher parity is strong on one canonical trace, but full recovery and
  termination parity remain unknown.
- The provisional diagnosis includes model capacity, but no clean capacity
  sweep exists because boundedness, architecture, initialization, and data
  balance changed together.
- PPO failures are informative that the current recipe is not working; they do
  not distinguish reward economics from state coverage without the faithful
  single-behavior controls.

## Gaps

No external source can establish whether this specific robot policy will learn;
the literature only supports the failure mechanisms and experimental method.
The repository lacks a durable, non-ephemeral report containing all PPO run
commands, seeds, per-bucket metrics, and comparable control-arm results. The
current working tree also has an unrelated modified `uv.lock`, which was left
untouched.

## Method

Subquestions were split into current status, evaluator/data fidelity, plan
sequencing, shared-policy failure mechanisms, and next-step design. Primary
repository sources were read first, followed by original papers for DAgger,
policy distillation, gradient interference, and progressive retention. Claims
are recorded in the adjacent [evidence ledger](evidence-ledger.md). The main
limitation is that generated training artifacts and reports under `/tmp` are
ephemeral; conclusions are therefore limited to tracked status files, source
code, manifests, and committed artifact reports.
