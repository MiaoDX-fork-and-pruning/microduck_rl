# Generalist G0 Merged Policy Plan

Status: active

Current checkpoint: P4 diagnostic evaluation complete; hybrid anchor
implementation is now wired and smoke-verified. The versioned teacher
manifest (`docs/plans/generalist-g0-teacher-manifest.json`) and legal transition
graph (`docs/generalist_g0_transition_graph.json`) are hash-verified; data
collection, BC/DAgger, direct PPO, hybrid PPO, ONNX parity, latency, and
fallback tooling are implemented and covered by focused tests. The current
candidate is not accepted: stand and legal-edge success gates remain below
the required thresholds, so work must continue on the merged-policy behavior
before adding any new skill.

This plan defines the first bounded merged-policy experiment for the
no-wheel/all-collisions Microduck. It is intentionally narrower than the
full generalist-v0 program: the goal is to prove that one conditioned model
can represent a small, already-validated transition graph before adding
dynamic skills.

## Decision

Train one no-wheel policy covering these three existing specialists:

```text
velstand_flat
velocity_flat
sitstand_flat
```

The choice is based first on existing no-reset switching evidence and second
on control-structure coverage:

- `velstand_flat`: static balance, stop/stand handoff, and recovery states;
- `velocity_flat`: continuous locomotion and command tracking;
- `sitstand_flat`: a second stable posture and posture transition.

Recovery is a state distribution under the `VELSTAND` condition, not a new
external behavior id. `ground_pick`, `ball_kick`, and `roulade` are explicitly
out of G0.

## Proven transition contract

Only transitions already demonstrated in the all-collisions Track A session
are allowed in the first training and evaluation graph:

```text
VELSTAND <-> VELOCITY
VELSTAND <-> SITSTAND
```

The initial graph does **not** assume that every pair can switch directly.
`VELOCITY -> SITSTAND` and `SITSTAND -> VELOCITY` must route through
`VELSTAND` unless a separate specialist battery proves a direct edge.

The source evidence is the no-reset Track A scenario and final report:

- `docs/specialist_demo_scenario.json`
- `artifacts/generalist-v0/specialist-switch-track-a-final.json`

The exact dwell times, command semantics, and handoff ordering from that
scenario are the initial transition sampling contract. The complete Track A
sequence containing ground-pick, kick, and roulade is retained as future
evidence, but those behaviors are not included in G0.

## Scope

### In scope

- one all-collisions/no-wheel conditioned actor;
- the existing `generalist-v0` schema v2 / 71D input and 14D raw action;
- immutable teacher data from the three accepted specialists;
- recovery states labeled as `VELSTAND`;
- offline BC and student-state DAgger;
- a small conditioned PPO environment and PPO fine-tuning comparison;
- transition-aware sampling on the proven graph;
- deterministic MuJoCo rollout, ONNX parity, and per-behavior regression gates;
- specialist fallback preservation.

### Out of scope

- roller hardware or any cross-profile model;
- `ground_pick`, `ball_kick`, `roulade`, kick-side expansion, or new skills;
- arbitrary direct switching between unproven policy pairs;
- modifying the production 61D specialist ABI;
- production runtime defaults or hardware rollout;
- a recurrent policy, MoE, motion-reference model, or full multi-task PPO over
  all skills;
- replacing the official runtime scheduler with a Python scheduler.

## Current starting point

The schema adapter, BC trainer, DAgger collector, model tests, and rollout
tools already exist. P2 evidence shows that the current shared dense and
multi-head BC/DAgger candidates have a stand/locomotion tradeoff and have not
passed the G0 rollout gate. Treat those runs as diagnostics, not accepted
models. Do not add skills until the gates below pass.

## Work phases

### P0. Freeze evidence and contract

1. Create an untracked/versioned teacher manifest with checkpoint and ONNX
   hashes for the three teachers, repository commit, scene/model id, and exact
   evaluation commands.
2. Freeze the 71D field layout, normalization, behavior order, command adapter,
   and golden vectors. Add metadata requirements for the eventual ONNX.
3. Add a machine-readable transition graph that names only the four legal
   graph edges and records unsupported direct edges.
4. Confirm each teacher independently and on the proven Track A sequence with
   fixed seeds, finite actions, non-positive penalties, and zero hidden resets.

Stop if any teacher or graph edge cannot be reproduced. Fix or exclude it
before collecting merged-policy data.

### P1. Collect balanced teacher and transition data

1. Collect nominal teacher rollouts for each behavior.
2. Collect recovery buckets under `VELSTAND`: face-up, face-down, left/right
   side, crouched, natural fall from locomotion, and post-recovery upright.
3. Collect boundary windows before and after each legal handoff using the real
   command and dwell semantics. Do not fabricate labels for unproven direct
   edges.
4. Balance by behavior, recovery bucket, command bucket, and transition phase;
   preserve previous action, episode timers, command-manager state, and all
   replay state needed to reproduce labels.
5. Validate finite values, exact 71D/14D shapes, field coverage, teacher hash,
   and deterministic replay.

Suggested progression: 10k samples per behavior for smoke, 100k for debug,
then a balanced G0 dataset sized by coverage rather than raw frame count.

### P2. Establish three comparable student baselines

Use the same data, seeds, architecture budget, rollout battery, and output
contract for:

1. **BC/DAgger baseline**: the best current conditioned student, repaired only
   through tracked changes. Include behavior-balanced sampling and DAgger on
   student states.
2. **Direct PPO baseline**: initialize from scratch in a two/three-behavior
   superset environment using the legal graph only. This tests whether reward,
   masking, reset routing, and transition sampling are learnable without
   teacher initialization.
3. **Hybrid PPO baseline**: initialize the actor from the best BC/DAgger model,
   then fine-tune with the same environment and schedule.

For hybrid training, add an adaptive teacher-KL anchor per active behavior as
 a measured experiment, not a permanent assumption. The anchor should weaken
 only after the candidate exceeds the corresponding teacher return or task
 threshold, allowing PPO to improve transitions and outcomes.

Every run starts with the 64-env, 5-iteration smoke test. Keep critic and
behavior sampling changes separate between runs.

#### P2 amendment: supervised hybrid curriculum (2026-09-03)

The initial PPO diagnostics showed that loading a BC actor once is not enough:
the actor can leave already-validated specialist behavior before it has learned
the shared task. This amendment makes the teacher anchor an explicit, measured
hybrid experiment while preserving the direct-PPO comparison.

1. **Direct PPO remains from scratch.** It must not use teacher actions, teacher
   features, or teacher losses; otherwise the baseline no longer measures
   learnability without initialization.
2. **Hybrid anchor uses the available teacher contract.** The frozen specialists
   are deterministic action policies (checkpoint/ONNX), so the first
   implementation uses per-step action imitation,
   `L_anchor = mean((a_student - a_teacher)^2)`, on the active behavior. A
   distributional KL may be added only when a validated teacher distribution is
   available; it is not required for this experiment.
3. **Anchor scope and schedule are explicit.** Apply the anchor during behavior
   hold windows and ordinary specialist states. Reduce it during legal
   transition windows so the student can learn handoff dynamics. Decay it only
   after the active behavior reaches 90% of its teacher success/return target;
   log anchor weight and per-behavior anchor error every iteration.
4. **Unlock the task in stages.** Start with `VELSTAND` hold/recovery, then add
   `VELOCITY` with only small forward and zero commands, then add `SITSTAND`
   hold/transition behavior, and only then enable all four legal edges. A stage
   unlocks from measured behavior success, not from elapsed iterations. The
   legal graph and unsupported direct edges remain unchanged.
5. **Use behavior/state-bucket resets.** During each stage, reset from the
   corresponding validated specialist distribution and balance recovery,
   command, and transition-phase buckets. Do not count a balanced behavior id
   as sufficient state coverage.
6. **Delay motion taxes.** During early skill discovery, set action-rate,
   body-angular-velocity, torque-rate, and rise/descent attempt-tax terms to
   their minimum measured weights. Restore them gradually after the relevant
   behavior passes its discovery threshold. All penalty sign and non-positive
   weighted-mass invariants still apply.

This is a diagnostic curriculum, not a change to the G0 acceptance contract.
Record each stage's seed list, unlock evidence, anchor schedule, reset-bucket
counts, and direct-vs-hybrid result. If the anchored hybrid still fails while
the specialist traces and isolated behavior stages pass, stop and diagnose
transition coverage or schema/action alignment before adding any skill.

### P3. Evaluate G0 behavior and transitions

Run fixed-seed batteries matching the specialist evidence:

- stand hold and command-zero behavior;
- velocity command buckets, including stop and turn-in-place where supported;
- sit hold and stand hold;
- `VELSTAND -> VELOCITY -> VELSTAND`;
- `VELSTAND -> SITSTAND -> VELSTAND`;
- recovery buckets under `VELSTAND`;
- chained legal sequence with no reset.

Record per behavior and per edge: success, fall rate, tilt, height, command
tracking, settling time, peak action/target jump, episode length, contacts,
finite/NaN status, and displacement. Evaluate the model both through the
existing MuJoCo route and after ONNX export.

### P4. Decide whether to continue

Accept a candidate only when all of these hold:

- each included behavior is at least 90% of its specialist success rate, with
  the fixed-seed confidence method from the teacher manifest;
- stand, locomotion, and sit/stand metrics are within 10% of baseline;
- recovery has no more than a 25% relative fall-rate increase from baseline;
- every legal transition reaches its destination without reset and has at least
  90% success;
- no unproven direct edge is silently exercised or reported as supported;
- no non-finite path, positive penalty, action-range violation, or ABI mismatch;
- PyTorch and ONNX agree on golden vectors and metadata;
- inference fits the 50 Hz budget with margin;
- specialists remain installable as fallback.

If direct PPO fails but hybrid passes, continue with hybrid for the next skill.
If both fail, stop and diagnose schema, reward masking, transition sampling,
or state coverage before adding any behavior. If BC/DAgger passes but PPO
regresses, retain the distilled model as the G0 candidate and treat PPO as a
failed optional refinement.

The hybrid implementation now includes a measured per-step action anchor:
frozen Torch teachers, metadata-aware rollout storage, same-step PPO anchor
loss, per-behavior anchor weights/errors, and an explicit threshold-gated
schedule hook. These changes are implementation-complete but do not alter the
acceptance thresholds below.

## Unified PPO reward contract

Do not use an unconditional sum of every specialist reward. The G0 environment
uses a per-environment active behavior and masks task terms accordingly:

```text
r = r_common_balance_safety
  + mask(VELSTAND)  * r_stand_recovery
  + mask(VELOCITY)  * r_velocity_tracking
  + mask(SITSTAND)  * r_posture_target
  + r_transition
```

`r_transition` is active only in boundary windows and must reward successful
handoff/settling, not merely spending time in a qualifying pose. Reward logs
must report weighted mass separately by behavior and term.

The training scheduler samples legal edges and dwell times from the frozen
graph. Random command changes are not considered transition coverage.

## Implementation ownership

Expected repository surfaces:

- `src/mjlab_microduck/generalist_schema.py` and `generalist_model.py`:
  contract, model, and metadata;
- `scripts/collect_generalist_dagger.py` and related dataset tooling:
  teacher/replay/transition collection;
- new `src/mjlab_microduck/tasks/microduck_generalist_g0_env_cfg.py` and a
  training-only transition router, composed from proven MDP functions;
- `scripts/train_generalist_bc.py`, plus a new explicitly named PPO runner;
- `scripts/rollout_generalist_bc.py` or a sibling canonical G0 evaluator;
- focused tests for graph legality, reward masking, replay determinism,
  behavior balance, transition accounting, and ONNX metadata/parity.

Do not edit specialist task contracts, official runtime state-machine code, or
roller configuration as part of this plan.

## Verification commands

```bash
uv run --with pytest pytest tests/test_generalist_schema.py \
  tests/test_generalist_model.py tests/test_collect_generalist_dagger.py
uv run train <G0_TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5
uv run --with pytest pytest tests/
```

The final report must include manifests, commands, hashes, seed lists, per-case
metrics, transition support/unsupported edges, videos, and the direct-vs-
hybrid decision. Generated datasets, checkpoints, and videos stay out of Git.

Evaluator contract correction: the canonical G0 evaluator now runs at 50 Hz
(four 5 ms physics steps per control action), follows the frozen Track A dwell
and command schedule, supplies trained phase/posture conditioning, delays the
SITSTAND-to-VELSTAND ownership handoff until the six-second rise window, and
excludes source preparation from edge scoring. Metrics snapshot positions and
enforce velocity displacement plus sit/rise height outcomes in addition to
finite, tilt, and action-range gates.

## Risks and explicit stop conditions

- **False transition assumption:** a demo sequence is not proof of arbitrary
  pairwise switching. The graph and unsupported-edge tests are mandatory.
- **Behavior interference:** if one condition improves while another regresses,
  do not add capacity or skills blindly; inspect balance, normalization,
  action scale, and boundary data first.
- **Reward farming:** any positive term that can be held in a fallen or bad
  state is a blocker; replace it with potential/progress or state-gated logic.
- **PPO instability:** retain the BC/DAgger artifact and stop PPO escalation if
  it fails to improve task metrics without regressions.
- **Sim/runtime mismatch:** no runtime integration or hardware claim follows
  from G0 simulation alone.

The plan is complete when one no-wheel G0 candidate passes the full acceptance
gate, or when the evidence demonstrates that this proven transition subset
cannot be merged with the current observation/action and reward contracts.
