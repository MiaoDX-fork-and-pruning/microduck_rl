# Generalist v0 Execution Plan

Status: active; P0 complete, P1 bridge smoke passes, rollers full gate passes, walk full gate fails displacement.

This plan turns the current specialist evidence into two hardware-specific
generalist tracks. It deliberately keeps single-policy validation separate from
multi-policy composition and reuses the official `pollen-robotics/microduck`
controller for composition instead of creating a second scheduler here.

## Goal

Produce reproducible evidence and training inputs for:

- one merged model for the no-wheel/all-collisions robot;
- one merged model for the roller robot;
- official-controller-driven combination tests for each compatible scene;
- specialist ONNX files retained as immutable teachers and fallback.

## Non-goals

- No cross-scene transition between the walk and roller models.
- No replacement of the production 61D specialist ABI.
- No Python reimplementation of the official controller state machine.
- No runtime default change or hardware rollout in this plan.
- No merged model training before every included specialist passes the single-
  policy battery.

## Authority And Contracts

The official controller owns policy selection, command encoding, action windows,
priority, filtering/gain, and safety fallback for combination tests. This repo
owns MuJoCo scenes, deterministic evaluation, teacher data, model training, and
golden vectors. A thin bridge may translate sensors/intents and return targets,
labels, and metrics, but must not duplicate scheduler decisions.

The specialist contract remains `61D -> 14D`. The generalist uses the versioned
conditioned contract already described as `generalist-v0` / schema v2 (nominally
71D), with an explicit hardware profile field or profile-specific manifest. The
same condition layout may be shared by both models; the ONNX files, scene,
joint/passive layout, normalization, and acceptance baselines remain separate.

Unknown behavior IDs, profile mismatches, unsupported transitions, missing
fallbacks, non-finite actions, and failed model loads must fail closed. They may
not silently route to a policy from the other hardware family.

## Phases

### P0. Single-specialist action battery

Extend the existing deterministic evaluator rather than introducing another
parity framework. Run each accepted specialist independently in its own scene
and reset state. For locomotion, sweep command values
`0.03, 0.05, 0.08, 0.10, 0.15, 0.20 m/s`; test both direct-step and command-EMA
inputs. Evaluate stand, locomotion, sit/rise, ground pick, kick, roulade, and
roller crouch where compatible.

Record fixed seeds, profile, policy/checkpoint/ONNX hashes, command/phase, raw
and applied action, trunk pose/tilt/height, world displacement, contacts,
termination, and success metrics. Add a machine-readable report with per-case
and per-behavior gates; do not accept an aggregate score that hides a failed
skill.

Stop gate: every teacher used by a generalist track has finite 61D/14D parity,
an independently reproducible report, and an explicit pass/fail outcome. A
failed teacher is fixed or excluded before composition work.

### P1. Official-controller combination bridge

Current evidence: official fork branch `test/official-fall-replay` at commit
`66d4fa8facd4c564f8346dc54f361ceaa5e28d59` has persistent replay coverage for the fall chain, independent long
walk and roller command sequences, and an explicit `robotd --replay-stdin`
NDJSON transport. The transport consumes official 15-joint sensor frames and
publishes targets, gains, torque writes, and authoritative `RobotState` frames
without changing the production startup path. The MuJoCo driver now maps the
14 policy joints explicitly around the runtime mouth slot and passes a 20-tick
smoke for both profiles. With homing physics frozen as an explicit bring-up
boundary, the 300-active-tick battery reaches official `walk`/`stand`
transitions. At the accepted `0.20 m/s` walk command, both profiles pass their
stability/displacement gates; walk also records an official fall and recovery
chain. P1 is ready for final verification before P2.

Pin an official `microduck` commit and record the exact controller API/schema.
Build the smallest headless bridge needed to run the official controller against
MuJoCo. The bridge may be NDJSON/stdin-stdout initially; it must be persistent
so controller state (`last_action`, filters, command EMA, skill timers, and
fall-recovery state) is not reset between frames.

Run Track A and Track B separately, using only transitions compatible with the
scene/profile. Report active policy label, effective command, gain/scale,
target discontinuity, switch latency, reset count, tilt, displacement, fall and
recovery. Overlay the controller label in the video, not a label inferred from
the scenario file.

Stop gate: a replay proves deterministic controller decisions and state across
one transition; official commit, bridge version, model hashes, and unsupported
edges are recorded. No claim of official-controller parity is valid without this
evidence package.

### P2. Walk merged model

Use the no-wheel teacher subset: stand/velstand, velocity, sitstand, ground pick,
kick, and roulade only after P0 passes. Collect balanced teacher data in the
all-collisions scene, including transition frontiers and recovery states. Train
the conditioned student with behavior/profile fields and retain specialists as
teachers and fallback.

Order: schema/golden vectors -> offline BC -> student-state/DAgger -> optional
small PPO fine-tune. Run the P1 official-controller composition battery after
each candidate; do not advance on aggregate loss alone.

### P3. Roller merged model

Run the same sequence independently on the rollers scene, starting with
`velocity_rollers` and `roller_crouch`, then add only skills proven compatible
with that model. Use the roller-specific command ranges and measured action
scale/filter contract. Never reuse walk-scene teacher states or transition
labels as roller data.

The roller candidate has its own teacher manifest, normalization statistics,
ONNX metadata, baseline report, and P1 composition battery. A walk candidate
cannot satisfy a roller gate, and vice versa.

### P4. Handoff, fallback, and release evidence

Package per-profile ONNX metadata, schema vectors, controller commit, scene/model
IDs, exact commands, baseline comparisons, and fallback matrix for the official
runtime repository. Verify shadow/canary semantics before any default switch:
specialists actuate in shadow, merged model logs; canary explicitly selects one
profile; rollback leaves specialists installable.

## Acceptance Gates

- P0 passes independently for every included teacher and command bucket.
- P1 has deterministic official-controller replay for both profiles.
- Walk and roller candidates pass separate per-skill and transition gates.
- No profile or behavior mismatch silently falls back across hardware families.
- Exported ONNX matches PyTorch on fixed vectors and carries profile/schema
  metadata.
- No non-finite path, unsupported transition, or hidden reset is accepted.
- Specialists remain available as fallback until repeated shadow/canary evidence
  supports a release decision.

## Execution Queue

| Order | Phase | First deliverable | Primary decision |
|---:|---|---|---|
| 1 | P0 | single-policy battery + report schema | which specialists are actually usable |
| 2 | P1 | pinned official bridge + one-transition replay | can official scheduling drive our ONNX |
| 3 | P2 | walk conditioned BC baseline | can one walk model imitate its teachers |
| 4 | P3 | roller conditioned BC baseline | can one roller model imitate its teachers |
| 5 | P4 | runtime handoff package | is shadow/canary integration ready |

Each phase is implemented and reviewed independently. A failed stop gate sends
work back to the owning phase; it does not justify adding scheduler logic to a
different phase.
