# IsaacLab to MJLab Alignment Handoff

Status: ready for a new implementation context

Date: 2026-09-11

## Purpose

The target is to make IsaacLab follow the existing MJLab Velocity-Flat
semantics as closely as the backend allows. MJLab remains the canonical
reference. This document does not propose changing the MJLab recipe to match
PhysX.

The BAM actuator is tracked separately. The IsaacLab BAM voltage, delay, sag,
and friction mathematics already match the reference implementation to floating
point precision. The remaining BAM issue is the PhysX-side application of
load-dependent friction and its same-step force timing.

This document covers the other material differences and defines the next
bounded experiments.

## Current Evidence

- Actor observation ABI is 61D, privileged critic ABI is 76D, and action ABI is
  14D on both task paths.
- HOME, action scale, joint order, target delay, voltage sag, commands,
  observations, rewards, terminations, reset behavior, and domain-randomization
  contracts have deterministic fixtures and are currently marked MATCHED.
- The accepted MJLab policy loaded into the IsaacLab runner passes the common
  six-case battery. This proves that the IsaacLab observation/action/task path
  can execute a walking policy.
- The strict IsaacLab checkpoint still fails forward and lateral command
  response. This remains a trained-behavior failure, not proof that the policy
  ABI is wrong.
- The fixed-root dynamics report shows IsaacLab is closer to MuJoCo's
  motor-only reference than to MuJoCo with BAM solver friction:
  `.cache/isaaclab-assets/fixed_root_dynamics_parity_1x12.json`.

## Difference 1: Physics Solver

### What differs

MJLab uses MuJoCo `implicitfast` with its configured solver and contact
iterations. IsaacLab uses the PhysX GPU solver. The current converted asset uses
articulation solver position/velocity iteration counts of `4/1`; the MJLab
reference uses a different solver/contact configuration.

This changes contact impulses, joint constraint stabilization, static friction,
energy dissipation, and the exact trajectory even when all task parameters are
identical.

### Can it be aligned?

Not exactly. PhysX cannot be made trajectory-identical to MuJoCo by copying
iteration counts. Solver settings can reduce the difference, but they cannot
remove the difference in solver formulation and contact handling.

The useful target is behavioral calibration, not byte-identical trajectories:

1. Keep the MJLab solver unchanged.
2. Run a fixed-root, fixed-target joint response grid with neutral DR.
3. Sweep PhysX position/velocity iterations and stabilization settings.
4. Select the PhysX setting that minimizes the measured joint-position,
   velocity, contact-force, and energy errors without damaging the walking
   battery.

### Acceptance

The solver experiment must report numerical errors for step and sine targets,
not only a visual result. It must also run the six-case battery with the same
checkpoint and reset instrumentation. Solver tuning is accepted only if it
improves the fixed-root metrics without making contact or walking behavior
worse.

Do not change rewards or PPO parameters to compensate for this difference.

## Difference 2: Native Damping and Friction Path

### What differs

MJLab's BAM `edit_spec` zeros native MuJoCo joint damping and friction loss,
then writes the actuator-computed values into MuJoCo solver fields every step.
IsaacLab's converted USD must likewise avoid importing a second hidden joint
friction budget. The current asset configuration explicitly sets native
friction and damping to zero and records the BAM viscous coefficient in the
actuator metadata.

This part is mostly aligned at startup and reset time. The remaining semantic
gap is not the authored USD number; it is whether the runtime PhysX joint
friction fields are updated with the same BAM budget and timing. That is tied to
the BAM friction bridge and must not be solved by re-enabling the USD's
`0.0048` nominal friction.

### Can it be aligned?

Mostly yes at the asset and runtime configuration level:

- keep USD/native joint friction at zero under BAM;
- keep native damping at zero under BAM;
- keep armature explicit and match the MJLab value;
- apply the BAM viscous and velocity-independent budgets through the IsaacLab
  actuator/PhysX bridge;
- assert the runtime tensors after reset and during a controlled friction
  sweep.

The full load-dependent runtime application is covered by the separate BAM
workstream. Until that bridge exists, the honest status is `PARTIAL`, not
`MATCHED`.

### Acceptance

The asset dynamics probe must continue to show finite, zero native friction and
damping under BAM, with armature and friction-scale changes observable. A
controlled fixed-root sweep must distinguish these three cases:

1. native PhysX friction only;
2. motor-only BAM path;
3. load-dependent BAM bridge.

No long training is valid if a hidden imported USD friction value is active.

## Difference 3: RSL-RL Implementation Version

### What differs

The accepted MJLab training uses `rsl-rl-lib 5.0.1`. The IsaacLab 3.0.0 image
currently supports `rsl-rl-lib 5.4.1`; a direct attempt to force 5.0.1 into the
IsaacLab entrypoint was incompatible.

The scalar PPO configuration is matched, but the trainer implementation is not
byte-identical. Differences may include distribution construction, optimizer
details, storage behavior, logging, and update ordering.

### Can it be aligned?

Possibly, but this is a software compatibility investigation rather than a
task-config change. Do not change the repository's MJLab dependency or silently
install a second RSL-RL version.

The next step is a controlled compatibility probe inside the pinned IsaacLab
container:

1. record the exact package versions and entrypoint imports;
2. test whether an IsaacLab-compatible runner can be built against 5.0.1 in an
   isolated environment;
3. compare one rollout/update on identical tensors between 5.0.1 and 5.4.1;
4. retain 5.4.1 if the entrypoint or tensor contract is incompatible.

### Acceptance

Version unification is useful only if the runner imports and the one-update
comparison passes. Otherwise retain 5.4.1 as an explicit backend delta and
focus on physics and sensor semantics. Do not launch another 6000-iteration
run solely to test a version hypothesis.

## Difference 4: Sensor Source and Measurement Pipeline

### What differs

MJLab reads the named IMU and other model sensors from MuJoCo. IsaacLab mostly
adapts root-state tensors and reconstructs some model quantities from rigid-body
data and canonical MJCF offsets.

The processing contract is already close: mounting misalignment, noise, delay,
joint encoder bias, foot-site offsets, contact terms, and critic dimensions are
covered by runtime probes. The raw measurement source is still different, so
solver-dependent effects can enter the policy observation even when the final
61D vector has the same shape.

### Can it be aligned?

The processing pipeline can be aligned. The raw source can only be made exact
if the USD/PhysX asset exposes equivalent named sensors and the IsaacLab task
reads those sensors instead of root-state adapters.

Prioritize the signals that affect walking:

1. gyro and projected gravity frame and timing;
2. joint position and velocity encoder view;
3. foot height, contact, and foot velocity;
4. privileged critic contact-force and air-time inputs;
5. subtree angular momentum.

For each signal, compare clean raw values first, then apply corruption, delay,
and normalization. Do not compare only the final policy observation because
normalization can hide a source mismatch.

### Acceptance

Use a seeded scripted trajectory and record raw MJLab and IsaacLab tensors at
the same control steps. The report must include per-signal max/mean error,
frame convention, delay index, and reset behavior. A signal is `MATCHED` only
when its source and processing semantics are both documented; finite output
alone is insufficient.

## Priority and Execution Order

1. **BAM PhysX bridge**: highest expected dynamics impact; tracked in the
   separate BAM document/workstream.
2. **Native damping/friction audit**: cheap guard against hidden duplicate
   friction; keep zero-value runtime assertions.
3. **Solver calibration**: run fixed-root and contact probes; tune only PhysX
   solver settings.
4. **Sensor raw-source comparison**: close any frame/timing mismatch before
   interpreting a training result.
5. **RSL-RL compatibility probe**: investigate in isolation; treat as medium
   priority unless the tensor/update comparison shows a concrete difference.
6. Only after these bounded probes pass: run a 64-env/5-iteration smoke, then a
   short checkpointed training experiment. A 6000-iteration run requires a
   single stated hypothesis and intermediate six-case batteries.

## Rules for the Next Context

- Do not modify the MJLab canonical recipe to make PhysX look better.
- Do not change rewards, command distributions, or PPO settings while testing
  solver, friction, sensor, or trainer hypotheses.
- Preserve the historical adapted IsaacLab profile as a separate diagnostic
  profile; do not merge it into the strict task.
- Treat total reward and episode length as secondary. The acceptance gate is
  forward/lateral/yaw/turn/zero behavior battery plus finite state and reset
  checks.
- Keep the existing strict and adapted checkpoints immutable. New experiments
  must use a new run directory and record the exact code/config/container
  identity.

## Primary References

- `docs/isaaclab_velocity_flat_parity_ledger.md`
- `docs/isaaclab_velocity_flat_backend_comparison.md`
- `docs/status/active/isaaclab-backend-implementation.md`
- `src/isaaclab_microduck/actuators/bam_actuator.py`
- `src/isaaclab_microduck/assets/microduck.py`
- `scripts/isaaclab/fixed_root_dynamics_parity.py`
- `scripts/isaaclab/asset_dynamics_runtime_probe.py`
- `scripts/isaaclab/velocity_flat_directional_trace.py`
