# IsaacLab to MJLab Alignment Handoff

Status: T26 current train/restore/evaluate verification complete; T23
lagged-friction hypothesis rejected; T24 sensor trace and T25 one-update
RSL-RL probe complete; ready for the next bounded alignment or deployment-
validation context

Date: 2026-09-12

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

## Follow-up: MJLab adaptive curriculum research

The fixed MJLab recipe remains the canonical parity baseline. The research
record and the proposed MJLab-only adaptive curriculum experiment are kept in:

- `docs/adaptive_curriculum_deep_research.md`
- `docs/mjlab_adaptive_curriculum_proposal.md`

The proposal is intentionally separate from IsaacLab backend alignment. It
preserves the final canonical task semantics and uses frozen command-conditioned
capability probes, one-axis advancement, hysteresis, checkpoint preservation,
and rollback. It must not be confused with RSL-RL's
`algorithm.schedule="adaptive"`, which controls PPO learning-rate/KL behavior.

## Current Evidence

- Actor observation ABI is 61D, privileged critic ABI is 76D, and action ABI is
  14D on both task paths.
- HOME, action scale, joint order, target delay, voltage sag, commands,
  observations, rewards, terminations, reset behavior, and domain-randomization
  contracts have deterministic fixtures and are currently marked MATCHED.
- The accepted MJLab policy loaded into the IsaacLab runner passes the common
  six-case battery. This proves that the IsaacLab observation/action/task path
  can execute a walking policy.
- T22 independently trained strictification checkpoints `model_500.pt`,
  `model_750.pt`, and `model_999.pt` pass the final strict live-manager
  six-case battery with zero resets. This closes the strict trained-policy
  gate without changing the canonical MJLab recipe.
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
6. T23's one-step-lag friction bridge is rejected for production after the
   canonical 300-step `turn_left` failure; retain it only as a diagnostic.
7. Only after a new bounded hypothesis is stated: run a 64-env/5-iteration
   smoke, then a short checkpointed training experiment. A 6000-iteration run
   requires intermediate six-case batteries.

## Continuation Results (2026-09-11)

The native damping/friction audit was rerun in the pinned IsaacLab 3.0.0 /
Isaac Sim 6.0.1 image with four environments. The runtime joint damping and
friction tensors are finite and exactly zero; reset-time armature and CoM DR
change, while startup mass/inertia remain stable. Evidence:
`.cache/isaaclab-assets/asset_dynamics_runtime_probe_handoff_4.json`.

The fixed-root harness was extended with per-run PhysX solver iteration
overrides and solver metadata in its report. With the canonical `4/1`
setting, the fresh 40-step result has step/sine `qdot` errors versus the
MuJoCo BAM-solver reference of `0.4020/0.2396` rad/s, while errors versus the
motor-only reference are `0.0856/0.0335` rad/s. An `8/2` point is effectively
unchanged and slightly worse (`0.4041/0.2405` vs BAM; `0.0824/0.0322` vs
motor-only). Evidence:
`.cache/isaaclab-assets/fixed_root_dynamics_handoff_4x1.json` and
`.cache/isaaclab-assets/fixed_root_dynamics_handoff_8x2.json`.

Decision: retain production PhysX solver settings at `4/1`. The sweep does
not provide a calibration win, and the remaining trajectory gap is the
expected solver formulation plus same-step external-load friction timing
delta. No reward, PPO, or long training run should be changed on this basis.

The current checkout also passed the required IsaacLab training bring-up. A
64-env/5-step runtime smoke reported finite 61D observations, 14D actions,
15 finite contact bodies, and zero resets in
`.cache/isaaclab-assets/velocity_flat_smoke_handoff_current_64.json`. The
official IsaacLab RSL-RL entrypoint then completed five PPO iterations at 64
envs, writing `model_0.pt` and `model_4.pt` under
`logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_09-47-26/`
with finite losses and no `nan_state` terminations. Reloading `model_4.pt`
through the fixed command battery remained finite with zero resets; its
translation/yaw failures are expected for an untrained five-iteration policy
and are not an acceptance result. Evidence:
`.cache/isaaclab-assets/velocity_flat_command_battery_handoff_model4.json`.

A fresh 24-step directional trace of strict `model_750.pt` further isolates
the failure: command tails are correct (`[0.2, 0, 0]` and `[0, 0.2, 0]`), raw
actions remain nonzero (mean absolute action `0.94-0.98`), and BAM efforts are
finite/nonzero. The rollout nevertheless accumulates resets while falling,
including negative root vertical velocity. Evidence:
`.cache/isaaclab-assets/velocity_flat_directional_trace_strict_model750_handoff.json`.
This is a learned stabilization/dynamics failure, not command-slot or action
suppression, so no ABI or reward workaround is justified by this trace.

As a cross-profile control, adapted `model_750.pt` was loaded into the strict
task configuration without changing the environment. All six 100-step cases
passed with zero resets: forward averaged `0.139 m/s`, lateral `0.059 m/s`,
yaw `0.431 rad/s`, and maximum tilt stayed below `0.11 rad`. Evidence:
`.cache/isaaclab-assets/velocity_flat_command_battery_adapted750_on_strict_handoff.json`.
This proves the strict task/runtime path can execute the walking behavior; the
remaining strict failure is in the strict training recipe/optimization window,
not an IsaacLab task ABI or termination incompatibility.

## Bounded strict warm-start experiment (2026-09-11)

The official IsaacLab/RSL-RL resume path was verified from the pinned runtime:
`OnPolicyRunner.load` restores the policy, normalizers, optimizer state, and
serialized iteration before continuing in a new run directory. A bounded
250-iteration continuation loaded the immutable adapted `model_750.pt` into
the unchanged strict task at 4096 environments. The run resumed at iteration
750 and completed through iteration 999 without NaNs:

`logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_10-20-16_strict_warmstart_adapted750_250it/`

The resulting strict-task `model_999.pt` passes the standard six-case battery
for 300 steps, 16 environments, and seed 2026 with zero resets and finite
state. Forward and lateral means are `0.153/0.081 m/s`, yaw is `0.314 rad/s`,
and maximum tilt is `0.570 rad` (all cases pass their configured gates).
Evidence: `.cache/isaaclab-assets/velocity_flat_command_battery_warmstart_adapted750_strict_999_300.json`.

The checkpoint has the expected 61D actor/76D critic schema and finite
normalizers (`velocity_flat_checkpoint_audit_warmstart_model999.json`). It was
exported through the official IsaacLab play path to
`.../strict_warmstart_adapted750_250it/exported/policy.onnx`; ONNX checker and
CPU ONNX Runtime validate `[1,61] -> [1,14]` with finite output. A bounded
MuJoCo/Xvfb rehearsal also loaded the graph and produced finite actions. The
warm-start provenance stays distinct from an independently strict-trained run.

The new headless CPU MuJoCo battery consumes the same six-case spec and closes
the directional runtime gate for this candidate. All six cases pass for 300
control steps with finite 61D/14D tensors, zero resets, and maximum tilt below
`0.08 rad`. Body-frame means are forward `0.101 m/s`, lateral `0.061 m/s`,
yaw `0.299 rad/s`, turn-left `-0.512 rad/s`, and turn-right `0.434 rad/s`.
Evidence: `.cache/isaaclab-assets/mujoco_onnx_battery_warmstart_model999.json`,
generated by `scripts/isaaclab/mujoco_onnx_battery.py`.

## Independent strict checkpoint gate (2026-09-11)

The existing independently trained strict checkpoint was evaluated with the
same IsaacLab six-case battery before considering any new training run. The
`model_250.pt` checkpoint from the corrected strict-from-scratch run is finite,
but it fails the behavior gate in every case: reset fraction is about `0.16`
per environment step, the zero command drifts at roughly `-0.219 m/s`, and
the commanded cases do not produce stable translation or yaw response. The
checkpoint therefore does not close the independent-training requirement.

Evidence: `.cache/isaaclab-assets/velocity_flat_command_battery_strict_model250.json`
(SHA256 `ba017cc70e3e4605048bc0e469a1c9ad7c7948c761dd292872d6fcc9f94a807e`).
This result reinforces the existing diagnosis: the strict task/runtime path is
finite and executable, while the from-scratch optimization trajectory does not
yet learn stabilization. Do not start another long run or alter rewards/PPO
without one bounded, explicitly stated training hypothesis.

## Bounded air-time hypothesis (2026-09-11)

As one bounded training calibration, the strict task was trained for `750`
iterations at `4096` environments with only `air_time.weight=1.0`; commands,
DR, terminations, PPO, and all other rewards were unchanged. The run remained
finite and recovered to roughly `950/1000` mean episode length, but its endpoint
`model_749.pt` still fails forward/lateral response. Zero, yaw, and turn cases
pass, while the policy settles near `0.071 m` root height. Evidence:
`.cache/isaaclab-assets/velocity_flat_command_battery_strict_airtime1_model749.json`.
This rejects early air-time reward dominance as the complete cause and keeps
the independent strict-training gate open.

## Bounded root-height bootstrap hypothesis (2026-09-11)

The second bounded training calibration disabled only the strict
`env.terminations.root_height` term for `750` iterations at `4096` environments;
reward terms, commands, DR, and PPO were unchanged. The endpoint
`model_749.pt` was evaluated under the unchanged strict task with the height
guard restored. It remained finite but failed all six battery cases: the zero
case accumulated `776` resets (reset fraction `0.1617` per env-step) and drifted
at `-0.199 m/s`; commanded translation and yaw also failed. The checkpoint's
mean root height was about `0.101 m`, with a minimum of `0.055 m`.

Evidence:
`.cache/isaaclab-assets/velocity_flat_command_battery_strict_rootbootstrap_model749.json`
(SHA256 `3d764508b657d657346c990edd5f7cb882b8d6b28a31262c53e182773118c256`),
from run
`logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_12-21-27/`.
This rejects low-height termination as the bootstrap explanation. The
training-only override is not a canonical task change, and the independent
strict-training gate remains open.

## Bounded action-rate hypothesis (2026-09-11)

The third bounded training calibration used a separately registered strict
diagnostic task with only the `action_rate_l2` curriculum frozen at `-0.1`.
The run completed `750` iterations at `4096` environments with finite losses,
no NaN terminations, and the live curriculum value fixed at `-0.1000` through
the endpoint. Its checkpoints were evaluated under the unchanged canonical
`IsaacLab-Velocity-Flat-MicroDuck` task. `model_250.pt` passed zero, forward,
lateral, and left-turn cases with zero resets, but failed yaw and right-turn;
`model_500.pt` and `model_749.pt` remained reset-free but lost most directional
response. The endpoint settled near `0.059 m` root height.

Evidence artifacts:
`.cache/isaaclab-assets/velocity_flat_command_battery_t13_action_rate_flat_model250.json`,
`.cache/isaaclab-assets/velocity_flat_command_battery_t13_action_rate_flat_model500.json`,
and
`.cache/isaaclab-assets/velocity_flat_command_battery_t13_action_rate_flat_model749.json`.
The endpoint checkpoint SHA256 is
`13fb9ae10437b3b431be3adcb8fef974388802de8134474c3d963967d4f3d41b`; the
resolved one-stage config SHA256 is
`aa427aeec0b6057f84c033d524738370a159ca3e7bba360a93152d38d6ed16e3`.
This rejects the action-rate curriculum as a complete explanation, while
showing that the fixed penalty improves early stabilization and creates a
shorter useful behavior window.

## Bounded angular-tracking hypothesis (2026-09-11)

The fourth bounded training calibration changed only strict
`env.rewards.track_ang_vel.weight` from `2.0` to `6.0`. Commands, strict
action-rate stages, terminations, DR, PPO, and every other reward remained
unchanged. The run completed `750` iterations at `4096` environments with
finite losses and no NaN terminations:

`logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_13-55-29_t14_angvel6_750/`

All three checkpoints were evaluated under the unchanged canonical strict task
with the common six-case battery (`16` environments, `300` steps per case).
`model_250.pt`, `model_500.pt`, and `model_749.pt` are finite and pass the
zero, yaw, turn-left, and turn-right cases with zero resets (the endpoint has
one negligible lateral reset), but all three fail forward and lateral command
response. This rejects increased angular tracking as a complete explanation;
the independent strict-from-scratch acceptance gate remains open.

Evidence artifacts:

- `.cache/isaaclab-assets/velocity_flat_command_battery_t14_angvel6_model250.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t14_angvel6_model500.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t14_angvel6_model749.json`

Checkpoint hashes are `model_250.pt` `26e41e40fe0f8093a33c4dfe116b9be9ee4c1088a07847c5ae1aab10f300e895`,
`model_500.pt` `84ed137389366ffb5e6065e1c7cf2b57bee21d3e18be72e46caf5bb54cf6e4e6`,
and `model_749.pt` `f369700d05fcbc39c57a1640d298a75d2bd6c84c3c42d88027daebb8e205a701`.
The resolved training environment YAML SHA256 is
`f031a87cba32d0edbd3707d32bab2d19f85684ddeb09ec1da0c271aef73c33b7`.

## Bounded linear-tracking hypothesis (2026-09-11)

The fifth bounded training calibration changed only strict
`env.rewards.track_lin_vel.weight` from `2.0` to `4.0`. The `4096`-environment
run completed `750` iterations with finite losses and no NaN terminations.
Under the unchanged canonical strict battery, `model_250.pt`, `model_500.pt`,
and `model_749.pt` all pass zero/forward/lateral/turn-left, but fail yaw and
turn-right. This is the complementary result to T14: stronger linear tracking
recovers translation while positive-yaw response is lost.

Run directory:
`logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_14-34-25_t15_linvel4_750/`

Evidence artifacts:

- `.cache/isaaclab-assets/velocity_flat_command_battery_t15_linvel4_model250.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t15_linvel4_model500.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t15_linvel4_model749.json`

Checkpoint hashes are `model_250.pt` `1c7014ddab1fbd27c2ab2bef7af714148610d41b9340fbf423aa912ee8a466c7`,
`model_500.pt` `3f9399948fc9b7d2c2dc64395c854290a36e5ff194880afa29e606967f9f373a`,
and `model_749.pt` `325e77a5b3e9ea38de2aeaa580355847315b6527600f271a9a382a2cbdec8193`.

## Bounded tracking-pair hypothesis (2026-09-11)

T16 changed only the adapted tracking pair in the strict task:
`track_lin_vel.weight=4.0` and `track_ang_vel.weight=6.0`. The run completed
`750` iterations at `4096` environments with finite losses and no NaN
terminations. Under the unchanged strict battery, all checkpoints pass
zero/forward/lateral; positive-yaw response remains absent, while turn-left is
present from `model_500.pt` onward. This rejects the tracking pair alone as a
complete solution.

Run directory:
`logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_15-09-43_t16_tracking_pair_750/`

Evidence artifacts:

- `.cache/isaaclab-assets/velocity_flat_command_battery_t16_tracking_pair_model250.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t16_tracking_pair_model500.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t16_tracking_pair_model749.json`

## Bounded symmetry hypothesis (2026-09-12)

The T17 diagnostic added left-right data augmentation using a new IsaacLab
61D mirror transform derived from the existing MJLab tables. The canonical
runner remains unchanged; CPU involution/sign contracts and a five-iteration
IsaacLab smoke pass. T17 retained the T16 tracking pair and completed `750`
iterations at `4096` environments with finite losses and no NaN terminations.
Under the canonical battery, `model_500.pt` passes zero/forward/lateral/
turn-left but fails yaw/turn-right; `model_749.pt` passes zero/yaw/turn-left/
turn-right but fails lateral. No checkpoint passes all six.

Run directory:
`logs/rsl_rl/microduck_isaaclab_velocity_flat_symmetry/2026-09-11_15-51-08_t17_symmetry_tracking_pair_750/`

Evidence artifacts:

- `.cache/isaaclab-assets/velocity_flat_command_battery_t17_symmetry_model250.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t17_symmetry_model500.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t17_symmetry_model749.json`

The independent strict-from-scratch gate remains open. T17 is rejected as a
complete solution, but the symmetry runner and transform are retained as a
reproducible diagnostic surface for any checkpoint-window follow-up.

## Bounded checkpoint-window continuation (T18, 2026-09-12)

T18 attempted to keep the T17 symmetry task and tracking pair unchanged and
overrode only `agent.save_interval=50` for a `750`-iteration,
`4096`-environment run. The
smoke passed, training completed normally with finite losses and no NaN
terminations, and all 16 checkpoints (`model_0` through `model_749`) were
replayed under the canonical six-case battery.

The pass vectors are ordered `(zero, forward, lateral, yaw, turn-left,
turn-right)`: `FFFFFF` at model 0; `PPPFFP` at model 50; `PPFFPP` at model 100;
`PPPFPF` at model 150 and models 350--600; `PPPFPP` at model 200; `PPPFFF` at model
250; and `PPFFPF` at model 300 and models 650--749. No checkpoint passed all six cases, so
T18 is rejected and the independent strict-from-scratch gate remains open.

Run directory:
`logs/rsl_rl/microduck_isaaclab_velocity_flat_symmetry/2026-09-11_16-34-04/`

Battery artifacts:

- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model0.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model50.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model100.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model150.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model200.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model250.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model300.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model350.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model400.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model450.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model500.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model550.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model600.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model650.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model700.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_t18_window_model749.json`

T18 changes observability only and does not alter the canonical strict task or
the MJLab recipe.

Post-run manifest inspection corrected T18's provenance: its `env.yaml` shows
the symmetry task's default tracking weights (`track_lin_vel=2.0`,
`track_ang_vel=2.0`), not T17's `4.0/6.0` pair. T18 is therefore an invalid
test of the T17 window and must not be used to close or reject that hypothesis.
The run is retained only as a finite default-pair diagnostic.

## Corrected checkpoint-window hypothesis (T19)

T19 is authorized as the corrected continuation. It must preserve T17's
symmetry task and exact tracking pair (`track_lin_vel=4.0`,
`track_ang_vel=6.0`), override only `agent.save_interval=50`, train 750
iterations at 4096 environments from scratch, and evaluate every checkpoint
under the canonical six-case battery. The independent strict-from-scratch gate
remains open until a checkpoint passes all six cases.

T19 completed with the exact T17 pair and dense checkpoint saves. The smoke
passed, training remained finite with no NaN terminations, and all 16
checkpoints were replayed under the canonical six-case battery. Pass vectors
(zero, forward, lateral, yaw, turn-left, turn-right) were `FFFFFF` (model 0),
`PPFFFF` (50), `PPPFFP` (100), `PPPPFP` (150), `PPPFFF` (200), `PPFPFF`
(250), `PPPFPF` (300--400 and 500), `PPPFPP` (450 and 550), and `PPFPPP`
(600--749). No checkpoint passed all six; T19 is rejected and the independent
strict-from-scratch gate remains open.

Run directory:
`logs/rsl_rl/microduck_isaaclab_velocity_flat_symmetry/2026-09-11_17-26-30/`

Battery artifacts:
`.cache/isaaclab-assets/velocity_flat_command_battery_t19_exact_pair_model{0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,749}.json`

## Rules for the Next Context

- Do not modify the MJLab canonical recipe to make PhysX look better.
- Do not change rewards, command distributions, or PPO settings while testing
  solver, friction, sensor, or trainer hypotheses.
- Preserve the historical adapted IsaacLab profile as a separate diagnostic
  profile; do not merge it into the strict task.
- Treat total reward and episode length as secondary. The acceptance gate is
  forward/lateral/yaw/turn/zero behavior battery plus finite state and reset
  checks.
- T22 strictification `model_999.pt` is the current independently trained
  cross-backend walking candidate; its IsaacLab and headless MuJoCo directional
  batteries pass. Retain the historical warm-start provenance separately.
- Keep the existing strict and adapted checkpoints immutable. New experiments
  must use a new run directory and record the exact code/config/container
  identity.
- T11 (reduced air-time reward) and T12 (training-only root-height guard
  removal) are rejected bounded hypotheses; neither recovered independent
  strict-from-scratch translation behavior.
- T13 (fixed action-rate curriculum), T14 (angular tracking weight `6.0`), and
  T15 (linear tracking weight `4.0`) are rejected as complete explanations;
  their checkpoint batteries remain diagnostic evidence only.
- T16 (tracking pair) and T17 (tracking pair plus symmetry augmentation) are
  rejected as complete explanations; their checkpoint batteries remain
  diagnostic evidence only.
- T18 is complete but invalid for the T17 window because its tracking-pair
  overrides were absent. T19 is complete and rejected; no other long run is
  authorized until a materially new bounded hypothesis is recorded. The
  independent strict-from-scratch gate was still open at that point because no
  checkpoint passed all six cases. T22 supersedes that historical state; its
  final strict live-manager profile closes the gate.
- T20 isolated the adapted command buckets (`rel_forward_envs=0.0`,
  `rel_lateral_envs=0.25`) behind a separate strict diagnostic task. Its smoke
  passed, but checkpoints 250/500/749 had pass vectors `PPFPFP`, `PPFFPF`, and
  `PFFFFP`; no all-six checkpoint passed. T20 is rejected. T21 combines those
  buckets with the existing symmetry augmentation in another separate
  diagnostic task; its 750-iteration battery pass vectors were `PPPFPF`,
  `PPFFPF`, and `PFFFPF`, so T21 is rejected. T22 tested an explicit
  adapted-to-strict curriculum from scratch and left the canonical task
  unchanged. T22 then completed with strictification active: model 250 was
  `PPFPFP`, while models 500, 750, and 999 were `PPPPPP`. The independent
  strict-from-scratch gate is closed by this result.
- T23 tested the reset-safe one-step-lag PhysX friction bridge. Fixed-root and
  100-step walking probes were finite and all-six passing, but the canonical
  300-step battery failed `turn_left` with one tilt reset. T23 is rejected for
  production parity; the bridge remains diagnostic-only and motor-only
  production behavior is unchanged.

## Continuation Probe (2026-09-12)

The PhysX force-timing diagnostic was tightened without changing production
task semantics. `scripts/isaaclab/force_timing_probe.py` now drives a nonzero
`0.35 rad` first-joint target and reports quantitative projected-force,
incoming-force, and actuation deltas across `after_write_before_step`,
`after_step_before_scene_update`, and `after_scene_update`. This prevents a
HOME-at-rest probe from hiding stale getter timing. Its source contract is
covered by `tests/test_isaaclab_force_timing_contract.py`.

The pinned raw `InteractiveScene` invocation was attempted, but this checkout's
direct asset harness still fails to resolve the articulation tensor view
(`.../Geometry/trunk_base/trunk_base`) before producing a new JSON artifact.
The existing task-based `force_timing_probe.json` remains the authoritative
runtime evidence. No friction bridge, reward, PPO setting, or training result
was changed from this probe. The gate status from this probe was superseded by
the T22 result below.

## T22 Strictification Result (2026-09-12)

T22 added a separate `IsaacLab-Velocity-Flat-MicroDuck-Strictification` task.
It starts from the measured adapted basin (`rel_forward_envs=0.0`,
`rel_lateral_envs=0.25`, tracking weights `4.0/6.0`, pose `0.5`, air-time
`1.0`, root-height threshold `0.0`, and action-rate `-0.1`) and mutates the
live command, reward, and termination managers to strict values at step 12000
(iteration 500). Later stages at iterations 750 and 1000 increase the
action-rate penalty to `-0.4` and `-0.6` while retaining strict command,
tracking, pose, air-time, and root-height values. The canonical task and its
reward recipe were unchanged.

The required 64-env/5-step smoke passed: actor observation shape `64 x 61`,
action dimension `14`, finite reset/steps/contact tensors, and zero resets.
The from-scratch 4096-env run completed 1000 iterations in
`logs/rsl_rl/microduck_isaaclab_velocity_flat_strictification/2026-09-11_19-26-41/`.
Training stayed finite with zero `nan_state` terminations. Under the canonical
six-case battery (zero, forward, lateral, yaw, turn-left, turn-right), the
pass vectors were:

- `model_250.pt`: `PPFPFP` (bootstrap; yaw and turn-right failed).
- `model_500.pt`: `PPPPPP` (first strictified all-six pass).
- `model_750.pt`: `PPPPPP` (later strict stage retained the pass).
- `model_999.pt`: `PPPPPP` (final strict stage retained the pass).

Battery artifacts are
`.cache/isaaclab-assets/velocity_flat_command_battery_t22_strictification_model{250,500,750,999}.json`,
and the smoke artifact is
`.cache/isaaclab-assets/velocity_flat_smoke_t22_strictification_64.json`.
The battery harness explicitly applies the final strict live-manager profile
before replay, so these vectors do not rely on the bootstrap stage.
This closes the independent strict-from-scratch gate. The accepted checkpoints
remain distinct from the historical adapted and warm-start checkpoints; no
cross-backend warm-start provenance was changed.

The official IsaacLab play path also exported the final T22 checkpoint to
`logs/rsl_rl/microduck_isaaclab_velocity_flat_strictification/2026-09-11_19-26-41/exported/policy.onnx`.
ONNX checker and CPU Runtime validation report finite `[1,61] -> [1,14]`
inference; the export SHA256 is
`371cb8d92300377363c89b5ada5c7c39683dd1464f93ea620c4c83859b3ce30e`.
The optional Kit video recorder could not run in this image because
`omni.replicator` is absent; this does not affect export or policy replay.
The exported graph passes the headless CPU MuJoCo six-case battery with zero
resets and max tilt below `0.068 rad` in
`.cache/isaaclab-assets/mujoco_onnx_battery_t22_strictification_model999.json`.

## T23 Lagged Friction Bridge Diagnostic (2026-09-12)

The next bounded backend experiment is now implemented as a diagnostic, without
changing the accepted task or the production `BamActuator`. The simulator-
independent `LaggedExternalEffort` buffer in
`src/isaaclab_microduck/actuators/physx_friction_bridge.py` stores one
post-step sample per environment and clears it on reset. It derives external
joint effort as
`get_dof_projected_joint_forces() - get_dof_actuation_forces()` and rejects
shape-mismatched or non-finite getter data.

`scripts/isaaclab/lagged_friction_bridge_probe.py` compares the existing
motor-only path with a one-step-lag path: force getters are sampled only after
`sim.step()` and `scene.update()`, then the sample is applied to the PhysX
friction fields on the following pre-step. The report records warm-up/reset
behavior, force magnitudes, coefficient values, finite-state checks, and the
joint-name order. This is a causal controller diagnostic, not BAM parity, and
is explicitly marked `production_wiring=diagnostic_only_not_applied`.

The CPU bridge and source contracts pass as part of the IsaacLab suite
(`150 passed`). The pinned Isaac Sim probe completed with six finite fixed-root
cases (motor-only and one-step-lag at friction scales `0.5/1.0/1.5`). All
cases report a zero reset buffer, a zero first-step external effort, 14 ordered
joint names, and finite projected/actuation-derived loads. Artifact:
`.cache/isaaclab-assets/lagged_friction_bridge_probe.json` (SHA256
`73a70df7303402e6a1ce78a79b57aa9881df0c97250fda3a8d19fb3deab99542`).

The walking follow-up was run against T22 `model_999.pt` with the canonical
16-environment, six-case command battery. At 100 steps, both the lagged and
production motor-only paths passed all six cases; the lagged report is
`.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_one_step_lag_100.json`
(SHA256 `b12e8e21e462d668e2104302a2441c825722aa685eb76382a0ae02364f45d17b`),
with the production comparison at
`.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_production_100.json`
(SHA256 `c2024a7195af01248061244d0466070bbc76c865c1163c2ec77121239b4bad81`).
At the canonical 300 steps, the lagged path failed `turn_left` with one tilt
reset (`max_tilt_rad=1.08818`, reset fraction `0.0002083`) while the other five
cases passed. The rejection artifact is
`.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_one_step_lag_300.json`
(SHA256 `ac4702d6b84b010149d1457a8f52df5cadfdf8d1a2de3e493aad3de0e1440263`).
T23 is rejected as a production BAM parity solution. The one-step-lag bridge
remains diagnostic-only; the accepted T22 task and production actuator remain
motor-only, with same-step solved external-load friction recorded as
`BACKEND_DELTA`. Do not tune the bridge to recover this checkpoint.

## Raw Sensor-Source Trace (2026-09-12)

The seeded 12-step source trace compares the same explicit root pose/velocity
and 14-joint pose/velocity sequence in both backends. The complete report is
`.cache/isaaclab-assets/sensor_source_comparison.json`; the raw inputs are in
`sensor_source_trace_mjlab.json` and `sensor_source_trace_isaaclab.json`.
Both traces use control-step delay index `0` and an explicit seeded reset; the
trace does not claim partial-reset coverage, which remains in the runtime
parity probes.

Measured results:

- root gyro adapter: max absolute error `8.94e-8 rad/s`;
- projected gravity adapter: max absolute error `3.58e-7`;
- policy-ordered joint position and velocity: exactly equal at float precision;
- reconstructed foot-site position: max absolute error `8.94e-8 m`;
- contact-found and contact-force outputs: equal zero no-contact values in the
  elevated scripted pose;
- foot-site velocity: max absolute error `8.84e-3 m/s`, retained as a PhysX
  rigid-body velocity-source delta;
- subtree angular momentum: max absolute error `7.99e-6` in the named-sensor
  comparison, retained as a source/reconstruction delta.

The trace closes frame/order/timing evidence for the matched signals. It does
not justify replacing the IsaacLab adapters with invented named sensors, and it
does not turn the foot-velocity or angular-momentum source deltas into reward
changes. The controlled RSL-RL compatibility probe follows below; no
additional long training run is justified without a new hypothesis.

## RSL-RL One-Update Probe (2026-09-12)

The pinned host/reference stack uses `rsl-rl-lib 5.0.1`; the IsaacLab 3.0.0
image uses `5.4.1`. Their runner source hash is identical, while PPO and
rollout-storage source hashes differ. A simulator-free synthetic probe ran the
same seeded 8-env, 4-step rollout and one PPO update in both stacks. Initial and
final actor/critic parameter hashes and all update metrics were identical:
`entropy=2.1806305051`, `surrogate=-0.0151484278`, and
`value=0.1463573426`. Artifacts:
`.cache/isaaclab-assets/rsl_rl_one_update_mjlab.json` and
`.cache/isaaclab-assets/rsl_rl_one_update_isaaclab.json`.

This closes the bounded compatibility question for the exercised feed-forward
configuration, but not byte-level equivalence for every optional RSL-RL path.
Keep `5.4.1` in the IsaacLab image because `5.0.1` is incompatible with its
entrypoint, and retain the version difference as an explicit training-stack
`BACKEND_DELTA`.

## Current IsaacLab Training Verification (2026-09-12)

The official IsaacLab entrypoint was rerun from the current checkout with
`64` environments and `5` PPO iterations. It completed in `6.5 s`, wrote
`model_0.pt` and `model_4.pt`, kept actor/critic shapes at `61/76`, and had no
`nan_state` terminations. The fresh smoke checkpoint is intentionally not a
walking-quality result: its 100-step reload battery is finite but fails tilt
and reset gates, as expected for five iterations.

The existing independently strict T22 `model_999.pt` was then reloaded through
the same current task and the canonical six-case, 300-step battery. All six
cases passed with zero resets, finite tensors, and max tilt below `0.18 rad`.
Evidence:

- `.cache/isaaclab-assets/velocity_flat_command_battery_current_smoke_model4.json`
- `.cache/isaaclab-assets/velocity_flat_command_battery_current_strict_model999_300.json`
- `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_21-07-34/`

This closes current-runtime train/restore/evaluate execution. It does not
claim that a five-iteration smoke checkpoint is a trained gait, and it does not
remove the documented PhysX solver/friction timing deltas.

## Deployment Validation (2026-09-12)

The accepted T22 strict checkpoint was validated through the deployment-side
ONNX path and the current IsaacLab live-manager path. The immutable inputs were
`logs/rsl_rl/microduck_isaaclab_velocity_flat_strictification/2026-09-11_19-26-41/model_999.pt`
(SHA256
`2034e7c3f6893f700de2d321746f1267e062aa3d3d29a8ceec4f3005e17e16ce`) and its
official export
`logs/rsl_rl/microduck_isaaclab_velocity_flat_strictification/2026-09-11_19-26-41/exported/policy.onnx`
(SHA256
`371cb8d92300377363c89b5ada5c7c3964dd1464f93ea620c4c83859b3ce30e`).

ONNX checker and CPU ONNX Runtime report the expected `obs [1, 61]` to
`actions [1, 14]` graph with finite output for a zero observation. The headless
CPU MuJoCo deployment battery ran all six shared cases for 300 control steps at
50 Hz: every case passed, all tensors were finite, there were zero resets, and
maximum tilt was `0.0678 rad`. Evidence:
`.cache/isaaclab-assets/deployment_validation_mujoco_model999_20260912.json`.

The same checkpoint was loaded by the current IsaacLab 3.0.0 / Isaac Sim 6.0.1
runtime using the production motor-only BAM path. Its 16-env, six-case,
300-step battery passed with zero resets and finite tensors; maximum tilt was
`0.1784 rad`. Evidence:
`.cache/isaaclab-assets/deployment_validation_isaaclab_model999_20260912.json`.

The deployment contract tests passed (`59 passed`), covering the unified 13D
command block, 61D observation construction, 14D action mapping, and runtime
policy routing. These results close simulator-side deployment validation. They
do not authorize hardware testing; the remaining gate is hardware transfer
review with the runtime team and a controlled robot test.

## Clean Canonical Curriculum Run (2026-09-14)

To isolate the canonical task from T22's adapted-to-strict bootstrap, a fresh
`IsaacLab-Velocity-Flat-MicroDuck` run was started from random initialization
with `4096` environments for `1000` PPO iterations. It used the production
motor-only BAM path and the stepwise curriculum mirrored from MJLab; no old
checkpoint and no `Strictification` task were used. The run completed at:

`logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-14_02-38-57/`

The 64-env canonical smoke passed (61D actor observation, 14D action, finite
contact/reset tensors, zero resets). Training stayed finite with zero
`nan_state` terminations and produced checkpoints at iterations 250, 500, 750,
and 999. However, the common 300-step, six-case battery did not show a stable
all-direction walking policy. Pass vectors in the order
`zero, forward, lateral, yaw, turn_left, turn_right` were:

- `model_250.pt`: `PPPFPF`
- `model_500.pt`: `PPFFFF`
- `model_750.pt`: `PFFFFF`
- `model_999.pt`: `PFFPFP`

All tensors remained finite and reset rates were effectively zero, but most
non-zero command cases had insufficient response. This is evidence that the
canonical IsaacLab recipe can learn a stable/low-motion basin in this backend,
but does not, in this clean 1000-iteration run, reproduce the trained walking
behavior that T22's strictification run reached. The result does not justify
changing rewards or adding another bootstrap stage without a bounded follow-up
hypothesis.

## Primary References

- `docs/isaaclab_velocity_flat_parity_ledger.md`
- `docs/isaaclab_velocity_flat_backend_comparison.md`
- `docs/status/active/isaaclab-backend-implementation.md`
- `src/isaaclab_microduck/actuators/bam_actuator.py`
- `src/isaaclab_microduck/assets/microduck.py`
- `scripts/isaaclab/fixed_root_dynamics_parity.py`
- `scripts/isaaclab/force_timing_probe.py`
- `scripts/isaaclab/lagged_friction_bridge_probe.py`
- `src/isaaclab_microduck/actuators/physx_friction_bridge.py`
- `scripts/isaaclab/asset_dynamics_runtime_probe.py`
- `scripts/isaaclab/velocity_flat_directional_trace.py`
- `scripts/sensor_source_trace_spec.py`
- `scripts/mjlab_sensor_source_trace.py`
- `scripts/isaaclab/sensor_source_trace.py`
- `scripts/compare_sensor_source_trace.py`
- `scripts/rsl_rl_one_update_probe.py`
