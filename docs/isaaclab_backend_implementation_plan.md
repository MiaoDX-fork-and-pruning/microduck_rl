# IsaacLab backend: local implementation and validation plan

Status: **T26 current IsaacLab train/restore/evaluate verification complete; T25 one-update RSL-RL probe complete; T24 raw sensor-source trace complete; T22 strict Velocity-Flat acceptance complete; backend deltas and deployment gates remain explicit**
Architecture reference: [`isaaclab_backend_architecture.md`](isaaclab_backend_architecture.md)  
Primary execution environment: local workstation(s) with Isaac Sim/IsaacLab-capable GPU  
Reference behavior: current `mjlab_microduck` tasks and real-robot sim2real lessons

## 1. Purpose

This plan is intended for long-running local development with Codex or another coding agent. It deliberately separates deterministic engineering work from expensive empirical training work.

The goal is to add an IsaacLab/PhysX backend to this repository while preserving the existing mjlab backend as the reference path.

A successful program produces a second backend that implements the same Microduck hardware-facing policy semantics and can independently train deployable policies. The first acceptance target is a strict semantic port of `Velocity-Flat`; other tasks remain later phases.

For this plan, equivalent training means matching the production 61D -> 14D
policy contract, reset/DR distribution, observations, commands, rewards,
terminations, actuator path, and PPO configuration. It does not require
identical MuJoCo/PhysX trajectories or reward curves. MuJoCo solver behavior
and the IsaacLab same-step external-load friction timing limitation are explicit
backend differences, not reasons to silently change task semantics.

## 1A. Locked parity decision

The implementation standard is:

```text
mjlab Velocity-Flat semantics
        |
        +-- same policy ABI, commands, observations, rewards, resets, DR,
        |   actuator delay/voltage behavior, and PPO recipe
        |
        +-- MuJoCo solver  !=  PhysX solver       (documented backend delta)
        +-- solved external torque timing differs (documented BAM limitation)
```

Do not compensate for a semantic mismatch by tuning rewards or PPO. Every
intentional divergence must be recorded in the parity ledger and covered by a
focused test or deterministic benchmark.

The next implementation slice is deliberately gated:

1. close action/BAM/asset semantics;
2. port reset/DR, actor sensor noise/delay/bias, commands, rewards, and
   privileged critic observations;
3. run the common fixed command battery and parity benchmarks;
4. run the 64-env/5-iteration smoke;
5. only then launch a new 4096-env/6000-iteration run.

The previous long run is diagnostic only: it used the old critic and failed the
walking battery. It must not be used as the strict-parity baseline.

## 1B. Parity ledger and acceptance rule

The implementation must maintain a machine-readable and human-readable parity
ledger at `docs/isaaclab_velocity_flat_parity_ledger.md` (with JSON fixtures in
`.cache/isaaclab-assets/` when measurements are needed). Each row records:

```text
surface | mjlab source | IsaacLab source | status | tolerance/evidence | owner
```

`status` is one of `MATCHED`, `BACKEND_DELTA`, `BLOCKED`, or `NOT_STARTED`.
`BACKEND_DELTA` is reserved for the two locked exceptions above and any solver
effect demonstrated by a deterministic benchmark. A missing implementation,
unmeasured difference, or reward/config workaround is not a backend delta.
The fixed command battery, ABI fixtures, actuator bench, and reward/config
tests are the evidence columns for closing rows. No long PPO run starts while a
P0 row is `BLOCKED` or `NOT_STARTED`.

The PPO software stack is part of the semantic comparison. First attempt to run
the IsaacLab image with the mjlab reference `rsl-rl-lib 5.0.1`; if IsaacLab
3.0.0 requires `5.4.1`, keep the supported version and mark the implementation
difference explicitly in the ledger with a controlled optimizer/rollout probe.

## 2. Working rules

1. Read `AGENTS.md` before changing existing task code.
2. Keep the current mjlab test suite green throughout the port.
3. Do not change a successful mjlab task merely to make IsaacLab easier.
4. Use one primary question per experiment.
5. Measure physics before launching PPO.
6. Never start a long training run before a minimal environment/step test and a small PPO smoke test.
7. Keep generated USD, checkpoints, videos, profiler traces, and datasets reproducible and clearly versioned.
8. Record exact Git SHA, IsaacLab/Isaac Sim version, Python/Torch/CUDA versions, GPU, asset revision, seed, and task config for every result used in a design decision.
9. Treat current real-robot behavior as stronger evidence than either simulator.
10. Do not add complexity (history, MoE, AMP, motion tracking) unless a measured failure requires it.

## 3. Suggested local workspace

One checkout, two simulator environments:

```text
~/work/microduck_rl/                   # repository
~/data/microduck_isaaclab/             # untracked large artifacts

microduck_rl/.venv-mjlab/              # current stack
microduck_rl/.venv-isaaclab/           # IsaacLab stack
```

Recommended artifact tree:

```text
~/data/microduck_isaaclab/
├── assets/
├── fixtures/
├── physics_bench/
├── runs/
│   ├── velocity_flat/
│   ├── velstand/
│   └── ...
├── videos/
└── reports/
```

Do not commit long-run artifacts to Git.

The canonical IsaacLab environment is the official Isaac Sim/IsaacLab
container or bundled Python distribution selected in Phase 0. Keep it separate
from the mjlab uv environment, and provide a checked-in launch wrapper plus
version manifest rather than attempting to merge both simulator stacks into a
single lockfile. The v0 implementation and training plan is local-first:
bring-up, smoke tests, physics benches, and the first full `Velocity-Flat` and
`VelStand` runs execute on an Isaac Sim-capable local workstation. Executor or
CloudML is not a current training target; it can be added later only after an
IsaacLab image, driver/runtime contract, storage mounts, and submission path
have been validated separately.

## 4. Phase 0 — dependency/environment spike

### Question

Can current mjlab and IsaacLab stacks coexist cleanly in one repository with independent uv environments?

### Tasks

- record current mjlab Python/Torch/CUDA/Warp requirements;
- select an IsaacLab/Isaac Sim release compatible with the available workstation GPU/driver;
- test installation using uv-supported package installation;
- decide whether backend-specific dependency groups, uv workspace members, or separate backend project files provide the cleanest isolation;
- add minimal developer commands/wrappers only after the approach is proven locally.

### Acceptance

- existing mjlab commands work unchanged;
- IsaacLab imports and starts headless from the repository checkout;
- each environment can be recreated from committed dependency metadata;
- no manual package mutation is required after setup.

### Stop condition

If a shared lock solution forces incompatible versions, stop trying to unify dependency resolution. Use separate backend environments/locks while retaining one Git repository.

## 5. Phase 1 — IsaacLab package skeleton

### Question

Can the repository host an IsaacLab backend without changing existing training behavior?

### Proposed additions

```text
src/isaaclab_microduck/
├── __init__.py
├── assets/
├── actuator/
├── mdp/
└── tasks/

scripts/isaaclab/
tests/isaaclab/
```

### Tasks

- create an IsaacLab extension/package entry point appropriate for package-installed IsaacLab;
- add task discovery/listing;
- add zero-action and random-action smoke scripts;
- ensure imports do not occur for mjlab-only workflows unless explicitly requested.

### Acceptance

- `import mjlab_microduck` remains unaffected;
- `import isaaclab_microduck` works in the IsaacLab environment;
- a placeholder environment can start/step/close;
- no existing task registrations change.

## 6. Phase 2 — robot asset and model parity

### Question

Does PhysX contain the same robot, not merely a visually similar USD?

### Tasks

Build or generate the Microduck USD from `robot_walk.xml` as the v0 mechanical
source. Add scripts or metadata documenting the generation path and emit a
machine-readable asset report. Do not accept a hand-maintained USD with no
source revision.

Extract a machine-readable comparison table from mjlab and IsaacLab for:

- actuated joint names/order;
- passive joint names/order;
- joint axes;
- hard/soft limits;
- default/HOME pose;
- body masses;
- inertias;
- COM offsets;
- body transforms;
- collision geometry identifiers;
- foot/head/trunk/mouth bodies used by tasks.

### Tests

Add tests that fail on:

- missing/renamed joints;
- reordered policy joints;
- HOME pose mismatch;
- limit mismatch beyond explicit tolerance;
- unexpected passive joints.

### Physical sanity checks

Before adding a learned actuator:

- spawn at HOME;
- check gravity direction and coordinate conventions;
- visualize collision geometry;
- confirm feet/head/trunk contacts fire on intended bodies;
- confirm no immediate self-intersection/explosion.

### Acceptance

A generated parity report explains every remaining physical difference. No unexplained mass, inertia, transform, joint, or collision discrepancy remains.

## 7. Phase 3 — policy ABI parity

### Question

Can both backends construct the same hardware-facing actor input and intended joint target?

### Tasks

Implement IsaacLab equivalents for the current actor terms:

```text
gyro                         3
projected gravity             3
joint position relative HOME 14
joint velocity               14
previous raw action          14
command block                13
-------------------------------
total                        61
```

Implement the same command order:

```text
[twist(3), head_pose(4), body_pose(6)]
```

Implement the current action transform:

```text
raw action -> HOME + scale * raw action
```

Do not use full-joint-range rescaling.

### Golden fixtures

Create simulator-neutral fixtures containing fixed sensor/joint/command states and expected 61D observation vectors.

Create fixed action vectors and expected pre-actuator target positions.

Run the fixtures against:

- current mjlab observation/action implementation;
- new IsaacLab implementation.

Where practical, keep fixtures usable later by the Rust runtime tests as well.

### Acceptance

- observation shape is exactly 61;
- action shape is exactly 14;
- golden values match within explicit numerical tolerance;
- coordinate/frame differences are resolved before training.

## 8. Phase 4 — BAM actuator bench

### Question

Can PhysX reproduce the existing actuator response closely enough that policy comparisons are meaningful?

### Tasks

Implement a custom explicit IsaacLab actuator for the XL330/BAM behavior used by Microduck.

Port/represent the current model components, including as applicable:

- position target to motor command;
- voltage scaling;
- back EMF;
- effort saturation;
- friction;
- friction scaling/randomization;
- battery voltage variation/sag;
- armature/inertia effects.

### Pure numerical bench

Before connecting to an articulation, evaluate both actuator implementations over a grid:

```text
position error
joint velocity
target step
battery voltage
friction scale
```

Save comparison reports with max/mean absolute error and plots.

### Dynamic bench

For a fixed isolated joint or constrained robot:

1. step target response;
2. sinusoidal target response;
3. velocity-dependent torque response;
4. low/high voltage response;
5. friction sweep.

### Acceptance

- equations and units are documented;
- numerical fixture parity is within chosen tolerances;
- any simulator-side differences are identified rather than silently compensated;
- the actuator remains vectorized enough for thousands of environments.

### Stop condition

Do not begin locomotion training while the actuator bench shows unexplained large discrepancies.

## 9. Phase 5 — deterministic physics battery

### Question

Are basic PhysX dynamics credible before RL is allowed to hide model errors?

### Battery

Run matched setups in MuJoCo and PhysX:

1. HOME settle for several seconds;
2. fixed crouch settle;
3. free fall from fixed height/orientation;
4. small roll/pitch perturbation;
5. horizontal base impulse/push;
6. foot slide under known vertical load;
7. trunk/head contact case;
8. prone/face-up resting poses;
9. joint target step with full robot;
10. command-independent NaN/stability soak.

### Metrics

Record:

- root height/orientation;
- joint positions/velocities;
- actuator torques;
- contact forces/impulses;
- energy/action statistics;
- settle time;
- slip distance;
- bounce/restitution behavior.

### Acceptance

No metric needs to be exactly identical, but each major difference must be understood and classified as:

- expected solver difference;
- asset mismatch;
- actuator mismatch;
- contact/material mismatch;
- bug.

Only the first category is acceptable without corrective work.

## 10. Phase 6 — Velocity Flat task

### Question

Can the direct-RL walking recipe be reproduced in IsaacLab without AMP or reference motion?

### Porting principle

Use `microduck_velocity_env_cfg.py` as the semantic reference. Do not port the Open Duck Mini IsaacLab AMP recipe.

Match, as appropriate:

- command ranges;
- zero-command sampling;
- turn-in-place sampling/buckets;
- observation noise;
- action semantics;
- reward formulas and signs;
- termination behavior;
- pushes;
- mass/friction/IMU/encoder/actuator DR;
- episode/control rate;
- PPO settings.

### Development sequence

1. construct env with one robot;
2. run random actions;
3. run 64-env stepping soak;
4. run reward-term unit tests;
5. run 64-env / 5-iteration PPO smoke test;
6. export/check policy shape if export path exists;
7. only then launch 4096-env training.

### Baseline report

Before comparing training, capture current mjlab velocity metrics using fixed command batteries and seeds.

IsaacLab candidate report should include:

- forward/lateral/yaw command error;
- zero-command stability;
- turn-in-place quality;
- fall rate;
- foot slip;
- action rate;
- torque/effort statistics;
- robustness to pushes;
- qualitative video.

### Acceptance

IsaacLab need not match mjlab reward curves. It must learn a stable, command-responsive gait with no obvious exploit and with task metrics in the same useful range.

## 11. Phase 7 — simulator cross-validation

### Question

What does the backend difference change after both can train walking?

### Experiments

Keep task semantics fixed and vary only the simulator backend.

Compare:

- learning speed;
- final task metrics;
- action distribution;
- gait frequency/step shape;
- contact/slip statistics;
- robustness under matched DR envelopes;
- exported-policy behavior in each simulator where practical.

A particularly useful matrix is:

```text
train mjlab  -> evaluate mjlab
train mjlab  -> evaluate PhysX adapter/rehearsal (if feasible)
train PhysX  -> evaluate PhysX
train PhysX  -> evaluate MuJoCo rehearsal (if feasible)
```

Cross-simulator execution may not be perfectly meaningful because normalizers and physics differ; use it as a diagnostic, not as a hard success criterion.

### Hardware gate

Only after simulation checks pass should a PhysX-trained candidate be considered for controlled hardware testing using the existing runtime safety/fallback procedure.

## 12. Phase 8 — VelStand / recovery

### Question

Can the backend reproduce a policy that both walks and recovers from falls?

Port the existing VelStand design rather than inventing a new recovery recipe.

Preserve lessons around:

- clean-walking data share;
- fall/recovery gating;
- potential-based upright/height progress;
- prone and crouch reverse curricula;
- failed-recovery timeout;
- recovery-success thresholds measured from actual standing envelope;
- avoiding rewards that can be farmed while fallen.

### Evaluation battery

Separate by initial state:

```text
upright
natural fall
crouch
face down
face up
left side
right side
```

Report success and time-to-recover independently for each bucket.

### Acceptance

Walking regression remains within an agreed tolerance and recovery succeeds across all intended spawn categories.

## 13. Phase 9 — incremental task ports

Port one task at a time after VelStand.

Recommended order:

1. SitStand;
2. GroundPick;
3. BallKick;
4. Roulade.

For every task:

- write the task-specific semantic mapping first;
- port pure reward math before framework glue;
- create deterministic config tests;
- smoke test;
- capture a specialist baseline from mjlab;
- run local training;
- compare task-specific metrics;
- document any deliberate divergence.

Do not use aggregate reward as the acceptance criterion.

## 14. Phase 10 — backlash

Start only after normal-foot non-backlash policies are credible.

### Tasks

- reproduce serial passive backlash geometry in USD/PhysX if feasible;
- verify servo-side vs output-side joint indexing;
- ensure actor observation uses the same intended encoder view as mjlab;
- ensure rewards tracking the same joints use the same view;
- build backlash/no-backlash A/B tests.

### Acceptance

The backlash variant differs only in intended mechanics/observation view, not through accidental robot-model differences.

## 15. Phase 11 — rollers

Rollers are last because passive small-radius wheel contact and rolling resistance can differ substantially across physics engines.

Before training:

- validate wheel DOFs and interleaved indexing;
- validate rolling friction/drag;
- validate lateral vs longitudinal contact behavior;
- compare passive coast-down tests;
- compare incline/descent tests.

Only then port roller tasks and curricula.

## 16. Export/deployment work

Do not change production runtime during early simulator work.

The production ABI target is 61D -> 14D. The legacy 51D inference mode in
`scripts/infer_policy.py` is compatibility-only and is out of scope for the
IsaacLab backend.

When a PhysX-trained specialist reaches deployment quality:

- export with the same actor observation normalization semantics;
- validate ONNX input/output shapes;
- attach backend/task/ABI metadata;
- run deterministic PyTorch-vs-ONNX parity vectors;
- rehearse in a CPU/simulator path if available;
- use existing runtime safeguards and specialist fallback for first robot tests.

Passing IsaacLab evaluation is not sufficient for hardware testing. The
candidate must first pass the existing 61D ONNX parity checks, fixed command
battery, and MuJoCo/runtime rehearsal path.

A backend tag is useful metadata but must not alter the runtime control contract.

## 17. CI implementation plan

### Always-on cheap tests

- common constants/schema;
- ABI golden fixtures;
- action-transform fixtures;
- pure reward formulas;
- pure actuator fixtures where imports permit;
- no accidental dependency import from the other backend.

### Backend-specific jobs

Maintain separate commands/jobs for mjlab and IsaacLab. Do not make Isaac Sim installation a requirement for documentation or mjlab-only changes.

### Scheduled/manual GPU jobs

- IsaacLab environment startup;
- physics battery subset;
- PPO smoke tests;
- selected parity benchmarks.

Full runs remain experiment jobs, not CI.

## 18. Experiment record format

Every meaningful experiment should produce a small Markdown or JSON manifest such as:

```yaml
name: isaac_velocity_flat_v003
git_sha: ...
backend: isaaclab
isaaclab_version: ...
isaacsim_version: ...
python: ...
torch: ...
cuda: ...
gpu: ...
asset_revision: ...
task: ...
seed: ...
num_envs: 4096
iterations: ...
changes_from_baseline:
  - "..."
result:
  success: true
  notes: "..."
artifacts:
  checkpoint: ...
  video: ...
  report: ...
```

This should be machine-readable enough that Codex can compare runs without relying on chat history.

## 19. Failure triage order

When IsaacLab training fails, investigate in this order:

1. asset/joint/frame mismatch;
2. action semantics mismatch;
3. observation semantics mismatch;
4. actuator mismatch;
5. contact/material mismatch;
6. reset/termination mismatch;
7. reward sign/formula mismatch;
8. DR too broad/incorrect;
9. PPO/training issue;
10. genuine simulator-specific learning difficulty.

Do not jump directly to reward tuning before the first five are checked.

## 20. Recommended initial Codex task queue

The following queue is intentionally concrete and can be handed to a local coding agent one item at a time.

### Task A — environment isolation spike

Deliver:

- selected IsaacLab version;
- reproducible uv environment setup;
- documented command to import/start IsaacLab headless;
- no changes to existing mjlab environment behavior.

Stop after the environment works.

### Task B — skeleton package

Deliver:

- `isaaclab_microduck` package;
- task registration/list command;
- zero/random agent smoke path;
- import tests.

No robot parity work yet.

### Task C — Microduck USD and asset report

Deliver:

- generated/imported USD;
- `ArticulationCfg`;
- joint/body parity report;
- tests for joint order/HOME/limits.

Do not train.

### Task D — ABI adapter

Deliver:

- 61D observation;
- 14D action mapping;
- command mapping;
- golden parity fixtures.

Do not tune physics.

### Task E — BAM numerical port

Deliver:

- explicit actuator implementation;
- fixture sweep comparing reference and IsaacLab math;
- benchmark/profiling report.

Do not train locomotion until unexplained errors are resolved.

### Task F — deterministic physics battery

Deliver:

- executable benchmark script;
- MuJoCo/PhysX report for settle/fall/push/slip/step-response cases;
- list of understood differences.

### Task G — Velocity Flat smoke

Deliver:

- task cfg;
- reward/config tests;
- 64-env 5-iteration smoke result;
- no long run yet.

Current evidence: the IsaacLab 3.0.0 / Isaac Sim 6.0.1 runtime
passes the 64-environment random-action soak and the official RSL-RL
5-iteration PPO smoke (7,680 steps). The run uses the checked-in
`spawn_ground_after_clone` prestartup hook required by the imported USD and
records finite 61D observations / 14D actions. It produces a smoke checkpoint
under the ignored `logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/` path.
The smoke is an integration gate only; its falling behavior is not evidence of
a trained gait.

### Task H — strict-parity walking run

Deliver only after the parity closure gates above pass:

- baseline manifest;
- training run;
- metrics report;
- video;
- comparison to current mjlab baseline.

Current evidence: the official IsaacLab 3.0.0 RSL-RL launcher loads the smoke
checkpoint and exits successfully after a 32-frame Kit playback. The video is
recorded at
`logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/2026-09-03_13-02-22/videos/play/clip_0000.mp4`;
see `docs/isaaclab_velocity_flat_playback_report.md`. This proves the runtime
and recorder path, not gait quality. The fixed-seed command battery now
completes with 16 environments and 250 steps each for zero, forward, lateral,
and yaw commands; all tensors are finite. Reset fractions remain about 0.021
per environment step and maximum tilt reaches about 0.85 rad, so this smoke
checkpoint is not a reliable gait. See
`docs/isaaclab_velocity_flat_command_battery_report.md`. A same-format current
  mjlab baseline comparison is now recorded in
  `docs/isaaclab_velocity_flat_backend_comparison.md`. The comparison is a
  forward-only directional slice because the available battery horizons and
  command sets differ; it shows the accepted mjlab rollout staying upright
  while the five-iteration IsaacLab checkpoint resets about 2.1% of
  environments per step and reaches about 0.85 rad tilt. Task H is therefore
  not accepted as a walking result. The completed 6000-iteration budget run is
  invalid as the final baseline because its critic architecture was stale and
  its battery failed. That run predates the strict-parity lock in this plan and
  was launched under the earlier "close enough" acceptance rule; it is not an
  execution of the current Task H. T22 subsequently completed the independent
  strict-trained policy gate with a separate adapted-to-strict curriculum;
  its final checkpoint and ONNX/MuJoCo rehearsal are recorded under T22 below.
  A future replacement run requires a new bounded hypothesis and manifest;
  external-load BAM friction parity remains an explicit PhysX limitation.

### Task I — decide whether to continue

Proceed to VelStand only if:

- no unresolved ABI/actuator/model issue remains;
- walking is qualitatively credible;
- task metrics are competitive enough to justify further porting.

Otherwise fix the foundation first.

The first IsaacLab judgment slice is intentionally limited to `Velocity-Flat`
followed by `VelStand` as the contact-rich task. The remaining specialist tasks
are subsequent milestones, not prerequisites for deciding whether the backend
is viable.

Hardware validation remains a separate three-stage release gate:

```text
actuator bench
    -> MuJoCo/runtime rehearsal
    -> human-approved controlled robot test
```

Before the final stage, a PhysX-trained policy must pass 61D/14D ONNX parity,
the fixed command battery, finite-duration MuJoCo/runtime rehearsal, and the
defined safety metrics. IsaacLab evaluation alone is not authorization for
hardware testing.

## 21. What not to do during local development

Avoid these shortcuts:

- copy the Open Duck Mini AMP environment and call the port complete;
- feed actor-only simulator states such as exact base linear velocity;
- use IdealPD as the final actuator and infer sim2real quality from it;
- rescale actions to full joint limits;
- tune reward weights to compensate for an unexplained model mismatch;
- port all tasks before walking is validated;
- add rollers/backlash before basic PhysX contact behavior is understood;
- refactor the existing mjlab backend solely for symmetry;
- introduce a generic simulator abstraction layer early;
- launch long runs without manifests and fixed evaluation batteries.

## 22. Completion criteria for the first IsaacLab program

The first program can be considered successful once all of the following hold:

1. one repository supports reproducible mjlab and IsaacLab developer environments;
2. existing mjlab workflows and tests remain stable;
3. Microduck USD/model parity is documented and tested;
4. actor ABI/action semantics are golden-tested across backends;
5. BAM actuator behavior is numerically and dynamically benchmarked;
6. Velocity Flat matches mjlab task semantics, including reset/DR, sensor
   corruption, commands, rewards, terminations, and privileged critic inputs;
7. Velocity Flat trains successfully in IsaacLab using direct RL and passes the
   fixed command battery with a new strict-parity manifest;
8. VelStand or one other contact-rich task demonstrates the backend is not limited to simple walking;
9. evaluation reports make simulator differences explicit;
10. a PhysX-trained policy can enter the existing controlled hardware-validation process without runtime API redesign.

After that point, additional tasks, backlash, rollers, generalist training, cameras, navigation, and embodied-AI work can be prioritized based on product/research needs rather than treated as prerequisites.

## 23. What already exists

- `src/isaaclab_microduck/tasks/velocity_flat.py` already provides the 61D
  actor group, 14D named action mapping, basic commands, rewards, and reset
  hooks; the parity work extends these paths instead of creating a second task.
- `src/isaaclab_microduck/actuators/bam_actuator.py` and
  `src/isaaclab_microduck/actuators/bam_math.py` already provide the vectorized
  BAM voltage/torque core; delay, voltage sag, and PhysX integration should be
  added there or at the IsaacLab manager boundary.
- `scripts/isaaclab/velocity_flat_command_battery.py`, the physics battery, and
  the existing contract tests are the required evaluation harnesses to extend.
- `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` and
  `src/mjlab_microduck/tasks/mdp.py` remain the semantic source of truth; do not
  duplicate their formulas in a new shared abstraction before parity is proven.

## 24. NOT in scope for this slice

- VelStand, SitStand, GroundPick, BallKick, Roulade, backlash, and rollers:
  deferred until strict Velocity-Flat acceptance prevents parallel task drift.
- Production runtime or hardware control changes: deployment remains behind the
  existing ONNX and human-approved hardware gates.
- A generic simulator abstraction layer: it would hide backend differences before
  the parity ledger has identified which differences are real.
- Exact trajectory/reward-curve equality across MuJoCo and PhysX: solver and
  same-step external-load friction timing are explicitly excluded exceptions.

## 25. Implementation tasks for the next context

- [x] **T1 (P1)** — Create and maintain the parity ledger; populate every
  Velocity-Flat surface from the current comparison audit and attach tests or
  benchmark evidence to each row.
- [x] **T2 (P1)** — Align action clipping/target scaling, BAM control-step delay,
  per-environment voltage sag, friction duplication, USD damping, and HOME/limit
  handling; add actuator and asset regression tests. The production mjlab
  runner's default `clip_actions=None` is now matched; the previous strict
  checkpoint used an Isaac-only clip and is diagnostic-invalid. Remaining
  solved external-load timing is the explicit `BACKEND_DELTA`.
- [x] **T3 (P1)** — Port reset distribution, CoM/mass/armature/friction/push DR,
  encoder bias, IMU and observation noise/delay, preserving non-accumulation;
  add deterministic seeded distribution tests. Implementation, CPU contracts,
  and seeded production-runtime distribution evidence are attached in the
  active parity ledger.
- [x] **T4 (P1)** — Port head/body commands, turn-in-place sampling, complete
  Velocity-Flat rewards/terminations, and privileged critic observations while
  preserving the 61D actor ABI; add pure-function and config tests. All terms
  are wired and finite in the production probes. Foot-site/direct count
  kernels, seeded commands, rewards, terminations, and the 76D critic are
  covered; the filtered self-contact view remains intentionally disabled in
  favor of the concrete raw PhysX view.
- [x] **T5 (P1)** — Attempt `rsl-rl-lib 5.0.1` in the IsaacLab image; if the
  supported stack must remain `5.4.1`, run a controlled optimizer/rollout probe
  and record the result as a software-stack delta.
- [x] **T6 (P1)** — Extend the common command battery to identical commands,
  horizons, seeds, reset rules, and metrics; the harness requires finite
  tensors and records translation, tilt, yaw response, and resets. Motion
  quality gates are evaluated on a trained checkpoint, not the 5-iteration
  integration smoke.
- [x] **T7 (P1)** — Run the 64-env/5-iteration smoke and launch a replacement
  4096-env/6000-iteration strict-parity run with a new manifest. The current
  checkout also passes a fresh 64-env/5-iteration random-action smoke and
  official RSL-RL training smoke, writing `model_0.pt` and `model_4.pt` under
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_09-47-26/`.
  The corrected strict long run and `[1,61] -> [1,14]` export are complete,
  but strict trained-policy acceptance remains blocked because its fixed
  battery fails forward/lateral response. The earlier 6000-iteration run
  trained with `clip_actions=1.0` remains diagnostic-invalid.

- [x] **T8 (P1)** — Verify official RSL-RL checkpoint resume and run one
  bounded strict-task warm-start hypothesis. Loading adapted `model_750.pt`
  into the unchanged strict task and continuing 250 iterations at 4096 envs
  produced strict-task `model_999.pt`. The 300-step fixed battery passes all
  six cases with zero resets; the checkpoint audit and official ONNX export
  also pass. This closes a viable strict-task walking candidate; its warm-start
  provenance remains distinct from an independently strict-trained result.

- [x] **T9 (P1)** — Run the exported warm-start checkpoint through a finite,
  headless CPU MuJoCo battery using the shared six-case command specification.
  All six cases pass with finite 61D/14D inference, zero resets, positive
  body-frame directional response, and bounded tilt. This closes runtime
  deployment evidence for the warm-start candidate; independent strict-from-
  scratch training remains the only trained-policy blocker.

- [x] **T10 (P1)** — Evaluate an independently strict-trained checkpoint with
  the common six-case battery before authorizing another long run. The
  corrected strict `model_250.pt` is finite but fails every case, with roughly
  `0.16` resets per environment step and negative zero-command drift. The
  independent strict-training gate therefore remains blocked. Any next run
  requires one bounded training hypothesis and a new manifest; this result does
  not justify reward/PPO changes by itself.

- [x] **T11 (P1, bounded hypothesis)** — Test whether the PhysX contact path
  makes the strict `air_time` term dominate early optimization. Keep the
  strict task, command distribution, termination/DR configuration, PPO recipe,
  and all other reward terms unchanged; override only `air_time.weight` to
  `1.0`, run `750` iterations at `4096` environments, and evaluate the saved
  checkpoint with the common six-case battery. The run completed normally and
  remained finite, with approximately 950/1000 mean episode length, but its
  `model_749.pt` still fails forward/lateral response and settles near `0.071 m`
  root height. Zero, yaw, and turn cases pass. Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_strict_airtime1_model749.json`.
  This rejects air-time dominance as the complete explanation; it is not a
  strict-parity acceptance result.

- [x] **T12 (P1, bounded hypothesis)** — Test whether the strict low-height
  termination creates the observed crouch basin during bootstrap. Keep the
  strict reward, commands, DR, and PPO recipe unchanged, train `750` iterations
  at `4096` environments with only `env.terminations.root_height=null`, then
  evaluate the checkpoint under the unchanged strict task (height guard
  restored). The finite endpoint `model_749.pt` fails all six battery cases:
  the zero case has `776` resets (`0.1617` per env-step) and `-0.199 m/s`
  drift, and commanded translation/yaw fail. Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_strict_rootbootstrap_model749.json`
  (SHA256 `3d764508b657d657346c990edd5f7cb882b8d6b28a31262c53e182773118c256`).
  This rejects the low-height bootstrap hypothesis; the training-only override
  does not alter the canonical MJLab recipe, and independent strict acceptance
  remains blocked.

- [x] **T13 (P1, bounded hypothesis)** — Test whether the strict action-rate
  curriculum suppresses gait discovery. A separately registered strict
  diagnostic task froze `action_rate_l2` at its initial `-0.1` weight and ran
  `750` iterations at `4096` environments. The run stayed finite with no NaN
  terminations and the live curriculum remained `-0.1000`. Under the unchanged
  strict battery, `model_250.pt` passed zero/forward/lateral/left-turn but
  failed yaw/right-turn; `model_500.pt` and `model_749.pt` lost additional
  directional response while remaining reset-free. Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_t13_action_rate_flat_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t13_action_rate_flat_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t13_action_rate_flat_model749.json`.
  This rejects action-rate freezing as a complete explanation, though it
  improves early stabilization. The canonical MJLab recipe remains unchanged.

- [x] **T14 (P1, bounded hypothesis)** — Test whether strict angular tracking
  reward is too weak for yaw/turn skill discovery. Only
  `env.rewards.track_ang_vel.weight` changed from `2.0` to `6.0`; commands,
  action-rate stages, terminations, DR, PPO, and all other rewards remained
  strict. The `4096`-environment run completed `750` iterations with finite
  losses and no NaN terminations. Under the unchanged strict six-case battery,
  `model_250.pt`, `model_500.pt`, and `model_749.pt` all pass zero/yaw/turn-left/
  turn-right but fail forward and lateral response (the endpoint has one
  negligible lateral reset). Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_t14_angvel6_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t14_angvel6_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t14_angvel6_model749.json`.
  T14 is rejected as a complete explanation; the independent strict-training
  gate remains open, and the canonical MJLab recipe is unchanged.

- [x] **T15 (P1, bounded hypothesis)** — Test whether strict linear tracking
  weight is too weak for translation discovery. Only
  `env.rewards.track_lin_vel.weight` changed from `2.0` to `4.0`; the
  `4096`-environment run completed `750` iterations with finite losses and no
  NaN terminations. Under the unchanged strict six-case battery, all three
  checkpoints pass zero/forward/lateral/turn-left but fail yaw and turn-right.
  Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_t15_linvel4_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t15_linvel4_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t15_linvel4_model749.json`.
  This establishes a complementary translation-versus-positive-yaw tradeoff;
  it is not a strict-parity acceptance result.

- [x] **T16 (P1, bounded hypothesis)** — Test whether the complementary
  tracking terms need the adapted profile's pair: change only strict
  `track_lin_vel.weight=4.0` and `track_ang_vel.weight=6.0`, train `750`
  iterations at `4096` environments, and evaluate under the unchanged strict
  task. Keep commands, action-rate stages, terminations, DR, PPO, and every
  other reward unchanged. The run completed finite with no NaN terminations.
  All three checkpoints pass zero/forward/lateral; `model_500.pt` and
  `model_749.pt` also pass turn-left, but yaw and turn-right fail. Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_t16_tracking_pair_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t16_tracking_pair_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t16_tracking_pair_model749.json`.
  The pair preserves translation but does not resolve yaw-direction asymmetry;
  the canonical MJLab recipe remains unchanged.

- [x] **T17 (P1, bounded hypothesis)** — Test whether the observed positive-yaw
  blind spot is an optimization asymmetry. Keep the T16 tracking pair
  (`track_lin_vel=4.0`, `track_ang_vel=6.0`) and all strict task semantics
  unchanged, but enable the explicitly registered left-right symmetry data
  augmentation runner. Train `750` iterations at `4096` environments and
  evaluate under the canonical strict task. Accept only a finite, zero-reset
  checkpoint with all six battery cases passing; do not enable symmetry on the
  canonical runner or alter the MJLab recipe. The mirror transform is covered
  by CPU involution/sign contracts and the IsaacLab smoke. The `750`-iteration
  run remained finite with no NaN terminations. Under the canonical battery,
  `model_500.pt` passes zero/forward/lateral/turn-left but fails yaw/turn-right;
  `model_749.pt` passes zero/yaw/turn-left/turn-right but fails lateral. No
  checkpoint passes all six, so T17 is rejected as a complete solution.
  Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_t17_symmetry_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t17_symmetry_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t17_symmetry_model749.json`.

- [x] **T18 (P1, bounded hypothesis)** — Test whether T17 contains a transient
  all-six behavior window that the coarse `250`-iteration checkpoint cadence
  missed. Keep the T17 symmetry task, tracking pair, commands, rewards,
  curricula, terminations, DR, PPO, and `4096` environments unchanged; override
  only `agent.save_interval=50` and train `750` iterations from scratch. Evaluate
  every saved checkpoint with the canonical six-case battery. Accept only a
  finite, zero-reset checkpoint passing zero/forward/lateral/yaw/turn-left/
  turn-right; this experiment changes observability, not policy semantics. The
  run completed normally and wrote checkpoints every 50 iterations, but no
  checkpoint passed all six: pass vectors (zero, forward, lateral, yaw,
  turn-left, turn-right) were `FFFFFF` at model 0, `PPPFFP` at model 50,
  `PPFFPP` at model 100, `PPPFPF` at model 150 and models 350--600, `PPPFPP` at
  model 200, `PPPFFF` at model 250, and `PPFFPF` at model 300 and models 650--749. T18 is
  rejected; however, post-run manifest inspection found that the invocation did
  not carry T17's `track_lin_vel=4.0` / `track_ang_vel=6.0` overrides. The
  resulting `2.0/2.0` run is retained as invalid provenance evidence and does
  not test the T17 checkpoint window.

- [x] **T19 (P1, bounded hypothesis)** — Repeat the T17 checkpoint-window test
  with the exact T17 recipe: symmetry augmentation enabled,
  `track_lin_vel=4.0`, `track_ang_vel=6.0`, `4096` environments, and only
  `agent.save_interval=50` changed for observability. Train `750` iterations
  from scratch, smoke-test the exact overrides first, and replay every saved
  checkpoint under the canonical six-case battery. Accept only a finite,
  zero-reset checkpoint passing all six cases; do not treat T18's default-pair
  run as evidence for this hypothesis. The exact-pair smoke passed and the
  `750`-iteration run completed normally with finite losses and no NaN
  terminations. All 16 checkpoints were replayed under the canonical battery;
  pass vectors were `FFFFFF` (0), `PPFFFF` (50), `PPPFFP` (100), `PPPPFP`
  (150), `PPPFFF` (200), `PPFPFF` (250), `PPPFPF` (300--400 and 500),
  `PPPFPP` (450 and 550), and `PPFPPP` (600--749). No checkpoint passed all
  six, so T19 is rejected and the independent strict-training gate remains
  open.

- [x] **T20 (P1, bounded hypothesis)** — Isolate the adapted command-bucket
  distribution as a training signal. The new
  `IsaacLab-Velocity-Flat-MicroDuck-CommandBuckets` diagnostic keeps strict
  rewards, terminations, observations, DR, actuator semantics, and PPO
  unchanged, but sets only `rel_forward_envs=0.0` and
  `rel_lateral_envs=0.25`, matching the known-good adapted profile. Run the
  64-env/5-iteration smoke first, then train 750 iterations at 4096 envs from
  scratch with checkpoints at 250/500/749. Replay each checkpoint under the
  canonical six-case battery. Accept only a finite, zero-reset all-six
  checkpoint. A pass would identify command coverage as the blocker and
  justify a later strict-final-stage curriculum; a fail rejects this signal
  without changing the canonical task or launching another blind long run.
  The smoke passed and the 750-iteration run completed with finite losses and
  zero NaN terminations. The canonical battery pass vectors were `PPFPFP`
  (model 250), `PPFFPF` (model 500), and `PFFFFP` (model 749), so no
  checkpoint passed all six. T20 is rejected: command buckets improved early
  translation/stability but did not recover yaw or right-turn response.
  Evidence: `.cache/isaaclab-assets/velocity_flat_smoke_t20_command_buckets_64.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t20_command_buckets_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t20_command_buckets_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t20_command_buckets_model749.json`,
  and `logs/rsl_rl/microduck_isaaclab_velocity_flat_command_buckets/2026-09-11_18-20-09/`.

- [x] **T21 (P1, bounded hypothesis)** — Combine the T20 adapted command
  buckets with the existing symmetry augmentation. Keep strict rewards,
  terminations, observations, DR, actuator semantics, and PPO unchanged;
  enable only `rel_forward_envs=0.0`, `rel_lateral_envs=0.25`, and the
  registered left-right symmetry runner. Smoke-test, then train 750 iterations
  at 4096 environments from scratch and replay models 250/500/749 under the
  canonical battery. Accept only a finite, zero-reset all-six checkpoint.
  This tests whether command coverage and yaw-direction augmentation are
  complementary; reject it without changing the canonical task if no
  checkpoint passes. The smoke passed and the 750-iteration run completed
  with finite losses and zero NaN terminations. The canonical battery pass
  vectors were `PPPFPF` (model 250), `PPFFPF` (model 500), and `PFFFPF`
  (model 749), so no checkpoint passed all six. T21 is rejected: symmetry did
  not recover the missing yaw/right-turn response.
  Evidence: `.cache/isaaclab-assets/velocity_flat_smoke_t21_command_buckets_symmetry_64.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t21_command_buckets_symmetry_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t21_command_buckets_symmetry_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t21_command_buckets_symmetry_model749.json`,
  and `logs/rsl_rl/microduck_isaaclab_velocity_flat_command_buckets_symmetry/2026-09-11_18-50-09/`.

- [x] **T22 (P1, bounded hypothesis)** — Train from scratch with an explicit
  adapted-to-strict curriculum. Bootstrap using the known-good adapted command
  buckets and reward balance, keep root-height termination effectively off,
  then switch all of those fields to the strict MJLab values at a stated
  curriculum boundary before evaluation. The production canonical task stays
  unchanged; the diagnostic must use live managers for reward, command, and
  termination mutations. Smoke-test first, then train 1000 iterations at 4096
  environments and evaluate checkpoints before and after strictification.
  Accept only a finite, zero-reset checkpoint that passes all six cases while
  the final live configuration is strict. This tests whether the strict
  recipe's optimization basin, rather than a missing ABI/physics primitive,
  prevents independent training. The 64-env/5-step smoke passed with finite
  61D observations, 14D actions, zero resets, and finite contact forces. The
  4096-env run completed 1000 iterations from scratch at
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_strictification/2026-09-11_19-26-41/`.
  The live manager curriculum switched at iteration 500 (step 12000) to strict
  command buckets, reward weights, root-height threshold, and action-rate
  schedule, then advanced its later stages at iterations 750 and 1000. No NaN
  terminations occurred. The canonical six-case battery pass vectors were
  `PPFPFP` (model 250), `PPPPPP` (model 500), `PPPPPP` (model 750), and
  `PPPPPP` (model 999). T22 is accepted: model 500 is the first independently
  strict checkpoint passing all six cases, and the later strict checkpoints
  retain the pass. The battery harness explicitly applies the final strict
  live-manager profile before replay. Evidence:
  `.cache/isaaclab-assets/velocity_flat_smoke_t22_strictification_64.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t22_strictification_model250.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t22_strictification_model500.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t22_strictification_model750.json`,
  and `.cache/isaaclab-assets/velocity_flat_command_battery_t22_strictification_model999.json`.
  The final `model_999.pt` has SHA256
  `2034e7c3f6893f700de2d321746f1267e062aa3d3d29a8ceec4f3005e17e16ce`.
  Official IsaacLab play exported a finite `[1,61] -> [1,14]` ONNX graph
  (SHA256 `371cb8d92300377363c89b5ada5c7c39683dd1464f93ea620c4c83859b3ce30e`),
  and the graph passes the headless CPU MuJoCo six-case rehearsal with zero
  resets in `.cache/isaaclab-assets/mujoco_onnx_battery_t22_strictification_model999.json`.

- [x] **T23 (P1, bounded hypothesis)** — Test whether a reset-safe one-step-
  lagged PhysX external-effort bridge can close the BAM load-dependent friction
  gap. Keep the accepted T22 task and production `BamActuator` motor-only;
  sample projected-minus-actuation joint effort only after `scene.update()` and
  apply it on the following pre-step. The fixed-root probe passed with finite
  14-joint loads, zero reset leakage, one-step warm-up, and distinct friction
  coefficients at scales `0.5/1.0/1.5`. A 100-step T22 `model_999.pt` battery
  passed all six cases for both lagged and production paths. The canonical
  300-step lagged battery failed `turn_left` with one tilt reset
  (`max_tilt_rad=1.08818`, reset fraction `0.0002083`); the other five cases
  passed. T23 is rejected as a production BAM parity solution. Evidence:
  `.cache/isaaclab-assets/lagged_friction_bridge_probe.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_one_step_lag_100.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_production_100.json`,
  and `.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_one_step_lag_300.json`
  (SHA256 `ac4702d6b84b010149d1457a8f52df5cadfdf8d1a2de3e493aad3de0e1440263`).
  The bridge remains diagnostic-only; do not tune it to recover this checkpoint.

- [x] **T24 (P1, bounded hypothesis)** — Compare raw sensor sources before
  attributing remaining policy differences to PPO or solver behavior. A seeded
  12-step scripted root/joint state trace was written for both backends with
  explicit delay index and reset metadata. Root gyro/gravity, policy-ordered
  joint state, and foot-site position match at floating-point precision;
  foot-site velocity differs by at most `8.84e-3 m/s`, and reconstructed
  subtree angular momentum differs from the named MuJoCo sensor by at most
  `7.99e-6`. These are retained as explicit source/backend deltas; no reward
  or observation workaround was introduced. Evidence:
  `.cache/isaaclab-assets/sensor_source_comparison.json`,
  `tests/test_isaaclab_sensor_source_trace_contract.py`.

- [x] **T25 (P1, bounded hypothesis)** — Isolate the `rsl-rl-lib 5.0.1`
  versus `5.4.1` implementation difference with one identical synthetic
  rollout/update. Both stacks produce identical initial/final actor and critic
  parameter hashes and update metrics for the exercised feed-forward path.
  The IsaacLab entrypoint remains on supported `5.4.1`; source differences in
  PPO and rollout storage remain an explicit training-stack delta. Evidence:
  `.cache/isaaclab-assets/rsl_rl_one_update_mjlab.json`,
  `.cache/isaaclab-assets/rsl_rl_one_update_isaaclab.json`, and
  `tests/test_rsl_rl_one_update_probe_contract.py`.

- [x] **T26 (P1, runtime gate)** — Re-run the official IsaacLab RSL-RL
  entrypoint from the current checkout, then reload both the fresh smoke
  checkpoint and the independently strict trained checkpoint through the
  evaluation harness. The 64-env/5-iteration smoke wrote `model_0.pt` and
  `model_4.pt` with finite losses and no `nan_state` terminations. The smoke
  checkpoint is correctly classified as untrained and fails motion-quality
  gates, while the current-code replay of T22 `model_999.pt` passes all six
  canonical 300-step cases with zero resets and max tilt below `0.18 rad`.
  Evidence:
  `.cache/isaaclab-assets/velocity_flat_command_battery_current_smoke_model4.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_current_strict_model999_300.json`,
  and `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_21-07-34/`.

## 26. Failure modes and verification

```text
config/DR mismatch -> seeded distribution test -> ledger BLOCKED, no long run
action/BAM mismatch -> actuator bench + target trace -> no reward retuning
sensor ABI mismatch -> 61D golden fixture -> export gate fails loudly
critic/actor mix-up -> group-shape/config test -> trainer startup fails
PhysX contact/solver delta -> deterministic physics battery -> BACKEND_DELTA
same-step external torque unavailable -> force timing probe -> BACKEND_DELTA
long-run regression -> fixed command battery + video -> baseline rejected
```

Every failure row must state whether it is tested, handled, and visible in the
report. Silent fallback or reward tuning is a parity failure.

## 27. Worktree execution order

The work has independent lanes until task wiring:

| Lane | Modules | Depends on |
| --- | --- | --- |
| A: actuator/asset | `actuators/`, `assets/`, USD inspection | — |
| B: task semantics | `tasks/`, mjlab source comparison | — |
| C: evaluation | `scripts/isaaclab/`, `tests/`, parity ledger | A/B contracts |
| D: training | IsaacLab runner and logs | A + B + C |

Launch A and B in parallel worktrees. Merge them before C; launch D only after
the ledger, smoke, and battery gates pass. A and B should not edit the same task
config lines without coordination.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | Not run; user supplied the scope decision |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | Not run |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 2 plan gaps fixed: explicit parity ledger/acceptance rule and RSL-RL version probe |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | Not applicable; backend-only plan |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | Not run |

**VERDICT:** ENG CLEARED — ready to implement the strict semantic parity slice.

NO UNRESOLVED DECISIONS
