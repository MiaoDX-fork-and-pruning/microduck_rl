# IsaacLab backend: local implementation and validation plan

Status: **proposed execution plan**  
Architecture reference: [`isaaclab_backend_architecture.md`](isaaclab_backend_architecture.md)  
Primary execution environment: local workstation(s) with Isaac Sim/IsaacLab-capable GPU  
Reference behavior: current `mjlab_microduck` tasks and real-robot sim2real lessons

## 1. Purpose

This plan is intended for long-running local development with Codex or another coding agent. It deliberately separates deterministic engineering work from expensive empirical training work.

The goal is to add an IsaacLab/PhysX backend to this repository while preserving the existing mjlab backend as the reference path.

A successful program produces a second backend that implements the same Microduck hardware-facing policy semantics and can independently train deployable policies. It is not necessary to port every task before the work is useful.

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

Build or generate the Microduck USD from the current mechanical/model source. Add scripts or metadata documenting the generation path.

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

When a PhysX-trained specialist reaches deployment quality:

- export with the same actor observation normalization semantics;
- validate ONNX input/output shapes;
- attach backend/task/ABI metadata;
- run deterministic PyTorch-vs-ONNX parity vectors;
- rehearse in a CPU/simulator path if available;
- use existing runtime safeguards and specialist fallback for first robot tests.

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

### Task H — first full walking run

Deliver:

- baseline manifest;
- training run;
- metrics report;
- video;
- comparison to current mjlab baseline.

### Task I — decide whether to continue

Proceed to VelStand only if:

- no unresolved ABI/actuator/model issue remains;
- walking is qualitatively credible;
- task metrics are competitive enough to justify further porting.

Otherwise fix the foundation first.

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

The first program can be considered successful without complete feature parity if all of the following hold:

1. one repository supports reproducible mjlab and IsaacLab developer environments;
2. existing mjlab workflows and tests remain stable;
3. Microduck USD/model parity is documented and tested;
4. actor ABI/action semantics are golden-tested across backends;
5. BAM actuator behavior is numerically and dynamically benchmarked;
6. Velocity Flat trains successfully in IsaacLab using direct RL;
7. VelStand or one other contact-rich task demonstrates the backend is not limited to simple walking;
8. evaluation reports make simulator differences explicit;
9. a PhysX-trained policy can enter the existing controlled hardware-validation process without runtime API redesign.

After that point, additional tasks, backlash, rollers, generalist training, cameras, navigation, and embodied-AI work can be prioritized based on product/research needs rather than treated as prerequisites.