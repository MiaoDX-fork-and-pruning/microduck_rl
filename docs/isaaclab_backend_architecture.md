# IsaacLab backend architecture for Microduck

Status: **proposal**  
Scope: add IsaacLab/PhysX as a second simulator backend inside this repository  
Reference backend: existing `mjlab_microduck` implementation  
Non-goal: replace mjlab or change the deployed policy ABI in the first phase

## 1. Summary

This repository should evolve into a **multi-backend RL monorepo** rather than creating a separate IsaacLab project.

The intended relationship is:

```text
                         shared Microduck semantics
                    joint order / HOME / policy ABI
                    command meaning / task parameters
                    actuator equations / test fixtures
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
      mjlab / MuJoCo Warp                 IsaacLab / PhysX
      current reference backend           secondary backend
               |                                 |
               +----------------+----------------+
                                |
                          policy export
                                |
                     existing robot runtime
```

The two backends do not need to share one Python virtual environment, one physics implementation, or identical framework glue. They do need to implement the same hardware-facing policy contract and should share pure specifications and deterministic parity tests wherever practical.

The initial objective is not feature parity with every existing task. The first objective is to establish a trustworthy IsaacLab foundation for the normal-foot robot:

1. load the correct Microduck robot in PhysX;
2. reproduce the existing 50 Hz policy interface;
3. reproduce the 61D actor observation and 14D action semantics;
4. reproduce the XL330/BAM actuator model closely enough for controlled cross-simulator tests;
5. train the flat velocity task;
6. compare MuJoCo, PhysX, and hardware-facing outputs with explicit parity metrics.

Only after those gates pass should sit/stand, recovery, ground pick, kick, roulade, backlash, and rollers be added.

## 2. Why keep both backends in one repository

The simulator wrappers are different, but the pieces that change most often during Microduck development are shared concepts:

- joint names and ordering;
- HOME/default pose;
- action scale and target semantics;
- observation layout and normalization assumptions;
- command semantics;
- actuator constants;
- domain-randomization ranges;
- reward targets and task constants;
- success criteria and evaluation metrics;
- generalist-policy interfaces;
- sim2real lessons encoded as tests and invariants.

Keeping these changes in one repository allows a single pull request to show the impact on both backends and on parity tests. This is preferable to maintaining synchronized changes across two repositories.

The repository boundary does **not** imply a shared runtime environment. Source ownership and environment isolation are separate concerns.

## 3. Proposed repository layout

The existing `src/mjlab_microduck` package remains the reference implementation. Add a sibling package and introduce a common package only when sharing is genuinely useful.

```text
microduck_rl/
├── src/
│   ├── mjlab_microduck/             # existing reference backend
│   │   ├── actuator/
│   │   ├── robot/
│   │   └── tasks/
│   │
│   ├── isaaclab_microduck/          # new PhysX backend
│   │   ├── assets/
│   │   ├── actuator/
│   │   ├── mdp/
│   │   └── tasks/
│   │
│   └── microduck_common/            # introduce incrementally, not upfront
│       ├── constants.py
│       ├── policy_abi.py
│       ├── behavior.py
│       └── actuator/
│
├── scripts/
│   ├── isaaclab/
│   └── parity/
│
├── tests/
│   ├── isaaclab/
│   └── parity/
│
└── docs/
```

Do not reorganize the existing mjlab tree merely to make the layout visually symmetric. New abstraction must be earned by real duplicated semantics.

## 4. Environment and dependency isolation

Use one Git repository with **separate uv environments** for the simulator stacks.

A reasonable developer workflow is:

```text
.venv-mjlab/       # current mjlab / MuJoCo Warp stack
.venv-isaaclab/    # Isaac Sim / IsaacLab / PhysX stack
```

The exact packaging mechanism should be selected after checking dependency resolution against the current Python, Torch, CUDA, Warp, and aarch64 constraints. Acceptable implementations include:

- root dependency groups if one lock solution is clean;
- uv workspace members for backend-specific packages;
- backend-specific project files/locks while keeping all source in this repo.

Do **not** make "one lockfile" a design requirement. If IsaacLab and mjlab require incompatible Torch/CUDA/Python solutions, preserve independent environments and keep the source monorepo.

The commands should eventually be simple and explicit, for example:

```bash
./scripts/dev/setup_mjlab_env.sh
./scripts/dev/setup_isaaclab_env.sh

./scripts/dev/run_mjlab pytest ...
./scripts/dev/run_isaaclab pytest ...
```

The wrapper names are illustrative; local implementation may use `uv run --project ...` or workspace equivalents.

## 5. Backend ownership and source of truth

During the first phases:

- **mjlab remains the sim2real reference backend**;
- **IsaacLab is an alternative implementation under validation**;
- real-robot behavior and existing successful mjlab policies win when a simulator disagreement is unresolved.

This avoids changing a proven task specification merely because PhysX gives a more convenient result.

Once IsaacLab policies have completed equivalent real-hardware validation, the two backends may be treated as peers.

## 6. Compatibility levels

Parity should be discussed using explicit levels rather than an ambiguous claim that the simulators are "the same".

### L1 — Policy ABI compatibility

Required before training:

- 50 Hz control frequency;
- same 14 policy-controlled joints and order;
- same HOME pose;
- same 61D actor observation order and units;
- same previous-action semantics;
- same 13D command block semantics;
- same 14D action meaning;
- same deployed ONNX input/output shapes and metadata requirements.

L1 is a hard requirement.

### L2 — Task compatibility

Required before comparing training quality:

- same command distributions;
- same zero-command and rare-command sampling intent;
- same reward definitions and signs;
- same reset-state intent;
- same curriculum boundaries/targets;
- same termination semantics;
- same success metrics.

Framework-specific implementation may differ as long as the semantics match.

### L3 — Statistical physics compatibility

A research/engineering target, not exact step-by-step equality:

- comparable standing equilibrium;
- comparable free-fall and recovery dynamics;
- comparable actuator response curves;
- comparable contact/slip behavior;
- comparable push response;
- comparable rollout distributions under the same command and DR envelopes.

Do not require identical trajectories from MuJoCo and PhysX.

## 7. Policy ABI must remain hardware-facing

The IsaacLab backend must not expose privileged simulator state to the deployed actor merely because it is available.

The initial actor contract remains the current 61D family:

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

The command block remains:

```text
twist       3
head pose   4
body pose   6
```

Critic observations may be privileged and backend-specific, but actor inputs must remain deployable on the robot.

Do not copy the Open Duck Mini IsaacLab example's large history observation or actor-side base linear velocity into Microduck v1.

## 8. Action semantics

IsaacLab must reproduce the current Microduck action meaning, not generic rescaling to full joint limits.

Conceptually:

```text
raw policy action[14]
        |
        v
HOME + action_scale * action
        |
        v
actuator target
```

Any per-skill tuning retained by the current runtime must be explicitly accounted for in the task/backend contract rather than hidden in framework defaults.

A custom IsaacLab action term is preferable to `JointPositionToLimitsActionCfg` if the standard term does not reproduce this mapping exactly.

## 9. Robot asset strategy

Do not manually maintain an unrelated USD robot revision.

The Microduck mechanical source of truth remains the current robot/Onshape/MJCF pipeline. The IsaacLab backend should add a reproducible conversion or generation process for USD and validate critical physical metadata.

At minimum verify:

- joint names/order;
- joint axes;
- joint limits;
- body mass;
- inertia tensors;
- body transforms;
- collision geometries;
- center-of-mass locations;
- passive-joint naming;
- contact geometry relevant to feet, trunk, head, mouth, and rollers.

Generated USD may be committed for reproducibility/startup speed, but its source revision and generation procedure must be recorded.

## 10. BAM / XL330 actuator strategy

This is the highest-priority physics component of the port.

Do not use a generic IdealPD actuator as the final sim2real model.

IsaacLab should implement a vectorized explicit actuator compatible with the current BAM model, including the portions currently relied on for sim2real, such as:

- target-position control law used by BAM;
- voltage dependence;
- back-EMF behavior;
- torque/effort saturation;
- friction model;
- actuator friction randomization;
- battery-voltage variation/sag when represented by the reference backend;
- armature/inertia effects where applicable.

Prefer to extract **pure actuator math** into `microduck_common` only if both backends can call the same tensor function without distorting either framework integration.

Otherwise keep two wrappers/implementations and validate them against common numerical fixtures.

The key requirement is mathematical parity, not DRY code.

## 11. Backlash

Backlash is intentionally deferred until the normal-foot non-backlash backend is stable.

When implemented, prefer the same physical interpretation as the current reference model: an explicit passive/serial degree of freedom whose output-side state is what the policy/rewards observe when that matches the hardware encoder view.

Avoid implementing only an actuator dead-zone if doing so changes which state is observed by the policy.

## 12. Tasks and port order

Port tasks incrementally:

```text
M0  robot + ABI + BAM bench
M1  velocity flat
M2  velocity rough / DR parity
M3  VelStand (walking + recovery)
M4  SitStand
M5  GroundPick
M6  BallKick
M7  Roulade
M8  Backlash variants
M9  Rollers
```

Do not begin with AMP/reference-motion training from the Open Duck Mini IsaacLab example. The Microduck backend should reproduce the current direct task-RL approach first.

AMP or motion-prior experiments can remain future optional work.

## 13. Cross-backend tests

Add deterministic fixtures where exact parity is meaningful.

### ABI fixtures

Given fixed:

- IMU state;
- joint positions/velocities;
- previous action;
- command;

both backend adapters should produce the same 61D actor vector.

### Action fixtures

Given fixed raw policy action, both backends should produce the same intended joint target before simulator-specific actuator dynamics.

### Actuator fixtures

Sweep representative values of:

- position error;
- joint velocity;
- battery voltage;
- friction scale;
- target changes.

Compare the computed actuator output/torque against the reference implementation with explicit tolerances.

### Reward fixtures

For reward functions that can be represented as pure tensors, compare outputs on synthetic states. Do not force framework-specific contact/state lookup code into one abstraction solely for test reuse.

## 14. Physics parity battery

Before trusting RL results, build deterministic single-robot experiments:

1. HOME-pose settle;
2. static joint target sweep;
3. single-joint step response;
4. servo response under voltage/friction variation;
5. free fall from fixed pose;
6. small-angle balance perturbation;
7. horizontal push response;
8. foot sliding under known load;
9. trunk/head contact impulse cases;
10. recovery-relevant prone/side contacts.

Record comparable metrics rather than expecting identical trajectories.

## 15. Training and evaluation parity

For each task that is ported, maintain a backend-neutral evaluation report with at least:

- command tracking error;
- episode/success rate;
- fall rate;
- action-rate statistics;
- torque/effort statistics;
- contact/slip metrics where relevant;
- task-specific success metrics;
- inference/export shape checks.

When both simulators train a candidate, compare per-skill metrics and qualitative video before interpreting aggregate reward.

## 16. CI tiers

Full Isaac Sim training must not become a required check for every ordinary pull request.

### Tier 0 — pure Python / semantics

Run on ordinary CI:

- ABI schema tests;
- joint/HOME constants;
- pure reward math;
- pure BAM math/fixtures;
- dataset/schema tests.

### Tier 1 — backend config/import

Run in backend-specific environments:

- task registration;
- asset/joint resolution;
- observation/action dimensions;
- environment construction where feasible.

### Tier 2 — physics smoke

GPU/simulator runner, manual or scheduled:

- small environment count;
- a few seconds of stepping;
- 64-env / 5-iteration smoke when training is wired.

### Tier 3 — full training

Never block normal PRs. Run manually/local/remote as experiments.

## 17. Generalist-policy relationship

The IsaacLab work should not block or redefine the separate generalist-policy program.

Long term, a useful target is:

```text
same generalist behavior condition
same hardware-facing actor ABI contract/version
        |
   +----+----+
   |         |
 mjlab     IsaacLab
```

IsaacLab may become valuable for large multi-task experiments, scene diversity, cameras, navigation, and embodied-AI work. The initial port should remain focused on backend parity rather than combining both research projects at once.

## 18. Explicit non-goals for v0

Do not include these in the first IsaacLab milestone:

- replacing mjlab;
- SONIC-style universal motion tracking;
- AMP/reference-motion locomotion as the default path;
- camera/VLA training;
- rollers;
- backlash;
- every existing task;
- one universal virtual environment;
- exact MuJoCo/PhysX trajectory equality;
- production runtime changes.

## 19. Definition of architecture success

The architecture phase is successful when:

1. both backends can coexist in one checkout without dependency confusion;
2. current mjlab workflows remain unchanged and green;
3. IsaacLab can load the correct Microduck asset;
4. L1 policy ABI parity is locked by tests;
5. BAM actuator parity has deterministic numerical coverage;
6. a flat velocity environment can be constructed at 50 Hz with the same command/action/observation semantics;
7. the team can run long IsaacLab experiments locally without requiring those dependencies for mjlab-only developers.

The companion implementation plan should be used for the staged local development program.