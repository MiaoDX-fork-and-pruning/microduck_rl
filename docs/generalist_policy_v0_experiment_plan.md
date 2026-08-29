# Generalist policy v0: distillation and integration plan

Status: **proposed execution plan**  
Architecture reference: [`generalist_policy_v0_design.md`](generalist_policy_v0_design.md)

This plan starts after the independent specialist execution plan has produced
accepted checkpoints, ONNX files, evaluation reports, and manifests. It covers
only distillation and Generalist integration; specialist training and the long
switching demo live in [`specialist_policy_demo_plan.md`](specialist_policy_demo_plan.md).
Each milestone below can be stopped and evaluated independently.

## 1. Working principles

1. Read `AGENTS.md` before changing task code.
2. Preserve existing specialist behavior until its baseline has been captured.
3. Never launch a long training run before the 64-env, 5-iteration smoke test.
4. Change one primary variable per experiment.
5. Report per-skill results; never use aggregate reward as the only decision signal.
6. Keep datasets, checkpoints, videos, and generated reports out of Git.
7. Commit implementation code separately from experiment results when practical.
8. Do not modify the production 61D observation contract.
9. Keep teacher checkpoints immutable and content-addressed.
10. Stop at distillation if it meets the product goal; do not add PPO or MoE by default.

## 2. Local workspace setup

A suggested layout:

```text
~/work/microduck_rl/                  # clean develop checkout
~/work/microduck_rl-generalist/       # experiment worktree
~/data/microduck-generalist/          # untracked datasets/checkpoints/reports
```

Example:

```bash
git clone https://github.com/MiaoDX-fork-and-pruning/microduck_rl.git
cd microduck_rl
git checkout develop
git worktree add ../microduck_rl-generalist -b exp/generalist-v0
cd ../microduck_rl-generalist

uv sync
uv run list-envs
uv run --with pytest pytest tests/
```

Before any training change:

```bash
uv run train Mjlab-VelStand-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5
```

Use a fresh `uv sync` at least once in a clean environment before declaring infrastructure complete.

## 3. Artifact layout and manifests

Use a local artifact root, for example:

```text
~/data/microduck-generalist/
├── teachers/
│   ├── manifest.yaml
│   └── *.onnx / checkpoint references
├── baselines/
│   ├── suite-v1.json
│   └── videos/
├── datasets/
│   └── <dataset-id>/
│       ├── manifest.json
│       ├── shard-00000.npz
│       └── ...
├── runs/
│   └── <run-id>/
│       ├── run.yaml
│       ├── metrics.json
│       ├── notes.md
│       └── checkpoint references
└── reports/
    └── comparison-<date>.md
```

Every dataset/run manifest should include:

- repository commit;
- dirty-tree status;
- exact command;
- task ids;
- teacher ids and hashes;
- random seeds;
- environment count;
- observation schema;
- model architecture;
- DR configuration;
- sample counts by behavior/state bucket;
- start/end timestamps;
- WandB run ids when used.

For teacher/DAgger data, the manifest must also identify the replay-state
format: command values and timers, phase state, previous action, delay/history
buffers, episode time, and task latches. A sample is not reproducible from
qpos/qvel alone. The schema manifest also freezes the v2 normalization transform
and field mask before M2 collection begins.

The manifest is part of the result. A run without provenance is not reusable evidence.

## 4. Branch and patch strategy

Recommended reviewable patches:

1. `feat: define generalist behavior condition and schema tests`
2. `test: add deterministic specialist evaluation suite`
3. `feat: collect versioned teacher rollouts`
4. `feat: train offline generalist student`
5. `feat: add student-state teacher relabeling`
6. `feat: add generalist task router and transition curriculum` — only if required
7. `feat: export generalist metadata and parity vectors`

Avoid one branch that simultaneously changes observation layout, rewards, runner, exporter, and runtime assumptions.

## 5. Milestone M0 — import and freeze specialist evidence

### Objective

Import the outputs of the independent specialist plan and freeze the exact
teacher baseline against which every Generalist candidate is compared. Do not
retrain specialists here.

### Deliverables

- `teachers/manifest.yaml` with the selected checkpoint/run for each skill;
- exported ONNX files or immutable references and SHA-256 hashes;
- a deterministic evaluation suite with fixed seed lists;
- one JSON report per specialist;
- representative videos for human review;
- a summary table of baseline mean, variance, and failure modes.

### Teacher selection

Prefer policies known to work in the current runtime path. Record both the training checkpoint and exported ONNX hash.

Initial set:

```text
VelStand          stand / locomotion / recovery
SitStand          sit / rise
GroundPick        ground-pick cycle
BallKick          kick teacher
Roulade           roll teacher
Velocity          locomotion quality reference
```

### Evaluation batteries

#### Locomotion

- zero command;
- forward/backward buckets;
- lateral buckets if supported;
- yaw-only turn-in-place;
- combined translation/yaw;
- push perturbations;
- flat and rough separately, but flat is the v0 gate.

Metrics:

```text
vx/vy/yaw tracking error
fall rate
time upright
mean/peak tilt
joint/action rate
foot slip/contact statistics
```

#### Recovery

Spawn buckets:

```text
face down
face up
left side
right side
crouch / near-upright frontier
natural fall from locomotion
```

Metrics:

```text
success rate
time to recovered state
final height/tilt
post-recovery locomotion success
impact proxies
```

#### Sit/stand

```text
stand -> sit
sit hold
sit -> stand
repeated toggle
command during transition
```

#### Ground pick

```text
mouth touch success
minimum mouth height
return-to-stand success
phase completion time
head/body impact
```

#### Kick

```text
ball displacement/velocity
correct-side contact
robot landing/upright rate
post-kick stand stability
```

#### Roulade

```text
full sagittal rotation
landing upright rate
side deviation
repeat/chained request stability
```

### Exit gate

M0 is complete only when the same evaluation command reproduces stable results from the frozen teacher artifacts.

The M0 manifest must freeze episode counts per behavior/state bucket, seed
lists, DR/reset settings, success definitions, and the confidence-interval
method used by later gates. For baseline rates below 5%, gates use an absolute
floor and report the raw rate.

## 6. Milestone M1 — schema and adapters, no policy training

### Objective

Implement the typed v2 behavior condition and prove that it maps exactly to legacy teacher inputs.

### Suggested files

```text
src/mjlab_microduck/generalist/
├── __init__.py
├── behavior.py
├── observation.py
└── legacy_adapter.py

tests/
├── test_generalist_obs_layout.py
├── test_generalist_behavior_encoding.py
└── test_generalist_legacy_adapter.py
```

### Required tests

1. v2 observation width is exactly 71.
2. Every documented block lands at the documented offsets.
3. behavior one-hot has exactly one active value.
4. non-applicable `phase`, `posture_target`, and `side` fields are zero.
5. ground-pick phase maps to the exact legacy command encoding.
6. sit/stand posture target maps to the exact legacy flag.
7. kick/roulade legacy commands remain all-zero where expected.
8. joint ordering and previous-action semantics match v1.
9. NaN/Inf inputs are caught or sanitized consistently with existing actor observations.
10. golden vectors are serializable for later Rust parity tests.

### Exit gate

No existing task registration, observation width, exported ONNX, or test changes behavior. The new code is unused except by tests/tools.

## 7. Milestone M2 — teacher rollout collector

### Objective

Generate balanced, versioned offline data from the existing specialists.

### Suggested command

```bash
uv run scripts/collect_generalist_teacher_data.py \
  --teacher-manifest ~/data/microduck-generalist/teachers/manifest.yaml \
  --output ~/data/microduck-generalist/datasets/g0-v1 \
  --behaviors stand,locomotion \
  --num-envs 1024 \
  --steps-per-shard 100000 \
  --seed 100
```

### Collector requirements

- run the original task environment for each teacher;
- obtain the exact v1 actor observation used by the teacher;
- construct v2 student observation from the same simulator state;
- record raw teacher action before action scaling;
- record behavior/state bucket labels;
- record terminal/success labels;
- reject non-finite samples;
- produce counts and summary statistics per shard;
- support deterministic small collection for tests;
- support headless GPU collection for production datasets.

### Dataset validation

Provide a script such as:

```bash
uv run scripts/inspect_generalist_dataset.py <dataset-dir>
```

It should report:

- schema/manifest version;
- shapes/dtypes;
- finite-value checks;
- behavior balance;
- command/phase/side ranges;
- per-joint action distribution;
- observation mean/std;
- success/failure counts;
- duplicate/constant-field warnings.

### Initial collection sizes

Do not begin with the largest possible dataset. Suggested progression:

```text
smoke       10k samples per behavior
debug       100k samples per behavior
baseline    1M effective balanced samples across G0
```

Dynamic skills should be balanced by phase/state bucket rather than given equal raw frame counts.

### Exit gate

A dataset shard can be loaded, replayed, and traced back to an immutable teacher and repository commit. Golden samples reproduce the stored teacher action within tolerance.

Replay tests must restore the recorded command-manager/episode state and prove
that the teacher emits the stored observation and action. Physical state alone
is not an acceptable replay fixture.

## 8. Milestone M3 — offline student baseline

### Objective

Determine whether one dense actor can imitate the initial teachers before adding online RL complexity.

### G0 scope

Start with:

```text
STAND
LOCOMOTION
recovery states labeled under STAND
```

This is intentionally close to VelStand and validates the new schema/training path.

### Baseline model

```text
input       71
hidden      512, 256, 128
activation  ELU
output      14 raw actions
history     H1
```

Run at least two initialization baselines:

- random initialization;
- expanded VelStand actor initialization.

### Training objective

Start with unweighted action MSE. Log:

- overall train/validation MSE;
- per-behavior MSE;
- per-joint MSE;
- error by state bucket;
- error percentiles, not only mean;
- output range and saturation.

### Experiment table

| ID | Initialization | History | Sampling | Primary question |
|---|---|---:|---|---|
| BC-00 | random | H1 | balanced | can the architecture fit G0? |
| BC-01 | VelStand expansion | H1 | balanced | does teacher initialization preserve gait/recovery? |
| BC-02 | best above | H4 | balanced | does short history improve fit materially? |

Do not add MoE or PPO during this milestone.

### Offline gate

A model may advance to rollout only when every included behavior has acceptable validation error and no joint/behavior is hidden by aggregate averages.

### Rollout gate

Run the full M0 battery. Interpret outcomes:

| Observation | Likely cause | Next action |
|---|---|---|
| poor offline fit | capacity/schema/data | fix before rollout work |
| good offline fit, immediate drift/falls | covariate shift | DAgger |
| stand works, locomotion regresses | sampling/initialization | rebalance or VelStand init |
| nominal works, perturbations fail | missing frontier/DR data | targeted collection |

## 9. Milestone M4 — DAgger / student-state relabeling

### Objective

Train on states produced by the student rather than only teacher trajectories.

### Loop

```text
train student on current dataset
        |
        v
roll student in each task environment
        |
        v
query matching teacher on student state
        |
        v
append balanced disagreement/frontier samples
        |
        v
retrain and evaluate
```

### Control mixing

Initial iterations may execute a mixture:

```text
a_exec = beta * a_teacher + (1 - beta) * a_student
```

or probabilistically choose teacher/student control per step/episode. Record the exact method and schedule. Reduce teacher control as student stability improves.

### Sample prioritization

Prioritize:

- high teacher/student disagreement;
- near-fall states;
- recovery frontiers;
- transition boundaries;
- rare command buckets;
- skill-specific late phases;
- valid states where task metrics fail despite finite actions.

Do not let every DAgger round become mostly prone, invalid, or already-failed states.

### DAgger experiment table

| ID | Seed model | Executed control | Selection | Question |
|---|---|---|---|---|
| DG-00 | best BC | teacher-heavy | uniform student states | basic covariate-shift recovery |
| DG-01 | DG-00 | decaying teacher mix | disagreement-prioritized | data efficiency |
| DG-02 | best | student-only collection | frontier buckets | robustness at failure boundary |

### Exit gate

G0 meets the initial per-skill thresholds in the design document on fixed-seed rollouts. If G0 cannot meet them, do not add more skills.

## 10. Milestone M5 — incremental skill addition

Add one skill family at a time. For each addition:

1. freeze the previous candidate and baseline report;
2. collect balanced teacher data for the new skill;
3. train/distill with replay from all previous skills;
4. run every previous evaluation battery;
5. DAgger the new skill and any regressed old skill;
6. make a go/no-go decision before continuing.

### G1: Sit/stand

Primary questions:

- does explicit `posture_target` remove legacy command ambiguity?
- can one actor hold both sitting and standing equilibria?
- does adding sit/rise damage locomotion or recovery?
- can transitions run without resetting previous action/history?

Required transition battery:

```text
stand -> sit
sit -> rise
rise -> locomotion
locomotion -> stop -> sit
repeated target toggles at legal times
```

### G2: Ground pick

Primary questions:

- is explicit phase sufficient?
- does shared balance improve return-to-stand?
- does phase-conditioned behavior interfere with locomotion command weights?

Balance data across phase bins and include the return-to-stand tail.

### G3: Kick

Start with mode + side. Keep intensity out of the schema until a teacher/training distribution varies it.

Primary questions:

- can one side parameter replace separate left/right sessions?
- is mirrored data physically valid with the current robot/model/ball setup?
- does the dynamic action distribution damage stand/locomotion?

Evaluate kick outcome and robot landing separately.

### G4: Roulade

This is the highest-risk merge and should be last.

Primary questions:

- can the same actor represent the dynamic roll without degrading recovery?
- does mode-only triggering reproduce the teacher consistently?
- does the landing naturally hand off to shared stand/recovery?
- is dense capacity sufficient, or is this the first measured case for an internal expert?

## 11. Milestone M6 — transition curriculum

### Objective

Move from “one file containing isolated modes” to a controller that safely changes modes in one episode.

Before building the full superset environment, pass a two-behavior CPU proof
slice showing per-environment behavior state, command/phase updates, reset and
termination routing, reward masking, and episode accounting. If this cannot be
implemented without shared-manager mutation, stop and use separate task
collectors with a transition harness.

### Transition scheduler in simulation

Implement a training-only scheduler that samples legal edges and dwell times from the product transition graph. It should expose the same condition fields the future runtime will emit.

Suggested initial graph:

```text
STAND <-> LOCOMOTION
STAND <-> SIT_STAND
STAND -> GROUND_PICK -> STAND
STAND -> KICK -> STAND
STAND -> ROULADE -> STAND
fallen -> STAND -> LOCOMOTION
```

### Transition-specific data

Collect teacher labels around boundaries by querying the source or destination teacher according to the scheduler state. Where neither specialist defines an ideal blend, use task metrics and optional RL fine-tuning rather than inventing arbitrary interpolated labels.

### Metrics

```text
source behavior success
destination behavior success
fall within 0.5/1/2 seconds of switch
peak |delta action|
peak |delta target|
settling time
resume-command tracking
illegal-interrupt handling
```

### Exit gate

Common transitions pass the fixed battery without a reset and without a regression beyond the design thresholds.

## 12. Milestone M7 — optional multi-task PPO

Do this only if BC/DAgger has good action fidelity but task outcomes remain measurably below target.

### Implementation preference

Use one all-collision superset environment with a per-environment behavior state:

```text
src/mjlab_microduck/tasks/microduck_generalist_env_cfg.py
src/mjlab_microduck/tasks/generalist_task_router.py
```

The environment should compose existing MDP functions and proven DR/noise, not fork them.

### First PPO scope

Start with G0 or G1, not all skills. Initialize the actor from the best distilled checkpoint. Keep reward changes minimal and traceable.

### Required logging

- active behavior fraction;
- episode counts by behavior;
- reward mass by term and behavior;
- return and advantage stats by behavior;
- success metrics by behavior;
- curriculum stage markers;
- old-skill regression metrics during training.

### Critic experiments

Run in order:

1. shared behavior-conditioned critic;
2. larger shared critic;
3. separate value heads only if value error/advantages conflict by behavior.

### Sampling experiments

Run in order:

1. uniform behavior sampling;
2. fixed weights based on episode length/data mass;
3. adaptive failure-based sampling.

Do not change critic structure and sampling policy in the same run.

## 13. Architecture escalation gates

### Add H4 history when

- H1 cannot resolve transition/recovery ambiguity;
- H4 improves rollout metrics consistently, not only offline MSE;
- board/runtime memory and input construction remain simple.

### Increase dense capacity when

- training and validation loss underfit all tasks;
- more data does not close the gap;
- inference estimates remain comfortably inside budget.

### Add soft MoE when

- each behavior fits well in isolation;
- joint dense training causes repeatable behavior-family regressions;
- balancing, replay, and larger dense capacity fail;
- per-task gradient/loss evidence supports specialization.

### Consider a motion-reference tracker when

- product requirements include arbitrary motion clips or community motion packs;
- new behaviors must be added without controller retraining;
- behavior ids and hand-maintained transitions become the dominant scaling cost.

None of those is a v0 prerequisite.

## 14. Failure triage guide

### “The loss is low but the robot falls”

Likely covariate shift. Run student-state collection and inspect error on states immediately before failure.

### “Walking became worse after adding a short skill”

Check sample counts, normalizer statistics, and replay balance. Short skill oversampling can still move shared features aggressively if its actions have larger magnitude.

### “The new skill works but old skills are forgotten”

Keep immutable replay from every previous milestone; compare per-behavior batches and learning rates; consider freezing lower layers only as a diagnostic, not a permanent assumption.

### “Transitions fail although both endpoints work”

Add boundary data and in-episode switches. Do not immediately enlarge the network.

### “Recovery works only under STAND”

Decide product semantics. v0 may intentionally force `STAND` when fallen. Training recovery under every behavior is a later robustness enhancement.

### “Ball kick distillation is unstable”

Separate policy imitation from outcome evaluation. The actor is ball-blind, so verify ball placement/reset parity before blaming the student.

### “Roulade breaks everything else”

Keep it outside the first release candidate. Measure whether more capacity or a soft dynamic expert solves the conflict. One model is a product optimization, not a reason to ship worse control.

### “Simulation passes, hardware does not”

Check observation layout, normalizer, action scale/gain/filter parity, inference latency, sensor biases, and DR coverage. Do not retune rewards before proving the runtime contract.

## 15. ONNX parity and export tests

Before runtime work, add a fixed battery of v2 observations containing:

- nominal stand;
- locomotion command extremes;
- each behavior one-hot;
- phase quadrants;
- posture targets;
- kick side values;
- perturbation/fallen states;
- random finite observations within physical ranges.

For each vector, compare PyTorch and ONNX outputs with documented tolerance.

Verify metadata:

```text
policy_family
policy_contract
obs_schema_version
obs_width
action_width
behavior_order
joint_names
training commit/run/checkpoint
```

Export tests must fail when behavior order or schema metadata is missing, not merely when tensor width is wrong.

## 16. Runtime handoff package

Runtime activation is a separate repository effort. This project should hand off:

- the exact v2 layout specification;
- Python golden input vectors;
- expected ONNX outputs;
- behavior/control-profile table;
- active-behavior lifecycle assumptions;
- model metadata contract;
- specialist baseline and generalist comparison report;
- measured inference requirements;
- known unsupported transitions;
- fallback teacher bundle.

Recommended hardware phases:

```text
shadow: specialists actuate, generalist logs only
canary: explicit configuration selects generalist
release candidate: repeated behavior/transition battery
fallback: specialist bundle remains installable
```

## 17. Run note template

Every meaningful experiment should end with a note similar to:

```markdown
# <run-id>

## Hypothesis
One sentence; one primary variable.

## Code and data
- commit:
- dataset manifest:
- teacher manifest:
- parent checkpoint:

## Command
Exact shell command.

## Changes from comparison run
Only the intentional differences.

## Smoke result
Build/step/NaN/export status.

## Training result
Per-behavior learning and loss summary.

## Evaluation
Table against fixed specialist/generalist baseline.

## Video observations
Specific behavior, contacts, failure clusters.

## Decision
keep / reject / repeat, with the next single question.
```

Do not use “looks better” without naming the metric or observed physical change.

## 18. Initial experiment queue

A practical first queue:

| Order | ID | Scope | Main variable | Required decision |
|---:|---|---|---|---|
| 1 | M0 | specialist manifest | none | import and freeze reproducible baselines |
| 2 | SCHEMA-00 | infrastructure | v2 71D | adapters/golden vectors correct? |
| 3 | DATA-G0-00 | G0 | small dataset | collector/replay correct? |
| 4 | BC-00 | G0 | random init | can dense H1 fit? |
| 5 | BC-01 | G0 | VelStand init | does it preserve baseline better? |
| 6 | DG-00 | G0 | DAgger | does rollout gap close? |
| 7 | HIST-00 | G0 | H4 | only if H1 transition/recovery fails |
| 8 | G1-BC | +sit/rise | skill addition | regression acceptable? |
| 9 | G1-DG | +sit/rise | DAgger | transitions acceptable? |
| 10 | G2 | +ground pick | phase skill | explicit phase sufficient? |
| 11 | G3 | +kick | side condition | one side-conditioned actor viable? |
| 12 | G4 | +roulade | dynamic skill | dense actor still viable? |
| 13 | TRANS-00 | all accepted | switch curriculum | one-episode transitions pass? |
| 14 | PPO-00 | smallest failing set | RL fine-tune | only if distillation misses gates |
| 15 | EXPORT-00 | candidate | ONNX/metadata | reproducible runtime handoff? |

P0 is intentionally sequential because it calibrates the infrastructure. Once
P0 passes, A1 and B1 are parallel waves; do not serialize independent policy
runs. Large hyperparameter sweeps are still forbidden until the baseline
pipeline is trustworthy, but independent task IDs should fill the available
GPU lanes.

## 19. Completion definition for the local program

The local program is ready for a runtime shadow PR when:

1. the chosen behavior set has one exported ONNX;
2. every included skill passes the agreed relative gate against its teacher;
3. common transitions pass without reset;
4. all observation/action/metadata parity tests pass;
5. long-run evaluation is finite and NaN-free;
6. the model and inference path fit the target budget estimate;
7. the candidate, datasets, and reports are reproducible from manifests;
8. unsupported skills/transitions are explicitly listed;
9. the specialist fallback bundle remains available;
10. the team agrees that further complexity is justified by a measured deficiency, not by architecture preference.
