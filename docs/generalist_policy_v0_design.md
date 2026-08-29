# Generalist policy v0: a skill-conditioned controller for Microduck

Status: **proposal**  
Scope: `microduck_rl` training and export design; runtime integration is described only as an interface contract  
Target: one foot-mode policy for the current closed set of product skills  
Non-target: a SONIC-style open-motion foundation model

Companion document: [`generalist_policy_v0_experiment_plan.md`](generalist_policy_v0_experiment_plan.md)

## 1. Summary

Microduck currently deploys several specialist policies behind a runtime scheduler. The policies already share the important mechanical ABI:

- 50 Hz control;
- 14 policy-controlled servo joints;
- one 61-dimensional actor observation layout;
- one ONNX export path with the observation normalizer baked into the graph.

The remaining separation is semantic. The runtime chooses a network first, then overloads the same 13 command slots differently for walking, sit/stand, ground pick, kicks, and roulade. A single policy cannot infer which behavior an all-zero or repurposed command represents.

This proposal introduces a **closed-set, skill-conditioned generalist policy**. A retained high-level scheduler supplies an explicit behavior condition; one low-level policy supplies joint actions. The scheduler continues to own behavior validity, priority, phase generation, busy windows, interruption rules, and safety fallback. The policy owns dynamic coordination, balance, disturbance rejection, and transitions between trained behaviors.

The first target is deliberately smaller than a universal motion tracker:

```text
intent / gamepad / autonomous brain
                |
                v
       behavior scheduler
       - behavior id
       - parameters
       - phase / side
       - transition rules
                |
                v
     one conditioned policy
       obs_v2 -> action[14]
                |
                v
       existing safety layer
```

The recommended path is **teacher distillation first**, using the existing successful specialist policies, followed by DAgger-style student-state collection. Multi-task PPO is an optional later step, not the starting assumption. The project should stop at the simplest approach that meets per-skill and transition gates.

## 2. Why this is the right abstraction

### 2.1 The product problem is currently closed-set

The near-term behavior set is finite and named:

- stand and balance;
- velocity locomotion and turning;
- fall recovery;
- sit and rise;
- ground pick;
- left/right kick;
- roulade.

The product does not currently require arbitrary human motion tracking, user-uploaded animation clips, text-to-motion, or adding a new motion without retraining the controller. Those requirements would justify a future reference-motion tracker. They do not justify carrying motion datasets, retargeting, a motion encoder, and a generative planner into v0.

### 2.2 The existing specialists are assets, not technical debt

Each current policy is a trained teacher that already contains reward-design, curriculum, domain-randomization, and sim2real lessons. Replacing all of them with a multi-task environment trained from scratch would throw away the most expensive part of the project.

The v0 program should preserve those teachers as:

- behavior-quality baselines;
- rollout generators;
- action labelers for offline distillation;
- action labelers on student-visited states during DAgger;
- deployment fallback until the generalist is proven on hardware.

### 2.3 A single model does not remove the scheduler

A neural policy should not decide whether a kick is legal while the robot is sitting, whether a roulade may be interrupted halfway through, or whether a user command outranks shutdown. These are discrete authority and lifecycle decisions and are easier to audit in ordinary code.

The scheduler remains responsible for:

- selecting a requested behavior;
- rejecting invalid transitions;
- holding one-shot behaviors for their required window;
- generating ground-pick phase;
- setting kick side;
- deciding when a behavior is interruptible;
- forcing a safe stand/recovery request;
- hardware-mode selection;
- reporting the active behavior to clients.

The generalist policy is responsible for the continuous control problem after that decision.

## 3. Goals and non-goals

### 3.1 Goals

1. Export one ONNX policy for the normal-foot behavior set.
2. Preserve the current 14-dimensional action semantics and 50 Hz control rate.
3. Make behavior semantics explicit rather than overloading legacy command slots.
4. Match specialist behavior quality within defined per-skill tolerances.
5. Improve or at least preserve transitions by training on in-episode condition changes.
6. Share balance and recovery representations across behaviors.
7. Keep the existing specialists usable as teachers and runtime fallback.
8. Add no runtime behavior change until a candidate passes deterministic simulation gates.
9. Leave a clean extension point for a future motion-reference encoder without implementing one now.

### 3.2 Non-goals

- Supporting arbitrary motion clips or human retargeting.
- Reproducing SONIC-scale motion pretraining.
- Merging roller hardware into v0.
- Removing the high-level scheduler.
- Letting the learned policy bypass the existing safety layer.
- Rewriting existing specialist tasks before their baselines are captured.
- Changing the current 61D contract in place.
- Proving sim2real solely from simulator results.

## 4. Existing system constraints

The design must preserve the repository invariants in `AGENTS.md`:

- actor observations and actions must have one exact, tested layout;
- the 14-servo order must remain canonical even when passive joints are present;
- BAM actuator physics and the existing domain-randomization stack remain the sim2real base;
- observation normalization must be baked into exported ONNX;
- action filtering must match runtime behavior;
- reset randomization must not accumulate;
- reward views must agree with observation sensor views;
- every long run starts with the normal 64-env, 5-iteration smoke test.

The generalist work adds a new contract; it does not weaken those invariants.

## 5. Scope and rollout order

### 5.1 Canonical physical model

Use the all-collision foot model used by VelStand/StandUp/SitStand/GroundPick/BallKick/Roulade as the canonical v0 robot model. It can walk, lie on the floor, make shell contacts, and recover. The stripped walking model cannot express all target behaviors.

The first baseline question is therefore not “can every skill be merged?” but:

> Does the current VelStand/all-collision locomotion baseline meet the gait-quality threshold required for the generalist?

If it does not, fix or document that baseline before adding more skills.

### 5.2 Skill inclusion order

Do not merge all skills in the first run. Add them in increasing semantic and dynamic distance:

1. **G0** — stand, locomotion, turn, fall recovery (largely the existing VelStand problem).
2. **G1** — add sit and rise.
3. **G2** — add ground pick.
4. **G3** — add left/right kick.
5. **G4** — add roulade.

Rollers stay separate in v0 because they change contact mechanics and add passive wheel joints. A later hardware-mode-conditioned policy can be evaluated after the foot-mode generalist is stable.

### 5.3 Teacher mapping

Initial teacher candidates:

| Generalist behavior | Primary teacher | Notes |
|---|---|---|
| stand / locomotion / recovery | `Mjlab-VelStand-Flat-MicroDuck` | canonical all-collision base |
| locomotion quality reference | `Mjlab-Velocity-Flat-MicroDuck` | evaluation baseline only unless successfully transferred to all-collision |
| sit / rise | `Mjlab-SitStand-Flat-MicroDuck` | posture-target command |
| ground pick | `Mjlab-GroundPick-Flat-MicroDuck` | legacy phase encoding must be adapted |
| kick | `Mjlab-BallKick-Flat-MicroDuck` | actor is ball-blind; mirror data for the other side when valid |
| roulade | `Mjlab-Roulade-Flat-MicroDuck` | mode-selected episodic teacher |

Checkpoint identities, exact export commands, hashes, and measured baselines must be captured in a versioned teacher manifest before collection begins.

## 6. Observation and behavior contract

### 6.1 Do not mutate the 61D contract

The current 61D layout is production ABI v1. It remains unchanged for every specialist task and existing runtime bundle.

The generalist gets a new versioned actor observation, provisionally called `generalist-v0` / schema v2. Training, ONNX metadata, inference tools, and the future Rust runtime builder must all validate the same layout.

### 6.2 Logical behavior condition

The proposed v0 behavior vocabulary is:

```text
0  STAND
1  LOCOMOTION
2  SIT_STAND
3  GROUND_PICK
4  KICK
5  ROULADE
```

Recovery is initially not an externally selected mode. A fallen robot under `STAND` is expected to recover, using VelStand data. Later experiments may train recovery under every behavior condition, but high-level callers should always be able to force `STAND`.

The logical condition contains:

- `behavior_one_hot[6]`;
- `twist[3]`: forward, left, yaw rate;
- `head_pose[4]`;
- `body_pose[6]`;
- `phase[2]`: cosine/sine phase for phase-driven skills;
- `posture_target[1]`: 0 stand, 1 sit;
- `side[1]`: -1 left, +1 right, 0 not applicable.

No “reserved” or permanently zero fields are added. A new semantic field requires a new schema version or an experiment demonstrating non-zero training coverage.

### 6.3 Proposed flat tensor layout

The single-frame baseline is 71D:

```text
index   width  contents
0..3        3  gyro, trunk frame, rad/s
3..6        3  projected gravity, trunk frame
6..20      14  joint position minus home pose
20..34     14  joint velocity
34..48     14  previous raw policy action
48..54      6  behavior one-hot
54..57      3  twist
57..61      4  head pose
61..67      6  body pose
67..69      2  phase [cos(2*pi*p), sin(2*pi*p)]
69          1  posture target
70          1  side
```

This layout is a proposed baseline, not permission to edit existing tasks. The implementation patch must include:

- an authoritative Python layout definition;
- fixed-width tests for every boundary;
- legacy-v1 to typed-v2 command adapters used only by the teacher collector;
- an equivalent Rust layout test before runtime activation;
- ONNX metadata containing schema version, behavior order, and dimensions.

### 6.4 Per-behavior encoding

| Behavior | Active condition fields | Scheduler behavior |
|---|---|---|
| STAND | head/body pose as supported | hold; use for forced recovery |
| LOCOMOTION | twist, head/body pose | continuous |
| SIT_STAND | posture target, optional head pose | hold target; policy handles transition |
| GROUND_PICK | phase | scheduler advances phase and owns completion |
| KICK | side | scheduler owns one-shot window; phase initially zero |
| ROULADE | behavior id only | scheduler owns one-shot/busy window |

For kick and roulade, start with teacher-compatible mode selection rather than inventing a new phase input. Add phase only if measured rollout variance or transition failures justify it.

### 6.5 History is an experiment axis, not a v0 assumption

The first student should use the 71D single-frame input. The current observation already includes previous action, and the specialist teachers prove that single-frame inference is viable.

Only test fixed frame history after the semantic baseline exists:

- H1: one 71D frame;
- H4: four 48D proprio frames plus current condition;
- H8: only if H4 materially improves transitions or recovery.

A recurrent policy is out of scope for v0 because hidden-state reset, ONNX export, replay determinism, and runtime diagnosis are more complex than a fixed ring buffer.

## 7. Action and control-profile contract

### 7.1 Preserve raw action semantics

The student continues to output 14 raw policy actions in canonical servo order. `previous_action` is the previous raw output, not a scaled motor target.

### 7.2 Keep control profiles outside the policy in v0

Current skills use different action scales, gains, gain ratios, durations, and filtering assumptions. For parity, v0 keeps those profiles in the scheduler/runtime and selects them from the explicit behavior condition.

This is intentionally conservative:

```text
condition -> one policy raw action
condition -> deterministic control profile
```

A later experiment may train one common action scale and gain. Do not make that a prerequisite for proving that model unification works.

Every training/evaluation path must apply the same control profile that deployment will apply. A model is not a candidate if it only works with an undocumented simulator-side scale or filter.

The profile table is part of the v0 contract, not an implementation detail. For
each behavior it must name action scale, gains/gain ratios, filtering, target
slew limits, and one-shot duration. On a behavior switch, the runtime and the
training scheduler keep the previous filter state, slew the target from its
current value, and change gains only at the documented switch boundary. A
profile change must never reset `previous_action` or create an unbounded target
step. Transition tests record both the raw policy output and the post-profile
target so a continuous network cannot hide a discontinuous actuator handoff.

## 8. Model architecture

### 8.1 Dense MLP first

Start with the repository’s proven MLP family, for example `(512, 256, 128)` with ELU, and a 14D action head. One-hot conditions are concatenated at the input in the baseline.

Do not start with a Transformer, recurrent network, or mixture of experts. First determine whether the problem is semantic/data coverage or actual representational capacity.

### 8.2 Initialization from VelStand

A useful baseline is to expand a VelStand actor checkpoint:

- copy proprioception input weights 0..48;
- map legacy twist/head/body weights into the v2 command locations;
- initialize behavior/phase/posture/side columns to zero or small values;
- copy downstream actor layers;
- seed normalizer statistics for shared fields;
- initialize the new condition fields from balanced synthetic samples.

This gives stand/locomotion/recovery a stable starting point before distilling other skills. The checkpoint-conversion utility must validate dimensions and emit a manifest; it must not silently guess a layout.

### 8.3 When to consider internal experts

A soft mixture of experts is a fallback only after measurements show negative transfer that cannot be fixed by data balance or optimization. Evidence would include:

- each skill fits well in isolation but loses quality only when jointly trained;
- gradient or loss statistics conflict persistently by task;
- increasing dense capacity does not recover the regression;
- regressions cluster by behavior family.

If needed, keep one external ONNX and one condition contract while routing internally among a small number of soft experts. Hard routing is not recommended initially because output discontinuities at behavior changes are exactly what the generalist is intended to reduce.

## 9. Normalization

The student owns one observation normalizer, baked into its ONNX through the existing export path.

Requirements:

- collect statistics from a balanced behavior distribution;
- do not let long locomotion episodes dominate short one-shot skills;
- preserve physical scaling and finite-value guards;
- include normalizer state in run manifests;
- validate exported ONNX against the PyTorch actor on a fixed observation battery.

The v0 baseline may normalize the full 71D input. If one-hot or bounded condition fields prove sensitive to dataset composition, introduce a custom split normalizer that normalizes continuous proprioception/commands and leaves categorical fields unchanged. That is a measured follow-up, not an initial complexity.

This choice is frozen before teacher collection: M1 records the exact
normalization transform and field mask in the schema manifest, and M2 uses that
same transform for dataset statistics. A later split-normalizer experiment is a
new dataset/checkpoint lineage, never an in-place reinterpretation of v0 data.

## 10. Teacher distillation

### 10.1 Offline dataset schema

For each sample, store at least:

```text
student_obs_v2        float32 [71]
teacher_obs_v1        float32 [61]
teacher_action_raw    float32 [14]
behavior_id           uint8
teacher_id            string/index
episode_id             integer
step_id                integer
seed                   integer
success/termination    labels
task progress/contact  evaluation labels when available
```

The collector should also record a manifest with:

- source repository and commit;
- teacher checkpoint/run path;
- exported ONNX hash;
- environment task id;
- observation schemas;
- command ranges;
- DR toggles and ranges;
- collector version;
- sample counts by behavior and outcome.

Teacher replay has a second, required state contract. Alongside each sample (or
in a restorable episode-state record), capture the command-manager state needed
to reproduce the teacher label: command values and resampling timers, phase
origin/progress, previous raw action, observation-delay/history buffers, episode
time, and task latches such as one-shot busy/completion state. DAgger must restore
this state before querying a teacher; physical qpos/qvel alone is insufficient.
The collector must either restore a complete snapshot or explicitly advance the
same state machine and prove equivalence with golden vectors.

Use versioned, chunked local shards with a simple manifest. Do not commit datasets or checkpoints to Git.

### 10.2 Legacy command adapter

For every student condition, the collector must construct the exact legacy teacher input:

- locomotion maps twist/head/body normally;
- sit/stand maps posture target to the legacy posture flag;
- ground pick maps phase into the legacy twist phase encoding;
- kick and roulade use the legacy zero-padded command expected by their teacher.

This adapter is high-risk code and requires golden-vector tests. A wrong adapter can produce a dataset that looks numerically valid while asking the teacher to perform a different skill.

### 10.3 Balanced collection

Balance by behavior and by meaningful state bucket, not by wall-clock frames. Examples:

- locomotion: command buckets, zero command, turn in place, push recovery;
- recovery: face up, face down, side, crouch, near-success states;
- sit/stand: sitting hold, downward transition, rising transition, standing hold;
- ground pick: phase bins and return-to-stand tail;
- kick: side, pre-kick, contact, landing, post-kick stabilization;
- roulade: launch, inversion, landing, failed/perturbed recovery.

A million easy standing frames are not a substitute for rare dynamic frontier states.

### 10.4 Behavior cloning objective

The first objective is raw-action regression:

```text
L_bc = mean((student_action - teacher_action)^2)
```

Report per-behavior loss and per-joint loss. Aggregate loss alone can hide complete failure of a short dynamic skill.

Optional later terms:

- weight hard transition/frontier samples more heavily;
- action-distribution or latent distillation when using PyTorch teachers;
- symmetry augmentation for behavior/side combinations where the task is physically symmetric.

Do not add state-tracking or task rewards until action imitation is understood.

## 11. DAgger and student-state coverage

High offline fit with poor rollout performance indicates covariate shift, not necessarily insufficient model capacity.

The recommended second stage is DAgger-like collection:

1. run the student in the appropriate task scene;
2. build the corresponding teacher-v1 observation from the same state;
3. query the behavior’s teacher;
4. append teacher labels on states actually visited by the student;
5. retrain and repeat.

Use a teacher/student control-mixture schedule early enough to avoid collecting only catastrophic fallen states. Record whether the executed action came from teacher or student.

Teacher labels may be unreliable far outside a teacher’s training distribution. The collector should flag invalid/NaN states, impossible contacts, and extreme pose deviations, and evaluations should distinguish “teacher cannot recover here” from “student failed to imitate.”

## 12. Optional multi-task RL fine-tuning

Distillation plus DAgger may already meet the product objective. Multi-task PPO is required only when measured task outcomes remain below gates despite good action imitation and state coverage.

### 12.1 Superset generalist environment

For online joint training and in-episode transitions, the preferred v0 architecture is a single all-collision superset environment:

- one canonical robot;
- flat floor initially;
- a ball entity present but parked/disabled outside kick episodes;
- per-environment behavior state;
- a typed command term producing v2 conditions;
- behavior-masked rewards, resets, and terminations;
- shared BAM/DR/noise stack inherited from the proven velocity/VelStand base.

Before implementing this environment, M6 must pass a CPU proof slice with two
behavior IDs in the same batch. The slice must demonstrate per-env condition
storage, command/phase updates, reset selection, reward masking, termination
handling, and episode accounting without mutating shared manager config. If that
slice cannot be made correct with the existing managers, stop the superset path
and use separate collectors/tasks plus an explicit transition harness; do not
hide per-env routing in ad-hoc global term mutation.

Reuse existing MDP functions and configs. Do not copy reward implementations into a parallel “generalist” version unless their semantics actually differ.

### 12.2 Why not start with a custom multi-env runner

A runner that interleaves several heterogeneous `ManagerBasedRlEnv` instances can preserve current task configs, but it makes in-episode transitions difficult and adds optimizer/value batching complexity. It remains a fallback if the superset scene becomes unmaintainable, not the first implementation.

### 12.3 Reward and advantage isolation

A generalist must not optimize a meaningless sum of all task rewards. The active behavior selects the task objective; common safety/smoothness terms remain shared.

At minimum:

- mask task rewards by active behavior;
- log reward mass per behavior;
- normalize or inspect advantages per behavior;
- balance rollout samples so locomotion does not dominate short skills;
- consider a behavior-conditioned critic;
- add separate value heads only if value interference is measured.

### 12.4 Curriculum

Add skills incrementally and keep earlier regression batteries active. Do not introduce all reset distributions, task taxes, and dynamic skills at one iteration. The VelStand run history shows that curriculum timing can remove the exploration window a skill needs.

## 13. Transition training

A generalist is not complete if each behavior only works in a reset-isolated episode.

Train a transition graph explicitly:

```text
STAND <-> LOCOMOTION
STAND <-> SIT_STAND
STAND -> GROUND_PICK -> STAND
STAND -> KICK -> STAND
STAND -> ROULADE -> STAND/recovery
fall -> STAND/recovery -> LOCOMOTION
```

Transition episodes change the condition without resetting observation history or previous action. The scheduler still enforces legal and non-interruptible windows.

Measure:

- success of both source and destination behavior;
- fall rate in a fixed interval after the switch;
- peak action and target discontinuity;
- settling time;
- ability to resume locomotion after one-shot skills;
- behavior under repeated or stale requests.

Transition failures should first be treated as missing training distribution, not automatically as a reason to add a larger network.

## 14. Evaluation and acceptance criteria

All comparisons use fixed seeds and specialist baselines captured before generalist code changes. Report distributions or confidence intervals, not one showcase rollout.

The evaluation protocol fixes the battery seed list, episode count per
behavior/bucket, DR and reset configuration, success definition, and confidence
interval method in the M0 manifest. Relative gates are evaluated against the
same battery; if a specialist success rate is below 5%, use an absolute floor
and report the raw rate instead of a misleading relative ratio.

### 14.1 Per-skill metrics

- locomotion: velocity tracking, yaw tracking, fall rate, command buckets, turn-in-place;
- stand: tilt, height, joint deviation, jitter, power/torque proxies;
- recovery: success and time by spawn class;
- sit/stand: transition success, target hold, settling time, impact;
- ground pick: mouth contact/touch success, return-to-stand success, duration;
- kick: ball displacement/velocity, correct foot, landing stability;
- roulade: full rotation success, sagittal alignment, upright landing, repeat stability.

### 14.2 Initial go/no-go gates

These are starting thresholds and may be revised once baseline variance is measured:

- no included skill below 90% of its specialist success rate;
- stand/locomotion/sit metrics within 10% of specialist baseline;
- no more than 25% relative increase in fall rate from a low baseline;
- transition success at least 90% for common transitions;
- no systematic positive penalty term or new NaN path;
- exported ONNX agrees with PyTorch on the fixed observation battery;
- candidate meets the eventual board inference budget with comfortable margin inside the 20 ms tick;
- specialists remain available as fallback.

Kick mirroring is not assumed for collection. Before using mirrored samples,
run a symmetry validation on the actual all-collision model covering joint
permutation/signs, ball reset geometry, contact-foot selection, and measured
teacher outcomes. If any check fails, collect left and right episodes
independently and record the side-specific teacher in the manifest.

A weighted average cannot waive a failed individual skill.

## 15. Export and metadata

The export path remains `scripts/export.py` or a small extension of it so the observation normalizer stays inside the graph.

A generalist ONNX must carry metadata sufficient for fail-fast loading:

```text
policy_family          microduck-generalist
policy_contract        generalist-v0
obs_schema_version     2
obs_width              71
action_width           14
behavior_order         stand,locomotion,sit_stand,ground_pick,kick,roulade
joint_names            canonical 14-servo order
home_pose/model id     explicit identifier
training commit        git SHA
run/checkpoint         provenance
```

The future runtime must reject a model whose metadata disagrees with its compiled condition layout, even if the tensor shape happens to match.

## 16. Runtime integration boundary

Runtime changes belong in `pollen-robotics/microduck` after simulation gates pass. This repository should produce the contract and test vectors needed by that work.

Recommended rollout:

1. **shadow mode** — specialists actuate; generalist runs on real observations and logs action deltas/latency;
2. **canary mode** — configuration selects generalist, with specialist bundle still shipped;
3. **default mode** — only after repeated hardware trials and update rollback tests.

The scheduler and safety layer remain authoritative in every mode. “One model” is a packaging/control-learning choice, not an authority change.

## 17. Future motion-reference extension

Do not implement a motion encoder in v0. Preserve an architectural seam:

```text
skill condition -> condition encoder --+
                                      +--> behavior representation -> controller
future motion  -> optional encoder ----+
```

The motion path becomes justified only when product requirements include arbitrary animation clips, community motion packs, live imitation, or adding many behaviors without retraining the low-level controller. Reaching that decision should be driven by product scope, not by a desire to imitate a larger humanoid project.

## 18. Risks and mitigations

| Risk | Detection | First mitigation |
|---|---|---|
| locomotion dominates the dataset | per-behavior sample/reward reports | balanced buckets and batches |
| dynamic skills regress | per-skill rollout battery | incremental inclusion; DAgger frontier data |
| zero command is ambiguous | schema/golden tests | explicit behavior one-hot |
| BC fits offline but falls in rollout | high validation fit, poor rollout | DAgger/student-state labels |
| value-function interference | per-behavior value error/advantage stats | conditioned critic, then separate heads |
| one dense actor lacks capacity | isolated fit good, joint fit bad | increase capacity, then soft MoE |
| transition falls | switch battery | explicit in-episode transition curriculum |
| simulator-only control profile | sim/runtime replay mismatch | one documented profile table and parity tests |
| observation schema drift | Python/Rust golden vectors | metadata + fail-fast loaders |
| sim2real regression | shadow/canary hardware tests | specialists and release rollback remain |
| experiment sprawl | many uncomparable runs | staged gates and one-variable experiments |

## 19. Proposed repository changes by phase

This document does not authorize all of the following in one PR. Suggested sequence:

1. **schema foundation** — behavior dataclass/layout, adapters, golden tests; no task behavior change;
2. **baseline evaluator** — deterministic specialist reports;
3. **teacher collector** — manifests and local dataset shards;
4. **BC baseline** — VelStand initialization and offline student;
5. **DAgger** — student-state collector and retraining loop;
6. **generalist superset env** — only if task-level RL fine-tuning is needed;
7. **transition curriculum**;
8. **export metadata/test vectors**;
9. **separate runtime shadow/canary PRs**.

Each phase should be independently reviewable and should keep current task tests green.

## 20. Decisions requested from experiments

The experiment program must answer, with measurements:

1. Is single-frame 71D input enough, or is H4 history materially better?
2. Can a dense `(512, 256, 128)` student fit and roll out G0/G1?
3. Does VelStand checkpoint expansion preserve locomotion better than random initialization?
4. Is offline BC sufficient for any skill family?
5. How much DAgger data is required per dynamic frontier?
6. Can one shared normalizer remain stable under balanced multi-skill data?
7. Are behavior-masked rewards enough, or does the critic need separate heads?
8. At which skill, if any, does measured negative transfer justify soft MoE?
9. Can mode-dependent control profiles be retained without transition discontinuity?
10. Does the final candidate preserve real-robot timing and sim2real behavior?

Those answers, rather than this proposal alone, determine the final implementation.
