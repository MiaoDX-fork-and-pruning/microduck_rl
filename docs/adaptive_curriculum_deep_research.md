# Adaptive Curriculum Deep Research

Status: preserved research record
Date: 2026-09-14

## Question

Is the current fixed-iteration curriculum the best practice, and should the
Microduck walking policy replace it with a more autonomous curriculum?

## Executive conclusion

The MJLab schedule is a valid and reproducible canonical baseline, but not a
universal optimum. The current recipe was produced by repeated experiments,
changes, and rollbacks; it should not be treated as a timeless theoretical
schedule. Adaptive curriculum is useful when it controls a difficulty or
domain-randomization axis from measured capability. It is not a reason to let
all reward weights, commands, terminations, and physics parameters change at
once.

The strongest project-level recommendation is a two-track design:

1. Keep the existing fixed MJLab recipe unchanged as the parity and regression
   baseline.
2. Add a separate MJLab adaptive experiment that preserves the same final task
   semantics, but advances one difficulty axis at a time using a frozen command
   battery, hysteresis, minimum dwell time, checkpoint preservation, and
   rollback.

This should be tested in MJLab first. IsaacLab is not a prerequisite for this
experiment and should not be used to explain away a failure in the canonical
MuJoCo pipeline.

## Evidence ledger

### Fixed or geometric curriculum

- Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*:
  https://arxiv.org/abs/1901.08652
  - Uses a fixed geometric ramp for motion penalties and disturbances while
    leaving the task objective active.
  - Demonstrates that objective-first regularization is a proven locomotion
    pattern, not evidence that fixed schedules are always optimal.
  - Confidence: verified mechanism; supported relevance to Microduck.

- RMA, *Rapid Motor Adaptation for Legged Robots*:
  https://arxiv.org/abs/2107.04034
  - Uses staged training and gradual difficulty changes, alongside an online
    adaptation module for deployment-time dynamics variation.
  - Confidence: verified mechanism; adaptive module would require a new policy
    ABI and is not a drop-in schedule replacement.

### Capability-gated locomotion curriculum

- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep
  Reinforcement Learning*:
  https://arxiv.org/abs/2109.11978
  - Each environment has a terrain level. Crossing a distance threshold
    promotes it; insufficient displacement demotes it; maximum levels are
    looped to preserve diversity.
  - This is the closest mature precedent for Microduck: difficulty follows
    measured capability rather than wall-clock iteration.
  - The paper also removes curriculum for a parallelism ablation because total
    reward is not comparable when stronger policies are automatically exposed
    to harder levels.
  - Confidence: verified mechanism; command-conditioned adaptation for flat
    biped walking remains a project-specific design.

- IsaacLab official `terrain_levels_vel` implementation:
  https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py
  - Uses commanded planar distance to move environments up or down terrain
    levels.
  - It does not solve Microduck yaw or zero-command evaluation: those need
    separate metrics.
  - Confidence: verified implementation; not a direct drop-in for this task.

### Adaptive domain randomization

- Akkaya et al., *Solving Rubik's Cube with a Robot Hand* (ADR):
  https://arxiv.org/abs/1910.07113
  - Samples parameter boundaries, expands them above a high performance
    threshold, and contracts them below a low threshold.
  - Parameters are updated independently with bounded step sizes and a
    performance buffer.
  - Appropriate for CoM, friction, delay, and voltage uncertainty ranges; not
    a justification for changing reward semantics.
  - Confidence: verified mechanism; project effectiveness is tentative.

- Mehta et al., *Active Domain Randomization*:
  https://arxiv.org/abs/1904.04762
  https://proceedings.mlr.press/v100/mehta20a.html
  - Learns to sample informative regions of a randomization space using rollout
    discrepancies rather than uniform sampling.
  - Warns that uniform DR can be high-variance and that active sampling can
    expose catastrophic regions; hard bounds and safety checks remain required.
  - Confidence: verified mechanism; too complex for the first Microduck trial.

- Muratore et al., *DORAEMON: Domain Randomization via Entropy Maximization*:
  https://arxiv.org/abs/2311.01885
  - Widens the distribution while enforcing a success constraint and a KL
    trust region; backtracks when the constraint is violated.
  - This is a useful design precedent for small-step expansion plus rollback.
  - Confidence: verified preprint mechanism; cross-task generality tentative.

- Chebotar et al., *SimOpt: Closing the Sim-to-Real Loop*:
  https://arxiv.org/abs/1810.05687
  - Uses real rollouts to update simulation parameter distributions while
    constraining distribution movement with KL divergence.
  - Best fit for BAM calibration after safe real-robot traces exist; not needed
    for the first simulation-only curriculum experiment.
  - Confidence: verified mechanism; requires real data and safety procedure.

### Learning-progress and level replay methods

- Portelas et al., *Teacher algorithms for curriculum learning of Deep RL in
  continuously parameterized environments* (ALP-GMM):
  https://arxiv.org/abs/1910.07224
  - Samples parameter regions with high absolute learning progress and keeps a
    random exploration fraction.
  - Useful for a later high-dimensional command/DR sampler, but adds history,
    GMM fitting, and a second source of non-stationarity.
  - Confidence: verified mechanism; not first choice for the current failure.

- Jiang et al., *Prioritized Level Replay*:
  https://arxiv.org/abs/2010.03934
  - Uses TD/GAE-based learning potential and staleness to replay useful levels.
  - Requires identifiable levels and replay bookkeeping. High TD error in a
    physical task can also reflect contact noise or failure, so it should not
    be the only safety or advancement signal.
  - Confidence: verified mechanism; low priority for current flat walking.

- Portelas et al., *Self-Paced Deep Reinforcement Learning*:
  https://proceedings.neurips.cc/paper_files/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html
  - Keeps a learned context distribution close to a target distribution using
    KL regularization.
  - Attractive when the final MJLab distribution is immutable, but heavier than
    the proposed gate controller.
  - Confidence: verified mechanism; not first implementation.

### Recent actuator-specific precedents

- *Actuator Dynamics Curricula for Narrow-Viability Tasks in Legged Robot
  Learning*: https://arxiv.org/abs/2609.09492
  - A 2026 preprint anneals actuator stiffness from an easy value toward an
    identified value using an EMA of episode length.
  - Relevant as a hypothesis for BAM authority, but it is a single handstand
    task and must not be copied directly to voltage-controlled XL330 actuators.
  - Confidence: verified preprint claim; cross-task relevance tentative.

- BAM documentation:
  https://bam.readthedocs.io/en/latest/usage/mjlab_gpu.html
  https://bam.readthedocs.io/en/latest/theory/models.html
  - Documents calibrated actuator models and bounded per-environment DR, not a
    model-specific RL schedule.
  - M4/M6 should therefore be selected from actuator trace/system-ID evidence,
    not from an assumed independent curriculum.

## Project-specific risks

- The current clean IsaacLab run entered a low-motion basin while multiple
  stages changed. This is evidence for schedule isolation, not proof that one
  term is solely responsible.
- Aggregate reward is unsafe as an advancement signal. It can improve while
  non-zero command response collapses.
- Adaptive sampling can forget nominal behavior. Keep a final-distribution
  anchor mixture and run a fixed held-out battery.
- Changing reward weights changes the optimization objective and the data
  distribution simultaneously. It should not be the first adaptive axis.
- `algorithm.schedule="adaptive"` in RSL-RL refers to PPO learning-rate/KL
  behavior, not environment curriculum.

## Research decision

Use a conservative capability-gated controller first. Reserve ALP-GMM, PLR,
SPDL, DORAEMON-style entropy optimization, and runtime RMA adaptation for later
experiments after the basic MJLab walking response is stable.

## Architecture Research: Where Adaptive Logic Usually Lives

### Question

Is an external train/evaluate/advance shell pipeline a general best practice for
adaptive RL training, or should the controller be part of the trainer?

### Executive finding

The common pattern is a **trainer-owned control loop with a separate evaluator
interface**. The environment or curriculum manager owns cheap per-rollout
updates; the trainer owns evaluation cadence, checkpointing, and the decision to
change training state. When evaluation is expensive or distributed, the
evaluator may run as a separate worker or process, but it reports back through a
typed callback/API. A shell pipeline is a useful experiment harness and batch
workflow, but it is not the usual source of truth for optimizer state,
curriculum state, or rollback.

This is a supported architectural conclusion, not a claim that every RL
implementation uses one framework API. The exact split varies with evaluation
cost and parallelism.

### Evidence ledger

| Claim | Source | Evidence | Confidence |
| --- | --- | --- | --- |
| Environment-local curricula are normally executed by an environment manager during training. | [Isaac Lab `CurriculumManager`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/managers/curriculum_manager.py) | The manager parses curriculum terms and calls each term through `compute()`; the official velocity curriculum updates terrain levels from commanded-distance outcomes. | Verified |
| A mature legged-RL precedent changes difficulty from measured capability inside the parallel environment loop. | [Rudin et al., Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978); [Isaac Lab `terrain_levels_vel`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py) | Terrain level is promoted or demoted from distance walked relative to the command, rather than from an external shell process. | Verified |
| Trainer callbacks are a standard place for evaluation, checkpoint, and model-state manipulation. | [Stable-Baselines3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) | Official docs define callbacks as hooks into training stages and document `EvalCallback`, checkpointing, and post-evaluation callbacks. | Verified |
| Distributed RL trainers expose explicit evaluation and training-result hooks. | [Ray RLlib callbacks](https://docs.ray.io/en/latest/rllib/rllib-callback.html); [RLlib API index](https://docs.ray.io/en/latest/rllib/rllib-curriculum.html) | RLlib exposes `on_train_result`, `on_evaluate_start`, `on_evaluate_end`, and checkpoint-loaded hooks, alongside separate evaluation runners. | Verified |
| The standard RSL-RL runner has a single synchronous PPO loop and checkpoint save/load, but no generic evaluation callback. | [RSL-RL `OnPolicyRunner`](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/runners/on_policy_runner.py) | `learn()` performs rollout, update, logging, and periodic save; the public runner exposes save/load and policy export, but no evaluator or curriculum callback contract. | Verified |
| Expensive adaptive domain-randomization methods still keep a policy-performance feedback loop, even when the teacher is a separate component. | [ADR](https://arxiv.org/abs/1910.07113); [Active Domain Randomization](https://arxiv.org/abs/1904.04762); [DORAEMON](https://arxiv.org/abs/2311.01885) | These methods adapt parameter distributions from measured policy performance, learning progress, or constrained success; the papers do not make a shell script the canonical state owner. | Supported |
| Fixed geometric schedules remain common in locomotion because they are reproducible and easy to compare. | [Hwangbo et al.](https://arxiv.org/abs/1901.08652); [Rudin et al.](https://arxiv.org/abs/2109.11978) | Representative locomotion systems use staged or geometric difficulty changes, while adaptive variants are introduced for particular level or DR problems. | Supported |

### Pattern comparison

| Pattern | Typical use | Strength | Main risk | Fit for Microduck |
| --- | --- | --- | --- | --- |
| Environment/manager curriculum | Cheap metrics available every reset or episode; terrain, command, or spawn levels | Fully synchronized with vectorized environments; low overhead | Can only use local signals; easy to mix training and evaluation signals | Good for simple per-environment level updates |
| Trainer callback plus evaluator | Evaluation every N PPO iterations; thresholds, promotion, rollback | Owns PPO/checkpoint state and gives deterministic decision points | Synchronous evaluation can pause training | Best default for the six-bucket capability gate |
| Asynchronous evaluator/teacher | Large policies, expensive simulators, population or level replay | Keeps accelerator busy and scales evaluation | Stale metrics and race conditions require versioning | Useful later if battery cost becomes material |
| External shell/workflow | Batch sweeps, cluster jobs, multi-stage reproducible pipelines | Easy to inspect, retry, and run across machines | Checkpoint discovery, state files, and resume semantics become fragile | Good harness and cluster wrapper; poor canonical state owner |

### Implication for this repository

The current external pipeline is a valid prototype and experiment harness. It
has already demonstrated that the battery and gate can be resumed across jobs.
It should not remain the final adaptive implementation because the current
`rsl_rl` runner has no evaluator callback, and the shell script currently owns
checkpoint selection, stage-file writes, and the transition boundary.

The appropriate target is a small adaptive runner extension:

1. Run PPO rollout and update as today.
2. At a configured iteration interval, snapshot policy, optimizer, RNG, and
   curriculum state.
3. Call a `FrozenCapabilityEvaluator` through an in-process or subprocess API.
4. Feed typed continuous metrics to `CapabilityGate`.
5. Apply one stage transition through live MJLab managers, then checkpoint the
   new state.
6. On preservation failure, restore the last known-good checkpoint in the
   runner and record the rollback.

The evaluator should remain independently runnable for audit and cluster use;
the runner should remain the authoritative owner of state. ONNX export should
stay at the final deployment-validation boundary, not in every internal
evaluation window unless the deployment graph itself is what is being tested.

### Limits and unresolved questions

- The surveyed framework documentation establishes common extension points, not
  a universal standard for adaptive curricula.
- RSL-RL's exact installed version in this repository may differ from the
  upstream API; the local runner must be patched against the installed version
  and tested with the repository's smoke contract.
- The present battery's binary scores saturate at `1.0`; continuous tracking,
  angular, survival, and lower-tail metrics must be fixed before an internal
  controller can make useful fine-grained decisions.
- Whether evaluation should share the training GPU or use a separate process is
  an operational choice. It should not change the ownership rule: one versioned
  runner state must define the next training stage.

### Method note

Research date: 2026-09-15. The review covered primary papers for adaptive
curriculum and domain randomization, official Isaac Lab, RLlib, Stable-Baselines3,
and RSL-RL documentation/source, plus the repository's installed architecture.
The search was bounded to trainer/environment architecture and did not attempt
to rank all curriculum algorithms or prove superiority for any one framework.
