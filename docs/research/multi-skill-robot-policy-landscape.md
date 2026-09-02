# Multi-Skill Robot Policy Landscape

Research date: 2026-09-05

Scope: community approaches to combining several robot skills in one deployable
policy, with emphasis on parameter sharing, capacity, conditioning, routing,
distillation, and transitions. This is a directional review for Microduck's
merged-policy design; it is not an exhaustive literature survey.

## Executive Summary

The community does not generally treat “five skills in one network with the
same size as one skill” as a safe default. The recurring choices are: (1) a
shared representation plus task/skill conditioning, (2) mixture-of-experts or
task-specific adapters, or (3) a high-level controller that selects modular
low-level skills. Large generalist systems increase capacity and data scale;
they do not rely on a single unchanged specialist MLP.

For legged robots, modularity is especially common when skills have different
contact modes, time scales, or failure semantics. A single policy is usually
trained only after the task/state interface and transition distribution are
made explicit. This aligns with Microduck's evidence: the shared 71D actor had
stand/locomotion interference, while the validated specialists remain stable.

## Findings

### 1. Large-scale multi-task systems share data and representations, but scale the system

MT-Opt studies continuous multi-task robotic RL at scale and describes a
system that learns a repertoire while sharing exploration, experience, and
representations across tasks. The paper's premise is large-scale data and
system infrastructure, not a same-capacity replacement for every specialist.
[MT-Opt, arXiv:2104.08212](https://arxiv.org/abs/2104.08212) (verified for the
paper's stated scope; architectural details should be read from the full
paper before implementation).

BC-Z similarly trains a broad behavior-conditioned policy from demonstrations.
Its central design is an explicit task/goal variable plus a large diverse
dataset, which supports the conclusion that conditioning and coverage are
first-class requirements rather than optional input bits.
[BC-Z, arXiv:2202.02013](https://arxiv.org/abs/2202.02013) (supported; this
review uses the paper's published abstract and project description).

### 2. Generalist policies use adapters, experts, or larger backbones

Open X-Embodiment/Octo presents a transformer policy pretrained over diverse
robot datasets and action spaces. The relevant pattern is a shared backbone
with flexible conditioning and substantial pretraining scale, rather than
copying a small single-task controller unchanged.
[Octo, arXiv:2405.12213](https://arxiv.org/abs/2405.12213) (verified for the
generalist-policy scope; exact capacity comparisons are not implied here).

In the broader multi-task RL literature, task-conditioned policies and
mixture-of-experts are used to reduce negative transfer: common features are
shared, while task-specific computation is selected or adapted. This is a
supported design pattern, not a universal winner; expert routing adds runtime
and training complexity and needs balanced task sampling.

### 3. Legged-robot work often keeps a modular controller for incompatible phases

Recent legged-control work explicitly calls out compounding errors and task
transition difficulty, then combines multi-task pretraining with online RL
fine-tuning for robust transitions. DMLoco is one example of this direction
for quadrupeds, with a large multi-task pretraining stage followed by PPO
fine-tuning and onboard deployment constraints.
[DMLoco, arXiv:2507.05674](https://arxiv.org/abs/2507.05674) (tentative for
specific implementation claims; verified for the abstract's stated pipeline).

A recent bipedal soccer-robot paper uses a posture-driven state machine and
separate ball-seeking/kicking and fall-recovery networks. Its stated rationale
is to prevent state interference between upright walking/kicking and recovery,
which is directly analogous to Microduck's observed stand/locomotion/recovery
interference.
[Adaptive multi-task control for bipedal soccer robots,
arXiv:2604.19104](https://arxiv.org/abs/2604.19104) (tentative: recent
preprint, not independently reproduced here).

### 4. Distillation is useful, but pointwise action matching is insufficient

Teacher-student distillation is widely used to compress or unify policies, but
robot control papers and benchmarks repeatedly expose compounding closed-loop
errors: low action MSE on teacher states does not ensure stability on student
states. DAgger-style relabeling, state distribution coverage, and explicit
transition data are therefore standard remedies. This is strongly consistent
with Microduck's BC/DAgger results, but the cited papers do not establish a
single recipe that guarantees success for this robot.

## Design Choices Seen In Practice

| Choice | Why teams use it | Cost / failure mode | Fit for Microduck |
|---|---|---|---|
| Shared trunk + behavior embedding/FiLM | Shares balance/body features while retaining task adaptation | Needs more capacity and balanced data | Best first true merged-policy experiment |
| Mixture-of-experts / routed experts | Limits negative transfer; each expert specializes | Router errors, more parameters, transition discontinuities | Strong candidate if shared trunk still interferes |
| Separate low-level skills + scheduler | Robust for contact/mechanics changes and recovery | Not one neural policy; scheduler remains central | Current validated product baseline |
| One unchanged specialist-sized MLP | Simple deployment and low latency | Capacity bottleneck and negative transfer | Not supported by our evidence |
| Distillation into one dense actor | Simple artifact and runtime | Closed-loop drift, transition coverage gaps | Useful only with DAgger/transition replay |
| Large transformer/generalist | Handles broad datasets and conditioning | Data, compute, latency, calibration burden | Overkill for current closed skill set |

## Implications For Microduck

1. The next true merged-policy experiment should increase capacity relative to
   one specialist and keep an explicit condition interface. A reasonable first
   ablation is 2x-4x specialist parameter count, shared proprioception trunk,
   behavior embedding, and small per-behavior adapters.
2. Train G0 first (stand, locomotion, recovery). Do not add sit, pick, kick, or
   roulade until G0 passes closed-loop per-behavior and transition gates.
3. Collect trajectories after the student enters off-manifold states. Balance
   behavior, reset bucket, and transition frontier; retain previous action,
   delay/history, timers, and latches in replay state.
4. Compare three controls under one battery: specialist scheduler, true shared
   conditioned actor, and expert/adaptor actor. Do not compare only aggregate
   reward or offline MSE.
5. Treat the six-expert ONNX created during this investigation as an upper-bound
   routing control, not as evidence that a shared merged policy has succeeded.

## Contradictions And Uncertainty

There is no single community consensus that all skills should share one actor.
Manipulation generalists benefit from massive shared datasets and scalable
backbones; legged recovery and contact-mode changes often favor modularity.
These are different task distributions and hardware constraints, so neither
camp directly proves the right architecture for Microduck.

The cited abstracts do not provide a controlled parameter-count comparison of
“one specialist-sized MLP” versus “five skills.” That specific question needs a
Microduck capacity sweep. The strongest current claim is architectural: equal
capacity is a weak prior, not a community norm.

## Method

Subquestions: capacity scaling, conditioning, expert/adaptor routing,
distillation/DAgger, and legged transition handling. Primary sources were
original papers and their arXiv records. Searches used arXiv metadata and
abstracts on 2026-09-05; browser automation was unavailable, so claims that
depend on full-paper implementation details are marked tentative or supported.
