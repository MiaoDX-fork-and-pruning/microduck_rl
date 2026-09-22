# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
The user authorized sustained necessary changes, training and checks through
`intuitive-flow`. Latest user requested status; the execution objective remains
active. Latest slice: teacher allocation repair and an inconclusive behavioral
comparison. The boundary-controlled replay and a 10-seed native yaw
robustness diagnostic are complete; no training process is currently running.

## Current evidence

Two bounded seed-17 6,000→6,500 campaigns have completed. Neither is accepted.
Scores are normalized capability scores, not success rates; each bucket must
reach .80. Both CoM axes remain at ±3 mm.

| Retained policy | Native stage minimum | Native final held-out minimum | Corrected CPU minimum |
| --- | ---: | ---: | ---: |
| Original stage-gate control | .248 yaw | .347 yaw | .000 yaw (5/6 pass) |
| Teacher-repair replay | .589 yaw | .748 yaw | .343 turn-right (3/6 pass) |

The replay's 6500 candidate failed right-turn preservation and was rolled back.
Its final actor and normalizer exactly equal the 6250 snapshot (25 tensors),
while consumed budget remains 6500. The rejected candidate's stage yaw .363
must not be presented as the retained policy's stage score. Retained native
held-out scores: zero .909, forward .899, lateral .804, yaw .748, left .799,
right .782. CPU: zero .970, forward .879, lateral .783, yaw .705, left .829,
right .343. The protected set contains zero and turn-right.

Authoritative summaries, exact report/checkpoint paths and hashes:

- Control: `/tmp/microduck-adaptive-stage-gate-s17-6500-beea4f8/experiment-summary.json`.
- Replay: `/tmp/microduck-adaptive-deficit-s17-6500-96fd557/experiment-summary.json`.
  Its `retained-policy-proof.json` verifies the rollback tensor equality.

## Controller repair and experiment limit

`96fd557` interrupts configured frontier dwell when another directional bucket
trails the current focus by more than .25 normalized capability. It preserves
.80 mastery, quarter-step probability updates, directional floors, zero/nominal
anchors and existing rollback. The old saved .784 lateral / .248 yaw state
incorrectly increased lateral probability and reduced yaw; regression tests
now redirect the bounded focus slice to yaw, including actual command sampling
and exact save/resume decisions.

The replay **did not exercise that intervention during training**. At 6250,
lateral .787 versus yaw .589 was below the .25 urgency margin; at terminal
6500 a mastered right-turn failure invoked the existing retention repair and
rollback. Last-window real samples remained .253 lateral versus .083 yaw.
GPU training diverged before the first gate despite matching seed/checkpoint,
so score differences cannot be credited to the repair. See
`first-gate-comparison.json` and `pre-intervention-comparison.json` in the replay.

A 6500→7000 extension was briefly launched on the mistaken premise that the
retained model had the rejected candidate's .363 yaw score. It was stopped
before completing the budget after verifying the rollback. Its terminal status
is `stopped_invalid_experiment_premise` at
`/tmp/microduck-adaptive-deficit-s17-6500-7000-96fd557/execution-status.json`.
Do not restart it or treat it as behavioral evidence.

## Remaining blocker and next proof

Blocker fingerprint: `direct_command_yaw_under_reset_dr_robustness`.
The strict boundary treatment and its matched no-injection control both
completed the same 6250→6500 budget. The treatment changed only the teacher
decision from lateral focus to yaw focus, increased yaw sampling from about
8.2% to 12.5%, and still degraded native held-out yaw to `.129` (corrected CPU
to `0`). The control retained native held-out lower-tail `.528` and corrected
CPU lower-tail `.458`; neither is accepted. These are rejection results, not
retained-policy scores.

The exact 6250 actor was then evaluated natively over ten reset/DR seeds at the
stage distribution (CoM and head-CoM ±3 mm):
`/tmp/microduck-yaw-robustness-6250-96fd557/capability-stage.json`. The
conservative aggregate lower tail is `0.000`. Pure yaw scores by seed were
`0, .766, .035, 0, .038, 0, 0, 0, 0, 0`; no member reached `.80`, while all
yaw rollouts survived without falling. The manifest records the same DR ranges
and reset sources for every member, with consumed-state hashes, so this proves
reset/DR sensitivity but does not yet identify one scalar friction, armature or
CoM realization as the cause.

Do not repeat direct-yaw exposure or turn-proxy exposure as if either repaired
the policy. The next bounded intervention must target reset-conditioned pure-yaw
acquisition and should record the realized reset/DR state needed to distinguish
an initialization sensitivity from a missing yaw behavior. Keep the smoke64/5
contract, native-vs-CPU separation, and exact retained-policy rollback rules.

## Proven implementation and portability

- 269 relevant tests pass; one existing specialist-manifest test is excluded
  because `artifacts/specialist_artifact_manifest.json` is absent. Focused Ruff
  (`E4,E7,E9,F`) and diff checks pass. Full-suite/full-Ruff cleanliness is not claimed.
- New-source real smoke64/5 passed with a three-member stage gate (±3 mm), final
  held-out distribution (body ±15/head ±10 mm), checkpointed sensor reset,
  normalized ONNX and corrected CPU frame. All six CPU reset-noise arrays match
  consumed seeds: `/tmp/microduck-adaptive-deficit-smoke-96fd557/runtime-proof.json`.
- Frozen source `/tmp/microduck-adaptive-deficit-source-96fd557/` has 679 verified
  files. Its manifest explicitly records the pre-existing uncommitted CPU seed
  overlay; label `96fd557-cpu-seed-overlay-5ae829076760`. Preserve that worktree
  patch and do not describe the overlay as the plain committed tree.
- `931e3c9` corrects CPU velocity measurement from principal-inertia/CoM
  `mjOBJ_BODY` to link-frame/origin `mjOBJ_XBODY`. Identical-ONNX replay changed
  measurements only. Historical old-frame CPU and actuator A/B tracking scores
  are superseded; proof: `/tmp/microduck-adaptive-cpu-link-frame-s17-6000/`.
- A fresh non-editable locked install, isolated imports, config loading and
  packaged 14-actuator model compilation pass on Linux x86_64. Explicit
  `--default-index https://pypi.org/simple` avoids local mirror/lock mismatch.
  Proof: `/tmp/microduck-adaptive-fresh-sync-proof-96fd557.json`.
  Fresh-environment GPU training and remote execution remain unproven.

## Acceptance and boundaries

Required: retained six-bucket native mastery, fresh final-distribution held-out,
rollout/video inspection, normalized ONNX/CPU rehearsal, the same automated
procedure across training seeds 17/23/47, then a matched-budget fixed baseline.
No hardware or adaptive-superiority claim exists. There is no external blocker
preventing the next diagnostic, and the goal is not complete.

Preserve canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward signs,
.80 thresholds and exact-zero/nominal anchors. IsaacLab, uv.lock, generated files
and unrelated processes remain outside this task. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Do not repeat unchanged
sensor-reset, slew, rejected yaw-planar or non-intervening teacher budgets.
Runs use TensorBoard and the validated local venv under `/tmp`.
