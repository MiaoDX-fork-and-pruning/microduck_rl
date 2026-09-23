# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: explicitly resume sustained changes, training and checks through
`intuitive-flow` after API interruption. The stale host `blocked` label is not
an external blocker. Project-status writer unassigned; no shared-status edits.

## Current verdict

Adaptive acquisition, consolidation, retention and full trainer rollback work.
All-six native mastery and an autonomous usable-policy recipe remain unproven.
Both CoM axes remain stage 0 ±.003. Scores are capability, not success rates.

The strongest retained seed23 arm used 50% calibration resets from initialization,
then automatically consolidated with entropy 0 from update 1000 to 1250.

| Early-reset candidate | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage cohort | .96942 | .81523 | .78533 | .77699 | .73985 | .81754 |
| Final native diagnostic | .96216 | .83673 | .79039 | .85497 | .74788 | .80060 |
| CPU diagnostic | .97959 | .85440 | .78006 | .45563 | .72832 | .86311 |

Its own stage minimum improved .6617→.7399 (+.0781), passing retention.
Against the original retained seed23 actor the native-minimum gain is only
+.04072 and CPU-minimum loss is −.32062. CPU yaw mean is 1.109321 rad/s over
six seconds; 1.161378 is the last-five-second mean. Neither verdict passes .80.

Blocker: `native_mastery_and_product_rehearsal_contract`. Foot/plane contact
generation explains the fully paired native/CPU regression below; the original
XML-PD product rehearsal is still unvalidated. No external blocker.

## Completed proof and rejected routes

- Early reset: `/tmp/microduck-early-sensor-reset-s23-v2/verification.json`.
- Original seed23 consolidation rejected and restored update-1000 trainer:
  `/tmp/microduck-adaptive-from-scratch-s23-v4/verification.json`.
- Matched late zero/half/full calibration reset all rejected; half/full gains
  +.04555/+.04457 miss the .05 retention requirement:
  `/tmp/microduck-seed23-sensor-reset-comparison-v4/verification.json`.
  Half candidate passes six CPU buckets on a reused diagnostic seed but native
  lateral is .7998718476 and stage retention fails. It is not accepted.
- Continued entropy-.01 acquisition and command-magnitude rehearsal are negative:
  `/tmp/microduck-seed23-acquisition-comparison-v1/verification.json`,
  `/tmp/microduck-seed23-command-rehearsal-v1/verification.json`.
- Uniform and hip-yaw action scale scans have no all-six passing window.
  Control-boundary sensor refresh worsens yaw .45563→.44376. Independent audit:
  `/tmp/microduck-cpu-transfer-causal-s23-v1/verification.json`, 11 reports/66
  traces. Actual sent targets, reset pairing, score recomputation, every ONNX
  action and source hashes pass; both old scans replay exactly. Original scan
  `applied_action`/metadata limitations are corrected here, retrospectively.
- CPU BAM M6 gives .98257/.73820/.86140/.57954/.65825/.68747; stiff friction
  gives .98040/.74205/.86900/.56243/.62216/.65884. Yaw improves but other
  buckets regress; both fail the joint intervention criterion.
  `/tmp/microduck-cpu-bam-response-s23-v1/verification.json`, 3 reports/18
  traces, independently rederived scores/actions/targets and exact XML replay.

Do not extend these unchanged recipes or scan more runtime scales. Earlier
seed17 references and negative reward/smoothing experiments are in the plan.
Sampled half-reset video frames show upright stepping, but are neither full
video review nor hardware proof.

## Paired contact diagnosis — verified

Complete capture: `/tmp/microduck-complete-native-s23-v2`. It records the compiled
model, all 13 expanded fields, reset/calibration, prior motor torque and every
5 ms target/lag/force/state. Reset already inserts one zero target into delay
history; first-policy-target clamping was an incorrect earlier approximation.

| Current unbiased actor view; paired BAM/DR/reset/delay | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | .96742 | .84734 | .82243 | .81978 | .73985 | .76863 |
| CPU original contacts | .95017 | .82833 | .74120 | .31496 | .74189 | .69148 |
| CPU native foot/plane contacts | .96718 | .84813 | .82006 | .81439 | .71768 | .76612 |

At first impact (substep 8, 40 ms), CPU has 2 contacts and Warp 8. At substep
638 they have 1 and 4. Warp replays all five selected native states exactly.
CPU 100 versus 10 solver iterations is identical. Replacing only contacts
reduces three contact-state maximum qvel errors .38745/.21135/.66790 to
.000102/3.12e-7/4.28e-7. This identifies contact generation as a causal difference.

Fresh contacts computed from the **live CPU pose** also close the six closed-loop
score gaps to ≤.02217; yaw mean .73741 versus native .73938 rad/s. The declared
≤.03 score / ≤.05 rad/s agreement gate passes. Both sides still fail mastery;
this diagnostic does not validate XML-PD deployment or hardware.

Audits: `/tmp/microduck-complete-cpu-s23-v1/verification.json` (12 traces and
7200 independently replayed CPU substeps), and
`/tmp/microduck-contact-closedloop-s23-v2/verification.json` (fresh contact
geometry, six scores, every ONNX action and actual delayed target). All 682
frozen source hashes pass. No training/evaluation sessions remain running.

The partial physical replay's `all_delivered_targets_verified` claim is corrected
to requested-target-only; it recorded targets before encoder-bias subtraction.
Both `/tmp/microduck-cpu-external-native-obs-s23-v{1,2}` have `INVALIDATION.json`:
open-loop replay, wrong timestep/BAM cadence and reset/termination defects exclude
their falls from causal evidence. Raw artifacts remain available.

## Next bounded work

Make the CPU rehearsal's contact semantics explicit and preserve separate
original-runtime and native-matched diagnostic verdicts. Do not silently replace
the product gate with the passing agreement result. Return acquisition work to
native moving-turn mastery: steady 1–5 s yaw means are +.708/−.729 for ±.8
commands in this paired diagnostic, with further post-push degradation.
Measure the unmodified actor's turn rollouts before choosing another training
intervention. Do not repeat runtime scale scans or unchanged training extensions.

## Native turn and visual profile — verified

The unmodified checkpoint was replayed from the original native traces and
captured as full 300-frame videos (no resimulation):
`/tmp/microduck-native-turn-profile-s23-v1/verification.json` and
`visuals.json`. The videos show an upright, foot-supported biped throughout;
the failure is tracking quality, not a fall or an obviously parked pose.

| Native diagnostic seed / bucket | 0–1 s yaw | 1–5 s yaw | 5–6 s yaw | Score |
| --- | ---: | ---: | ---: | ---: |
| 20260918, yaw +.8 | +.718 | +.809 | +.893 | .855 |
| 20260919, turn-left +.8 | +.627 | +.700 | +.768 | .748 |
| 20260920, turn-right −.8 | −.819 | −.815 | −.903 | .801 |

There is no hip-yaw limit occupancy in these turn buckets and the largest
realized control is below the 1.75 A torque boundary. The lateral bucket is
different: left/right hip-yaw occupy the hard-limit proximity band for 26%/37%
of steps and the recorded control reaches the current boundary. The existing
`joint_pos_limit_proximity` function is not wired into the canonical velocity
reward stack, so this is a candidate lateral intervention, not an applied fix.

A direction-paired diagnostic reuses both turn reset/calibration seeds and flips
only the yaw sign. Positive-yaw moving turns undertrack in both paired seeds:
the original +.8 case averages +.700 rad/s over 1–5 s, while the reversed −.8
case averages −.726; the original −.8 case averages −.815, while reversed +.8
averages +.718. Reset fields and the first non-command observation match; ONNX
parity is below 9e-7. Audit: `/tmp/microduck-native-turn-sign-s23-v2/verification.json`.
This supports a directional native acquisition gap, not a CPU-contact explanation.

## Fixed lineage and boundaries

Source: `/tmp/microduck-auto-consolidation-source-d503d63-v5`, 682 files.
Manifest SHA: `8aa8f6a42c8f46ae38023ed33687a1a31657a7e4136b8a342e9a903128a60ea6`.
Task: `Mjlab-Velocity-Flat-Adaptive-LateralDrive-MicroDuck`.
Source label: `2a4f73f-early-sensor-reset-s23-8aa8f6a42c8f`.
Campaign manifest owns checkpoint path:
`/tmp/microduck-early-sensor-reset-s23-v2/campaign-result.json`.
Checkpoint SHA: `d9de3852ad9f81aa8eb8b8ab6dcf68f527ef6ac5c818d71a705088baa8fda574`.
ONNX: campaign root `cpu-transfer/2026-09-23_13-23-55_matched-lateral-drive-s23.onnx`;
SHA `79a68d44ecbd53631a9081cd52318db2bdcd2971bfda8ccbfee2fb24533c6637`.
Use `uv run --no-sync`; prepend frozen `src`/`scripts` before imports.

Production remains `2a4f73f`; 176 focused tests and prior smoke/live/lineage
proof passed. No production edits in this slice; no repeat test run is needed.
New long training requires smoke64/5. Preserve canonical Velocity, BAM M6,
product DR, unfiltered 61D/14D, reward signs, .80 gates and nominal/zero anchors.
Preserve unrelated IsaacLab/generated/`uv.lock` edits and the preexisting CPU
seed overlay in `scripts/run_specialist_action_battery.py`. Use unique `/tmp`
roots; worktree training logs are unwritable. Do not restart historical agents.

Remaining acceptance: all-six final native mastery, fresh consumed held-out
seeds, full rollout/video review, baked-normalizer ONNX/CPU rehearsal, the same
automated procedure across training seeds 17/23/47, final-range fine-tuning,
matched-budget fixed/axis comparisons. Reused seeds remain diagnostic.
Advisor, stronger teachers, generalist and hardware deployment remain parked.
