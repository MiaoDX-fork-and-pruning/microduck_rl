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

Blocker: `native_mastery_and_unexplained_cpu_transfer`. CPU actuator/scale
sensitivity is established; actuator root cause is **not**. No external blocker.

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

## Next bounded proof

Capture the complete compiled native model, **all** expanded live model fields,
reset state and actual actuator delays. Compare paired CPU dynamics/actor
rollouts to distinguish model/parameter/timing mismatch from policy robustness.
Preserve product settings; no reward changes based on unlocalized sensitivity.

Partial capture `/tmp/microduck-native-physics-capture-early-s23-v1` completed
and reproduces native scores. It saves selected DR fields only, so cannot support
an exact-physics claim. The old CPU wrapper adaptation failed at import before
simulation; `/tmp/microduck-cpu-physical-transfer-early-s23-v1.log` is not a
behavioral result. No training or evaluation sessions remain running from this
slice. Use the complete capture for the next paired proof; do not rerun scans.

Stop condition: bounded comparison changes the implementation/training decision,
or an actual external dependency prevents required proof. A negative result
must not be labeled acceptance or silently lead to another unchanged extension.

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
