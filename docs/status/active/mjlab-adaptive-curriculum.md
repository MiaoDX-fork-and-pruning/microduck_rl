# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — adaptive control plane is executable, but no usable adaptive policy is accepted. Updated: 2026-09-24.
Control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: continue sustained implementation, training and checks after API
interruption via `intuitive-flow`. The stale host `blocked` label is not an
external blocker. Project-status writer unassigned; leave shared status alone.

## Status review (2026-09-24)

The original objective remains open: use adaptive curriculum to produce a
policy that passes the product gate. The adaptive control plane is ready for
bounded follow-up experiments: capability evaluation, checkpoint preservation,
rollback, resume, consolidation, final-range fine-tuning, axis isolation, and
normalizer-baked ONNX export have all been implemented and exercised.

The behavioral acceptance gates are still outstanding. No candidate has been
promoted, all training and evaluation sessions are finished, and no process is
running. This is an experiment-ready state for a new falsifiable hypothesis,
not a deployment-ready policy state. The next experiment must address the
measured directional-tracking/upright trade-off and must not repeat a rejected
exposure-only or unchanged final-range window.

## Current decision

Blocker: `native_mastery_and_product_rehearsal_contract`.
Acquisition, consolidation, retention, complete trainer rollback and explicit
final-range resume work. Six-bucket native mastery, CPU usability and automatic
replication across training seeds 17/23/47 remain unproven. Scores are normalized
capability, not success probabilities; every bucket must reach .80.

Latest comparison uses the retained seed23 turn-exposure **control** at 1250
updates. Its original fresh final cohort (20261001–03) failed. Those seeds are
now reused paired diagnostics; subsequent product acceptance needs new seeds.

| Final native cohort | Zero | Forward | Lateral | Yaw | Left | Right | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source control | .9478 | .7103 | .7929 | .7746 | .7929 | .7257 | .7103 |
| Both CoM axes final | .9621 | .8085 | .7939 | .7104 | .8250 | .6960 | .6960 |
| Trunk CoM only final | .9449 | .7117 | .7790 | .7348 | .7796 | .6781 | .6781 |
| Head CoM only final | .9608 | .7905 | .7799 | .7296 | .7929 | .7081 | .7081 |

All three treatments are rejected. Simultaneous final ranges lose .0642
in yaw (beyond .05 tolerance); trunk-only loses the lower tail without the
required weak-bucket gain. Head-only improves forward by .0802 but its minimum
.708083 is below the source .710326. Keep the source reference and do not extend
any of these windows unchanged. Each arm runs smoke64/5 and exactly 250 new
updates at 4096 environments (24,576,000 transitions), with unchanged entropy, exposure,
calibration reset fraction, rewards and non-owned curricula.

## Completed slice and next action

All training and evaluation sessions have finished; none remain running.
The bounded axis-isolation contract has reached its rejection stop condition.
No candidate is promoted. Reference remains the update-1250 source below.

The audit changes the failure classification: full-final and trunk-only are
limited by sustained right-turn undertracking (1–5 s means −.712/−.662 for
−.8 commands). Head-only's weakest right turn tracks at −.797 over 1–5 s,
but its 95th-percentile tilt is .1783 rad (10.22°): upright component .7081,
tracking component .8302. All limiting traces survive six seconds. This is a
tracking/upright trade-off; another exposure-only or unchanged DR continuation
is unsupported. Next work must use the measured component trade-off to define
a new bounded hypothesis before changing rewards or allocating more training.
No external dependency blocks that work; full autonomous usability remains open.

Contract: `/tmp/microduck-final-axis-isolation-s23-v1/experiment-contract.json`.
Audit command: `uv run --no-sync python /tmp/microduck-final-axis-isolation-s23-v1/audit.py`.
Result: same directory `verification.json`, `complete=true`, all arms rejected.
It verifies fixed budgets/penalty signs, checkpoint contracts, paired reset
fields/initial observations, report/trace hashes, scores recomputed from traces
and all exported actions. Proof covers 72 native traces, 18 CPU traces and
21,600 ONNX action comparisons; native max error 9.54e-7, CPU max error 0.
Full video review has not been performed for these failed candidates. Their
ONNX exports/CPU rehearsals are diagnostic artifacts, not accepted policies.

Completed artifacts:

- Both final: `/tmp/microduck-final-range-finetune-s23-v2/`.
- Trunk only: `/tmp/microduck-final-axis-isolation-s23-v2-com/paired-cohort/capability.json`.
- Head only: `/tmp/microduck-final-axis-isolation-s23-v2-head/paired-cohort/capability.json`.
- Source cohort: `/tmp/microduck-turn-exposure-control-fresh-final-s23-v1/capability.json`.

## Exact inputs and source limits

Source checkpoint is owned by
`/tmp/microduck-turn-exposure-ab-s23-v1/control/campaign-result.json`.
Checkpoint SHA: `0552c8fb458cf81a32586a04838606b912f3f92bc5d17d69801ecc01f2a3daf7`.
Head-only frozen source: `/tmp/microduck-final-axis-source-762a1b0-v1`.
262 files; manifest SHA:
`c345ec7af0dcd3f4ba227d7e9a53072833919fcfcd05ac432163cb9f235305b1`.
Prepend this snapshot's `src` and `scripts` to `PYTHONPATH` for replay. Full-final
and trunk-only ran from the worktree; their source strings are labels, not content hashes. The snapshot was taken after trunk-only
training and before head-only. Do not claim pre-launch immutability for earlier
arms. Frozen native evaluator implementation and consumed reset state agree.

Commits: `fbfbc21` final-range contract, `762a1b0` axis isolation, `d2c865e`
registration isolation and resume-axis checks. 332 adaptive tests, focused Ruff,
smoke and live training pass. Worktree follow-up fixes did not alter the frozen
head-only training. Evaluation and product gates have not been relaxed.

## Prior evidence and boundaries

Directional exposure transfer, symmetry, hip-yaw limit, action-scale scans,
sensor-refresh, BAM-scale and unchanged continuations were rejected; see the
canonical plan. Do not repeat them unchanged. Fully paired native/CPU BAM replay
identified foot/plane contact generation as the main paired simulator gap.
Live CPU contact replacement reduces all six score differences to ≤.02217;
it does not validate original XML-PD product rehearsal or native mastery.
Evidence: `/tmp/microduck-contact-closedloop-s23-v2/verification.json`.

Keep canonical Velocity, BAM M6, product DR, unfiltered 61D/14D, reward signs,
.80 gates and nominal/zero anchors. Preserve unrelated IsaacLab/generated/uv.lock
edits and the existing CPU seed overlay in `scripts/run_specialist_action_battery.py`.
Use unique `/tmp` outputs and `uv run --no-sync`. Do not restart historical agents.

Remaining product gates: all-six final native mastery, new held-out cohort,
full rollout/video review, baked-normalizer ONNX/CPU rehearsal, autonomous
training seeds 17/23/47 and matched-budget fixed/axis comparisons. Advisor,
stronger teachers, generalist and hardware deployment remain parked.
