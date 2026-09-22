# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
The user authorized sustained necessary changes, training and checks through
`intuitive-flow`; completion remains a reproducible procedure producing usable
walking policies. Latest continuation retains that objective. Previous goal
turn classification: progress (stage-gate repair, real smoke and bounded run).

## Current evidence

The 6,000-update seed-17 checkpoint is at
`/tmp/microduck-adaptive-sensor-reset-s17-6000-r2/` (4,096 training environments).
It is not accepted. Scores are normalized capability scores, not success rates.

| Evaluation of that same checkpoint | Lower-tail score | Result |
| --- | ---: | --- |
| Three-member native gate, final CoM | .267 | fail |
| Same native cohort, checkpointed ±3 mm CoM | .701514 | fail |
| Independent native held-out, final CoM | .653 | fail |
| Corrected CPU/XML rehearsal | .635991 | fail |

The native stage probe keeps the final reference step and gives the same result
as the earlier initial-distribution probe. Its scores are zero `.953`, forward
`.702`, lateral `.773`, yaw `.733`, left `.818`, right `.792`. Artifact:
`/tmp/microduck-adaptive-stage-gate-s17-6500-beea4f8/before-training/capability.json`.

**The former CPU score of zero is superseded.** `931e3c9` fixes a measurement
bug: `mjOBJ_BODY` uses rotated principal-inertia axes at the CoM; commands and
MJLab root-link metrics require `mjOBJ_XBODY`. Same-ONNX replay preserved every
observation, action and physical trajectory exactly in all six buckets.
Corrected CPU scores are zero `.956`, forward `.841`, lateral `.739`, yaw `.636`,
left `.643`, right `.721`. Proof, trace hashes, copied sources and exact frame
conversion are at `/tmp/microduck-adaptive-cpu-link-frame-s17-6000/`.
This repairs measurement; it does not establish transfer usability. The old
actuator A/B tracking scores also used the wrong frame and must not drive tuning.

## Current experiment

Blocker fingerprint: `native_stage0_acquisition_retention`.
The feedback-distribution mismatch is repaired in `beea4f8`: acquisition gates
use checkpointed CoM stages, while independent product checks retain final CoM
and the original .80 threshold. Distribution changes explicitly rebaseline
mastery, EMA and rollback evidence. Final policy acceptance is separate.

A bounded 6,000→6,500 campaign runs under PID `2665136` at
`/tmp/microduck-adaptive-stage-gate-s17-6500-beea4f8/`. Its contract records
commands, checkpoint hash, seeds, 500-update budget and stop condition. Source
is read-only `/tmp/microduck-adaptive-stage-source-beea4f8/`; 679 tracked file
hashes were verified against the adjacent `.manifest.json`.

The 6,250 gate is valid but fails (minimum zero): zero `.956`, forward `.730`,
lateral `.796`, yaw `.000`, left `.747`, right `.726`. On native seed 20260816,
yaw collapsed from a working turn to near-idle without a fall; the other two
members still turned. This is measured acquisition/retention instability.
The 6,500 window and final held-out results remain pending. Do not repeat the
same segment without a decision-changing hypothesis.

The frozen campaign uses the old CPU frame. After completion, rerun its exported
ONNX with the corrected evaluator and attach that result separately. The seeded
CPU proof used an existing uncommitted reset-seed patch in
`run_specialist_action_battery.py`; only the owned frame changes were committed.
A tracked-only snapshot omits this seed patch. Before portable product runs,
explicitly verify consumed reset seeds and record the actual source overlay.
Do not silently describe such a run as the plain committed source tree.

## Proof and next decision

- Stage contract: 250 relevant tests and a real 64-env/5-update, three-member
  gate smoke passed. Runtime proof:
  `/tmp/microduck-adaptive-stage-gate-smoke-beea4f8/runtime-proof.json`.
- Frame correction: 264 relevant tests passed; one existing specialist test
  requires the absent untracked `artifacts/specialist_artifact_manifest.json`
  fixture and was explicitly deselected after reproducing that failure.
  Focused Ruff (`E4,E7,E9,F`) and diff checks pass. Full Ruff is not claimed clean.
- Sensor reset is checkpointed and restored; actor 61D/action 14D, normalized
  ONNX, command isolation and native cohort provenance are verified.
- Next: inspect the live process and `execution-status.json`, then both gate
  windows and the retained final actor. Correct the final CPU report. Compare
  with the before-training report before choosing another training intervention.
- A potential retention gap to assess: migration discards old evidence but the
  external before-training stage report is not yet an in-run rollback baseline.
  Do not infer mastery from old final-distribution reports.

## Remaining acceptance and boundaries

Required: retained six-bucket native mastery, fresh held-out final-distribution
proof, rollout/video inspection, normalized ONNX/CPU deployment rehearsal, the
same automated procedure across independent training seeds 17/23/47, then a
matched-budget fixed baseline. No hardware or adaptive superiority claim exists.

Preserve canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward signs,
0.5 s signed-EMA metrics, thresholds and exact-zero/nominal anchors. IsaacLab,
uv.lock, generated files and unrelated processes remain outside this task.
Advisor, stronger teachers, generalist and hardware deployment remain parked.
Do not repeat sensor-reset-only, slew-only or rejected yaw-planar budgets.

Runs use the validated local venv and TensorBoard under `/tmp` (repo logs are not
writable; no local W&B key). Fresh-sync portability is still unproven. Campaign:
`scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17 --output
<fresh-dir> --iterations <cumulative-updates> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval 250 --gate-cohort-size 3`. Fresh adaptive runs use
stage gates; resumes inherit their saved distribution. A legacy final gate
migration uses `--gate-distribution stage --rebaseline-gate`.
