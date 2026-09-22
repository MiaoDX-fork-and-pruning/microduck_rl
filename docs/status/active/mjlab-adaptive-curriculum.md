# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
The user authorized sustained necessary changes, training and checks through
`intuitive-flow`; completion remains a reproducible procedure producing usable
walking policies. Latest continuation retains that objective. Previous goal
turn classification: progress (completed campaign, corrected CPU measurement,
and a reproduced teacher-allocation failure). Latest user asked for status;
the sustained execution objective remains active.

## Current evidence

The 6,000→6,500 seed-17 campaign has finished (`execution-status.json` is
`evaluated`); no training process remains. It is not accepted. Validated compact
evidence and exact checkpoint/report paths:
`/tmp/microduck-adaptive-stage-gate-s17-6500-beea4f8/experiment-summary.json`.
Scores are normalized capability scores, not success rates; the pass boundary
remains .80 in every bucket.

| Bucket | Native stage before | Native stage after | Native final held-out | Corrected CPU |
| --- | ---: | ---: | ---: | ---: |
| zero | .953 | .977 | .953 | .976 |
| forward | .702 | .737 | .880 | .918 |
| lateral | .773 | .784 | .814 | .867 |
| yaw | .733 | .248 | .347 | .000 |
| turn-left | .818 | .836 | .711 | .862 |
| turn-right | .792 | .815 | .807 | .816 |

Both CoM axes remain at ±3 mm. The retained known-good set contains zero and
the two turns; the CPU rehearsal passes 5/6 but pure yaw fails. On native seed
20260816, yaw changed from turning to near-idle without a fall. Successful
turning earned more reward than idling; another reward change is not justified
by that observation alone.

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

Blocker fingerprint: `teacher_ignores_severe_unmastered_deficit`.
The feedback-distribution mismatch is repaired in `beea4f8`: acquisition gates
use checkpointed CoM stages, while independent product checks retain final CoM
and the original .80 threshold. Distribution changes explicitly rebaseline
mastery, EMA and rollback evidence. Final policy acceptance is separate.

The teacher still focuses on lateral (.784), although yaw scores .248. Measured
last-window sample fractions are lateral .250 versus yaw .081. Replaying this
score vector from the saved teacher increases lateral probability to .265 and
reduces yaw to .087: the ordinary four-window dwell ignores a severe deficit in
a bucket that has not yet reached mastery and cannot trigger retention repair.

Next bounded test: interrupt configured frontier dwell when another directional
bucket trails the current focus by more than .25 normalized score. Keep the
.80 mastery boundary, .25 sampling update rate and all anchors. After regression
and save/resume tests plus smoke64/5, replay 6,000→6,500 from the same starting
checkpoint, settings and seeds as the completed control. Compare the first gate
before feedback diverges, actual subsequent exposure and retained six-bucket
scores. Stop after 500 updates; do not repeat an unchanged unsuccessful budget.
The canonical plan owns the full experiment contract.

The frozen control's old-frame CPU report is superseded by
`/tmp/microduck-adaptive-stage-gate-s17-6500-beea4f8/cpu-transfer-link-frame/capability.json`.
Its checkpoint and source hashes were verified. The seeded CPU proof used an
existing uncommitted reset-seed patch in
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
- Teacher repair: the two severe-deficit regressions fail on old logic; all
  269 relevant tests now pass (the same missing-fixture test is deselected).
  Real command resampling increases yaw while keeping anchors, and resumed
  feedback matches uninterrupted decisions. Focused Ruff/diff checks pass;
  GPU smoke and behavioral comparison are next.
- Sensor reset is checkpointed and restored; actor 61D/action 14D, normalized
  ONNX, command isolation and native cohort provenance are verified.
- Next proof: regression of the actual .784 lateral / .248 yaw vector, bounded
  real command resampling, identical decisions after resume, then smoke64/5
  and the matched 500-update teacher experiment. A sampling repair alone does
  not prove policy recovery.
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
