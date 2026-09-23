# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
User intent: sustained necessary changes, training and checks through
`intuitive-flow`; the status request does not cancel execution.
Project-status writer: not assigned; project-status delta: none.

## Current verdict

Phase 0/1 automation works. Phase 2A acquisition and deployment rehearsal fail.
No long training is running. The −.4 smoothing comparison completed; it failed
its joint-improvement criterion and must not be extended or promoted. The
unchanged entropy-zero continuation is also stopped. **6750 remains the product
diagnostic reference**, not an accepted policy. Both CoM axes remain stage 0,
±3 mm. All six capability buckets must score ≥ .80; scores are not success rates.

## Latest completed treatment

Same 6750 actor/critic/optimizer, seed 17, 4096 envs, 250 updates to 7000,
entropy zero, only relief weight −.2→−.4. Canonical final smoothing is −1.0.
Smoke64/5 and live settings/61D/14D checks pass. All 179 training-source files
match the −.2 control; all 681 source files and report/trace hashes verify.
Session `74497` ended. All 13 retained actor tensors equal candidate/known-good
and differ from start; no rollback. Mean action std .10117→.06785.

| −.4 treatment, retained 7000 | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, worst of 20260815–17 | .964 | .822 | .777 | .672 | .818 | .783 |
| Final native, seed 20260915 | .938 | .885 | .807 | .848 | .746 | .790 |
| CPU v4, seed 20260915 | .970 | .892 | .858 | .613 | .793 | .794 |

CPU left/right improve over the unchanged −.2 continuation, but CPU yaw drops
.761→.613 and native stage yaw drops .790→.672 relative to 6750. Native/CPU yaw
means .885/1.041 rad/s for a .8 command. No joint improvement; **stop treatment**.
Result: `/tmp/microduck-adaptive-smooth-consolidation-s17-250-v2/experiment-summary.json`.
Checkpoint SHA: `29dd98bb98506fc9f9618b88a2cfd25a32b997e42372f2b7495fca809de071f7`.
The summary contains full checkpoint paths, exact contract and retention proof.

## Reference and diagnosis

6750 evidence:
`/tmp/microduck-adaptive-entropy-consolidation-s17-v2/experiment-summary.json`.
Checkpoint:
`/tmp/microduck-adaptive-entropy-consolidation-s17-v2/entropy0-250/training/logs/rsl_rl/matched_lateral-drive/2026-09-23_07-08-07_matched-lateral-drive-s17/model_6749.pt`.
SHA: `95d8fd11b8b7a5acc44669584981f87dbb2b8d8c48460bb6cb51e91bad49be2e`.
Stage cohort scores: .961/.782/.783/.790/.838/.777.
Final native: .936/.875/.812/.857/.815/.797. CPU v4:
.966/.892/.861/.531/.700/.788. Native/CPU yaw means .817/1.104 rad/s.
One final-native diagnostic passes five of six; cohort and CPU still fail.

Blocker fingerprint: `reset_conditioned_acquisition_and_cpu_yaw_rate_transfer`.
Classification: entropy-zero consolidation improves acquisition, while native
sensor/push robustness and CPU turn stability remain open. The unchanged
7000 control retains its actor but fails CPU right stability (turning/surviving,
not idle). The stronger-smoothing follow-up gives mixed results, not a repair.

`/tmp/microduck-transfer-diagnostics-6750/summary.json` verifies 19 reports and
120 traces. Clean native actor inputs or nominal calibration pass six on one
reused seed; nominal encoder calibration affects both observations and targets.
CPU actuator/delay/sensor-lag/integrator changes do not resolve yaw. Corrected
CPU library BAM must match friction constraints by DOF IDs; installed dependency
is unchanged. Partial matched physics is not exact cross-engine replay. Product
CPU stays XML PD with runtime 1.75 A cap (`918bfb6`); do not change evaluator
physics or gates to obtain a pass. Historical details remain in the plan.

## Completed resume repair

New checkpoints persist live PPO entropy. Full resume/rollback restores it;
actor-only load leaves trainer settings alone. Legacy checkpoints lack this
field and still need their explicit launch entropy. A deliberate full-resume
change uses `--env.adaptive-entropy-coef-override 0.0` (also applied on rollback).
The campaign exact-path env factory now inherits relief weight, thresholds,
windows and scope. Conflicting explicit controller settings remain rejected.
No automatic entropy selection or smoothing taper is implemented by this fix.

Proof: 109 focused tests; fresh64/5, legacy-resume64/5, ordinary-resume64/5.
The last run has no entropy/relief override and retains live .0/−.4; all steps
have finite rewards, 61D observations and 14D actions. Source manifest verifies
681 files. Export through `scripts/export.py` bakes normalization; all eight
PyTorch/CPU-ONNX comparison cases pass at 1e-5 tolerance. No additional Ruff
findings; focused Ruff and diff checks pass.
Evidence: `/tmp/microduck-resume-settings-smoke-a408d80/summary.json`.
Source: `/tmp/microduck-resume-settings-source-a408d80`, manifest SHA
`aded72b5050d10e73c7e00e44210843fa339d45c228126f6b4b5f156850a9207`.
MJLab writes launch YAML before runner restore; effective restored entropy is
recorded in checkpoint/result metadata and live proof. These are restoration
checks, not policy-quality continuation or acceptance.

## Next slice and stopping rule

Use frozen gate feedback to make a bounded runner-owned consolidation decision,
reproducing the successful 6500→6750 low-entropy attempt with persisted budget
and rollback. The negative unchanged continuation constrains its stop rule.
Prove it with focused tests, smoke64/5 and the same native/CPU comparisons before
claiming autonomous consolidation. Do not extend the stopped −.2/−.4 treatments,
relax .80 gates, or change product sensor/actuator defaults.

Required acceptance remains: all-six retained native mastery at final ranges,
fresh held-out seeds, rollout/video inspection, normalized ONNX and CPU rehearsal,
autonomous training seeds 17/23/47, canonical final-range fine-tuning and matched-
budget fixed/axis comparisons. Used diagnostic seeds are no longer held out.
Goal is active; no external blocker prevents further work.

Preserve canonical Velocity, BAM M6, unfiltered actions, 61D/14D, reward signs,
zero/nominal anchors and the existing CPU seed patch in
`scripts/run_specialist_action_battery.py`. IsaacLab, generated files, `uv.lock`
and unrelated processes are outside scope. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Use unique `/tmp` directories;
worktree training logs are unwritable.
