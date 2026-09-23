# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
User explicitly resumed sustained changes, training and checks through
`intuitive-flow` after an API interruption. The host goal's stale `blocked`
label is not an external blocker. Project-status writer: unassigned; no delta.

## Current verdict

Automation acquires command behavior, makes a consolidation decision and rolls
back correctly, but all-six mastery and a reproducible autonomous recipe remain
unproven. Scores below are capability scores, not success rates. Both CoM axes
remain stage 0 ±.003; no fresh held-out or multi-training-seed claim is justified.

The seed23 run completed 1250 updates from scratch. At 1000, all native cohort
scores exceeded .60 and zero exceeded .80, automatically triggering a 250-update
zero-entropy window focused on left turn. Five scores improved, but the weakest
left score fell .699→.667. The controller rejected the candidate and restored the
1000 actor/critic/optimizer, entropy .01 and teacher. The spent budget remains 1250.

| Seed23 evidence | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained 1000 actor, stage cohort minimum | .938 | .766 | .737 | .731 | .699 | .777 |
| Rejected 1250 candidate, stage cohort minimum | .977 | .789 | .789 | .813 | .667 | .818 |
| Retained actor, final native diagnostic | .944 | .852 | .793 | .818 | .823 | .811 |
| Retained actor, CPU diagnostic | .966 | .913 | .810 | .776 | .833 | .887 |
| Rejected candidate, final native diagnostic | .956 | .813 | .810 | .873 | .844 | .848 |
| Rejected candidate, CPU diagnostic | .993 | .915 | .870 | .847 | .796 | .873 |

The candidate's final-native pass is one reused diagnostic seed. It remains
rejected; its CPU left and stage cohort fail. Removing encoder bias raises the
failed stage left score .667→.823, while forward/right get worse. Identity IMU
alone gives left .735; both nominal give .759. This supports calibration
sensitivity with interactions, not a sufficient calibration-removal repair.

## Completed sensor coverage comparison

All zero/half/full-reset arms completed and audited:
`/tmp/microduck-seed23-sensor-reset-comparison-v4/verification.json`.
Each consumed 250 updates and rejected, restoring the same 1000 trainer.
Their retained-model comparison is unchanged. Candidate stage scores:

| Reset coverage | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | .976 | .825 | .782 | .775 | .702 | .715 |
| Half of resets | .963 | .764 | .771 | .823 | .783 | .745 |
| Every reset | .967 | .777 | .744 | .756 | .778 | .758 |

Half coverage improved the baseline minimum .699→.745 (+.04555); full coverage
gave .744 (+.04457). Both miss the predeclared +.05 requirement. Each arm's
12 reports/48 unique traces, initial trainer/RNG/physical equality, bootstrap
and nonpositive penalties verify. Half coverage refreshed 20109 of 37315
reset-environment visits (small batches round upward); full coverage refreshed
all 37313 visits across 5931 calls. Session 98081 and audit 97119 exited 0.

Frozen control/half candidate diagnostics:
`/tmp/microduck-seed23-reset-candidate-diagnostics-v1/summary.json`.
Control CPU scores .974/.916/.782/.843/.817/.773; half coverage
.968/.888/.868/.853/.819/.880, all six passing on this reused diagnostic seed.
Half-coverage final-native lateral is **.7998718476**, still below .80; its
remaining native scores pass. All 24 diagnostic traces and matched initial
physical/observation fields verify. Original rejection decisions are unchanged.

Control (`v2/control`) and half coverage (`v3/reset50`) were reused in V4; exact
roots are in its contract. The full-coverage rejected candidate has no separate
CPU diagnostic; final CPU reports assess its restored 1000 actor. Invalid starts
and observer recovery remain documented in the canonical plan and artifacts.

## Live acquisition timing proof

Poll session `95494`. Root: `/tmp/microduck-seed23-acquisition-comparison-v1`;
wrapper `/tmp/run_microduck_seed23_acquisition_v1.py`, observer
`/tmp/train_microduck_seed23_acquisition_v1.py`; prepared auditor
`/tmp/verify_microduck_seed23_acquisition_v1.py`.

Hypothesis: immediate consolidation interrupted ongoing acquisition. The original
750→1000 minimum rose .424→.699 before the switch. The new arm continues from
the identical 1000 actor/critic/optimizer/RNG/physical state, with original entropy
.01, original adaptive exposure and zero sensor-reset coverage. It tests the phase
switch package, not entropy alone: consolidation also redirects exposure to left.

Fresh smoke64/5 passed; run exactly 250 updates at seed23/4096 envs with unchanged
stage cohort 20260815–17 and reused final diagnostic 20260915. Imports are isolated
to the frozen snapshot. Require a retained candidate, native minimum gain ≥.05
over the matched zero-reset consolidation candidate and no CPU minimum regression
versus the original retained actor. A supported result motivates progress-aware
readiness; failure does not justify extending this treatment unchanged. No
production readiness change yet, and no seed47 before usable behavior.

## Verified artifacts and caveats

Seed23 completed run: `/tmp/microduck-adaptive-from-scratch-s23-v4`;
`experiment-summary.json`, `verification.json`, `evaluation-recovery.json`.
Original session 85159 exited 1 only because its driver asserted a 6750/7000 budget
after production completion. Recovery session 2245 exited 0, validated the actual
1250 deadline/checkpoint, then completed final evaluations without retraining.
The original failure is preserved. Final `model_1249.pt` is the restored 1000
actor; SHA `4d2b4731cdae613ab2b81ca622bacef66208f904e88134cbbec7dbeb069b6881`.
Audit: 682 source files, 222 worktree-equal source/script files, 24 reports, 102 unique
traces, 1250 penalty logs per term all ≤ 0, complete trainer rollback equality.
The old wrapper imported training from the editable worktree; equality was
verified before/after, but this was not import isolation. V1–v3 failed before PPO.

Candidate diagnostic: `/tmp/microduck-seed23-consolidation-candidate-diagnostic-v1/summary.json`;
4 reports/12 traces, equal six physical reset fields and first raw actor observation.
The wrapper's audit confused raw CPU ONNX hashes with PT hashes; the completed
reports were audited separately, without rerunning evaluations.
Sensor probe: `/tmp/microduck-seed23-sensor-diagnostic-v1/summary.json`;
24 traces/native-ONNX parity checks pass, physical resets and additive noise match,
actual calibration interventions verify, and all original product trace arrays
replay exactly. Its initial audit rejected the extra `onnx_action` column;
`/tmp/verify_microduck_seed23_sensor_diagnostic_v1.py` verifies the core replay
and that additional parity field. Neither diagnostic changes retention.

Source for current trials: `/tmp/microduck-auto-consolidation-source-d503d63-v5`,
682 files; manifest SHA `8aa8f6a42c8f46ae38023ed33687a1a31657a7e4136b8a342e9a903128a60ea6`.
Production commit `2a4f73f` has 176 focused passing tests plus smoke/live/lineage
proof. Source code has not changed in this resume; do not rerun unchanged tests.

Earlier seed17 V5 retained an improvement at 6750 but failed cohort/CPU mastery.
Its stronger-yaw-weight comparison is negative (16 reports/60 unique traces);
reward-weight-only continuation must not be extended unchanged. Full evidence
and the earlier transfer/smoothing negatives are indexed in the canonical plan.

## Remaining acceptance and boundaries

Require all-six native mastery at final ranges, fresh consumed held-out seeds,
rollout/video inspection, normalizer-baked ONNX/CPU rehearsal, the same procedure
across training seeds 17/23/47, canonical final-range fine-tuning and matched-budget
fixed/axis comparisons. Sampled frames and reused seeds are diagnostic evidence.
Blocker fingerprint: `acquisition_timing_and_reset_conditioned_precision`;
no external blocker. Sensor coverage alone did not satisfy retention; the live
comparison tests continued acquisition before changing controller readiness.

Preserve canonical Velocity, BAM M6, product DR bounds, unfiltered actions,
61D/14D, reward signs, .80 gates, zero/nominal anchors and the preexisting CPU seed
patch in `scripts/run_specialist_action_battery.py`. IsaacLab, generated files,
`uv.lock` and unrelated processes are outside scope. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Use unique `/tmp` roots;
worktree training logs are unwritable.
