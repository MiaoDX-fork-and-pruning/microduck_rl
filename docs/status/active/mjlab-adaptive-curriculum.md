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

## Current comparison and live proof

The matched zero/half-reset comparison is complete:
`/tmp/microduck-seed23-sensor-reset-comparison-v3/verification.json`.
Both arms consumed 250 updates and rejected, restoring the same 1000 trainer.
Their retained-model comparison is therefore unchanged. Candidate stage scores:

| Reset coverage | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | .976 | .825 | .782 | .775 | .702 | .715 |
| Half of resets | .963 | .764 | .771 | .823 | .783 | .745 |

Half coverage improved the baseline minimum .699→.745 (+.04555), below the
predeclared +.05 requirement. Each arm's 12 reports/48 unique traces, initial trainer/
RNG/physical equality, bootstrap equality and nonpositive penalties verify.
The treatment observed 20109 calibration refreshes across 37315 reset-environment
visits (5873 reset calls, including initialization; small batches round upward).

Frozen rejected-candidate diagnostics completed in session 2896:
`/tmp/microduck-seed23-reset-candidate-diagnostics-v1/summary.json`.
Control CPU scores .974/.916/.782/.843/.817/.773; half coverage
.968/.888/.868/.853/.819/.880, all six passing on this reused diagnostic seed.
The half-coverage final-native minimum is **.7998718476** (lateral), still a fail;
its remaining native scores exceed .80. All 24 diagnostic traces and matched initial
physical/observation fields verify. Both rejection decisions remain unchanged.

**Live:** poll session `98081`. Root:
`/tmp/microduck-seed23-sensor-reset-comparison-v4`. Wrapper:
`/tmp/run_microduck_seed23_sensor_reset_v4.py`; driver:
`/tmp/train_microduck_seed23_sensor_reset_v2.py`; audit:
`/tmp/verify_microduck_seed23_sensor_reset_v4.py`.

The new arm refreshes calibration on every reset within existing product bounds.
It uses the exact same pre-consolidation 1000 trainer/RNG/physical state, seed23,
4096 envs, automatic consolidation and stage cohort 20260815–17. It runs fresh
smoke64/5, then requests 1500 with the expected controller stop at 1250. Completed
control (`v2/control`) and half coverage (`v3/reset50`) are reused, not retrained;
exact absolute roots are in `experiment-contract.json`. Imports use the immutable
snapshot. Success still requires the unchanged retention/native/CPU gates;
full product acceptance is separate. Do not launch seed47 before usable behavior.

Derived inputs initialize a fresh opt-in controller for frozen bootstrap, change
branch-local rollback/audit metadata and set the treatment fraction. V1's waiting
controller was stopped before PPO (`invalid-start.json`). V2 treatment smoke
completed, but its observer missed registry-cached event functions
(`invalid-monitor.json`). The corrected observer reads `_reset_idx` before/after
without replacing event logic. Original trials/checkpoints remain unchanged.

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
Blocker fingerprint: `reset_conditioned_tracking_precision_and_transfer`;
no external blocker. The live comparison tests full calibration refresh during consolidation after
half coverage improved candidate transfer but failed the native retention rule.

Preserve canonical Velocity, BAM M6, product DR bounds, unfiltered actions,
61D/14D, reward signs, .80 gates, zero/nominal anchors and the preexisting CPU seed
patch in `scripts/run_specialist_action_battery.py`. IsaacLab, generated files,
`uv.lock` and unrelated processes are outside scope. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Use unique `/tmp` roots;
worktree training logs are unwritable.
