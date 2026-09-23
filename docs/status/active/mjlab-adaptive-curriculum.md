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

The early-sensor-reset arm is the strongest acquisition result so far. Applying
50% sensor calibration resets from initialization and retaining the same
automatic consolidation window improved the stage minimum .6617→.7399 (+.0781),
so the controller retained the candidate. Its final native diagnostic was
.962/.837/.790/.855/.748/.801, but CPU was
.980/.854/.780/.456/.728/.863. Relative to the original retained actor, native
minimum gain was only +.0407 and CPU minimum fell by .3206. The CPU pure-yaw
mean was 1.161 rad/s versus .917 for the retained actor. This is diagnostic
evidence of better acquisition with worse transfer, not a usable policy.

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

## Negative timing comparison and command-coverage proof

Continued acquisition completed and audited (sessions 95494/1091):
`/tmp/microduck-seed23-acquisition-comparison-v1/{experiment-summary,verification}.json`.
Original entropy .01 and exposure, zero sensor-reset coverage, same 1000 trainer,
250 updates. Native stage scores .945/.753/.696/.776/.739/.682; CPU
.972/.927/.758/.842/.860/.903. Native minimum is .02077 below the matched
consolidation candidate, CPU minimum .01831 below the original retained actor.
The ordinary preservation gate kept this actor, but the treatment is unsupported.
Eight reports/30 traces, initial trainer/RNG/physics/exposure and all penalties
verify. Do not change consolidation readiness or extend acquisition unchanged.

The half-reset rejected candidate has six exact 300-frame CPU video replays at
`/tmp/microduck-seed23-reset50-cpu-visual-v1/{summary,review}.json`.
Sampled-frame review shows upright idle and stepping/turning; maximum tilt 4.13°.
Yaw oscillation remains; lateral has three sampled trunk/leg self-contact pairs
(max penetration .258 mm). This is sampled-frame/trace evidence, not continuous
video or hardware acceptance. The original rejection is unchanged.

A 100000-command sampler probe shows only .593%/.393% of all samples near the
low-speed left/right command combinations (±20% around .08 m/s and .8 rad/s):
`/tmp/microduck-seed23-command-amplitude-coverage-v1.json`. Broad bucket allocation
alone does not ensure exposure to these command magnitudes. Hypothesis: explicitly
rehearsing a neighborhood improves precision under the unchanged product DR.

The new arm uses the same 1000 trainer/RNG/physics, half sensor-reset coverage,
automatic consolidation and 250 updates as the completed half-coverage control.
Half of nonzero directional samples use ±20% neighborhoods around deployment
command magnitudes; the other samples, nominal pool, zero anchor and signs stay
intact. Independent sampler RNG does not displace the original RNG stream.
Fresh smoke64/5 passed; source imports remain isolated. Require retention plus
≥.05 native minimum improvement over the half-reset candidate without CPU
minimum regression, or all six stage and CPU scores ≥.80. Failure does not
justify unchanged extension. No production command-sampler change yet.

The command-rehearsal arm completed and was rejected:
`/tmp/microduck-seed23-command-rehearsal-v1/verification.json`. It preserved
the same trainer and half-reset setup while placing half of directional samples
near deployment magnitudes. Native minimum gain over the half-reset control was
−.04555 and CPU minimum gain was −.04299; the left-turn score remained about
.701. Do not extend this exposure unchanged.

The fixed early-reset candidate then received a diagnostic CPU action-scale scan
at `/tmp/microduck-early-sensor-reset-s23-v2/action-scale-scan-v1/verification.json`.
The fixed ONNX was replayed at scales .75/.85/.95/1.0/1.1 across all six
buckets, with 30 finite 61D/14D traces and no runtime or training mutation.
The best lower-tail score was only .5748 at 1.10; yaw improved to .575 but
turn-left fell to .627. No uniform action scale reaches .80, so scaling the
whole action cannot repair transfer. The current blocker is now
`cpu_action_amplitude_and_actuator_transfer_mismatch`; the next training
experiment must match actuator behavior or learn a transfer-robust policy.

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
Blocker fingerprint: `cpu_action_amplitude_and_actuator_transfer_mismatch`;
no external blocker. Sensor coverage alone, delayed consolidation and explicit
command-amplitude rehearsal failed the bounded comparisons.

Preserve canonical Velocity, BAM M6, product DR bounds, unfiltered actions,
61D/14D, reward signs, .80 gates, zero/nominal anchors and the preexisting CPU seed
patch in `scripts/run_specialist_action_battery.py`. IsaacLab, generated files,
`uv.lock` and unrelated processes are outside scope. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Use unique `/tmp` roots;
worktree training logs are unwritable.
