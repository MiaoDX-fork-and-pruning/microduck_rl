# MJLab Adaptive Curriculum v2 Plan

Status: Phase 0/1 automation works; Phase 2A usable-policy acquisition remains
open. Seed23's early sensor-reset arm was automatically retained, but the stage
cohort minimum is .7399 and CPU minimum is .4556; both still fail .80. Matched
late sensor coverage, delayed consolidation and command-magnitude rehearsal did
not yield an accepted recipe. Uniform and hip-yaw action scaling, sensor refresh
and nominal CPU BAM replacement all failed their joint transfer gates. These
results establish sensitivity, not an actuator root cause. Complete model/DR/
reset/delay pairing now localizes the main paired CPU regression to foot/plane
contact generation. Replacing only that contact calculation closes all six
native/CPU score gaps to ≤.02217, passing the declared .03 agreement bound.
Native moving-turn mastery and the original XML-PD product rehearsal remain
open; the contact-aligned diagnostic is not product acceptance.
Final-range mastery, fresh held-out evidence, automated training seeds 17/23/47
and matched-budget comparisons remain unproven.
Date: 2026-09-23
Related:

- [`mjlab_adaptive_curriculum_proposal.md`](../mjlab_adaptive_curriculum_proposal.md)
- [`adaptive_curriculum_deep_research.md`](../adaptive_curriculum_deep_research.md)
- [`status/active/mjlab-adaptive-curriculum.md`](../status/active/mjlab-adaptive-curriculum.md)

## Active execution contract

### Sustained objective and acceptance

Status: **ACTIVE**. Task control plane: thread
`01a0c2ae-6895-7700-accd-89a0e7a46e1b`, workspace `holy-ape`.
The user authorized sustained necessary changes, training and checks through
`intuitive-flow` until adaptive training produces usable policies, including
bounded reward/controller/evaluation repairs in the adaptive recipe. The earlier
exposure-only experiment's reward freeze is not the current scope boundary.
Canonical Velocity and the sim-to-real invariants below remain protected.

Acceptance is behavioral: retain all six native capability scores at or above
.80 under the final distribution, then reproduce the automated procedure across
training seeds 17/23/47 with fresh held-out native evidence, rollout/video
inspection, normalizer-baked ONNX and CPU rehearsal. Canonical final-range
fine-tuning and matched-budget fixed/axis comparisons remain required. Native
usability, CPU transfer and adaptive superiority are separate verdicts; no
hardware success follows from simulation. Advisor and stronger teachers remain
parked until acquisition is reliable.

Metrics use signed-error EMA over .5 s before magnitude, with linear/angular
normalization thresholds .12 m/s and .6 rad/s and separate samplewise stability
caps. Scores are normalized capability, not success probability. Keep the .80
pass threshold, 20% exact-zero anchor, nominal exposure and directional floors.

### Best retained consolidation reference

Authoritative evidence:
`/tmp/microduck-adaptive-entropy-consolidation-s17-v2/experiment-summary.json`.
Two branches resume the exact global-relief 6500 adaptive snapshot on immutable
`610fbc5`, seed 17, 4096 envs, with 250 updates each. Both smoke64/5 checks pass;
saved agent configs differ **only** in `algorithm.entropy_coef` (.01 vs .0).
Actual argv, wrapper/source hashes, report/trace hashes and all 680 source files
are verified. This is a local single-training-seed comparison, not superiority
or final acceptance evidence.

The .01 control loses left/right preservation and rolls back to the starting
actor. The .0 treatment retains the new 6750 actor; all 13 actor state tensors
match the accepted candidate/known-good snapshot and differ from the start.
Learned mean action std is .101 (start .280; rejected control .291).

Scores (zero/forward/lateral/yaw/left/right): stage gate cohort
.961/.782/.783/.790/.838/.777; final-native diagnostic seed 20260915
.936/.875/.812/.857/.815/.797; corrected CPU v4 .966/.892/.861/.531/.700/.788.
Native/CPU yaw means are .817/1.104 rad/s for a .8 command. CPU overspeed is
smaller than the previous 1.390, but remains a blocker. Every-bucket .80 mastery
is still absent. CoM axes remain at ±3 mm.

The seed-20260916 full-push case now survives (lateral .739 vs .298), but recovery
tracking is below mastery. The encoder-sensitive seed-20260919 yaw improves
.146→.864, mean .360→.836 rad/s, without weakening sensor DR. Both comparisons
match six recorded reset fields and the initial raw actor observation. Twelve
native/ONNX action parity cases pass (max error 7.16e-7); export correctness does
not establish transfer. These seeds remain diagnostic, not fresh held-out proof.

### Completed 7000 continuation: stop unchanged budget

The follow-up consumed another 250 updates from that exact 6750 checkpoint,
seed 17, 4096 envs, inherited global relief and adaptive command teacher,
unchanged stage gate and immutable source. Smoke64/5 passed. The retained
7000 actor equals its candidate and known-good snapshot across all 13 actor
tensors; no rollback. Mean action std fell .101→.072. Session `43041` ended.
Evidence: `/tmp/microduck-adaptive-entropy-continuation-s17-250/experiment-summary.json`.
Checkpoint SHA: `94c29a191bea9bc7bdfdc7e5f93fcd68fc621845d188f56b02229219a548ee86`.

Scores (zero/forward/lateral/yaw/left/right): stage cohort
.981/.819/.776/.746/.843/.776; final-native diagnostic
.937/.877/.797/.866/.764/.797; corrected CPU v4
.975/.892/.851/.761/.699/.000. Native/CPU yaw rates .824/.946 rad/s improve CPU
overspeed, but native lateral/left and CPU right stability regress. CPU right
survives and turns at -.990 rad/s: samplewise error .615 exceeds the .6 stability
cap; filtered error .201 alone would score .665. It is not idle.

The declared stop criterion fired: no further unchanged low-entropy window.
6750 stays the better product diagnostic reference. These historical checkpoints
predate entropy persistence, so they still need an explicit zero-entropy launch
setting. New checkpoints preserve the live coefficient; see Resume integrity
below. No automatic entropy-consolidation controller or default change has
behavioral acceptance.

### Corrected CPU rehearsal and paired delay diagnosis

`918bfb6` shares the runtime's 1.75 A current cap (±.6405236195572268 Nm) with the
CPU battery, which previously used XML ±.96 Nm. CPU remains an XML position-PD
rehearsal, distinct from native BAM. Evaluator v4 records its actuator profile
and hashes inference implementation as well as battery/rollout code.
Focused proof: 30 tests pass; two missing-specialist-artifact failures reproduce
on pre-fix `610fbc5`. Focused Ruff/diff pass; no added `infer_policy.py` findings.

Full v4 recheck: `/tmp/microduck-cpu-v4-recheck-6750-7000/summary.json`.
Both existing normalized ONNX artifacts verify by hash; all 12 six-second cases
are finite 61D/14D. Reset perturbations and initial observations match v3.
6750 exactly reproduces the earlier manually aligned current-limit diagnostic.
The current fix changes a few scores slightly, but leaves yaw and the 7000
right-turn failure unchanged. Product acceptance still fails for both actors.

Fixed-6750 delay comparison:
`/tmp/microduck-actuator-delay-6750/analysis-summary.json`.
Native delay 3–6 counts 5 ms physics steps, so training latency is 15–30 ms.
The optional CPU inference delay counts 20 ms policy steps; product default is
zero. Diagnostic CPU interventions explicitly delay position targets at the
physics timebase with first-command startup clamping. Native bypass still runs
the original buffer and consumes its RNG, then applies the undelayed command.

Native nominal/zero-delay yaw scores .857/.861 and rates .817/.811 rad/s;
CPU 0/15/20/30 ms scores .531/.345/.291/.223 and rates
1.104/1.229/1.271/1.313 rad/s. This rejects actuator-delay mismatch as a sufficient
explanation or repair for the tested yaw overspeed. Keep product delay defaults.
All recorded physical/sensor resets, first native observation and sampled-lag
sequences match; nominal native reproduces earlier final trace arrays exactly.
All 36 trace hashes and native/ONNX parity pass. One fixed actor and one reused
diagnostic seed limit the conclusion; this is not held-out acceptance.

Diagnostic source: `/tmp/microduck-current-limit-source-918bfb6`, git archive plus
the unchanged pre-existing CPU seed overlay; 681 files verified before/after.
Manifest SHA: `a22c92f5a416f33e954e49ecc47055a0345e177e71ed870ba9267ae85e6be791`.
The following treatment passed smoke64/5 before training on this source.

### Transfer comparison limits and completed smoothing comparison

Diagnostic index: `/tmp/microduck-transfer-diagnostics-6750/summary.json`.
It verifies 19 reports and 120 traces. Replacing native actor slots 0:34 with
current, unbiased physical measurements gives six passing scores on the reused
seed: .954/.895/.830/.901/.847/.840; yaw mean .824 rad/s. Nominal encoder/IMU
calibration also passes all six. That latter intervention changes both actor
observations and physical position targets: mjlab's `JointPositionAction`
subtracts encoder bias before sending targets. Clean actor inputs alone leave
that target perturbation active. Neither diagnostic establishes product DR
robustness, and neither reproduces the CPU overspeed.

CPU Euler and implicitfast produce identical trace arrays at both 5 and 1 ms;
shrinking timestep changes yaw mean 1.104→1.107, so integration is not a repair.
A 20 ms joint-velocity observation lag worsens CPU yaw score .531→.375, improves
left and worsens right. Native removal of that observation lag leaves yaw near
target. Product sensor/actuator delays remain unchanged.

The installed library's `bam.mujoco.MujocoController` matches friction `efc_id`
against joint indexes. On the actual floating-base robot, CPU MuJoCo returns
DOF indexes 6–19, while servo joint indexes are 1–14. Native BAM already uses
DOF indexes. Temporary diagnostic v2 corrects that lookup; no installed package
or product code is changed. Older CPU BAM comparisons are not evidence of exact
native actuator equivalence. Corrected nominal CPU BAM yaw still averages 1.029;
matching captured physical parameters, reset and encoder target bias gives
1.001. Native actuator delay and solver call timing are not replicated, so this
comparison remains a partial physical alignment, not cross-engine parity.

These results do not justify changing a product-harness default to obtain a
pass. Native acquisition exists on this diagnostic condition, while product
sensor robustness and CPU turn oscillation remain open. A bounded consolidation
treatment reduced action-rate relief from −.2 to −.4 at unchanged entropy zero.
Canonical final weight is −1.0; −.4 was an experimental intermediate weight.

Contract: `/tmp/microduck-adaptive-smooth-consolidation-s17-250-v2/experiment-contract.json`.
Authoritative result: `/tmp/microduck-adaptive-smooth-consolidation-s17-250-v2/experiment-summary.json`.
Session `74497` completed. Seed 17, 4096 envs, exactly 250 updates from the same
6750 actor, gate cohort 20260815–17, final diagnostic seed 20260915, preserved
product DR and .80 gates. The derived start differs only in relief weight and
rollback path; actor, critic and optimizer equal original 6750/known-good.
All 179 training-source files equal the earlier control, and all 681 source
manifest entries, report/trace hashes and final checkpoint hash verify.
Smoke64/5 and live −.4 reward-manager, entropy-zero and 61D/14D checks pass.

The retained 7000 actor equals the candidate and known-good across all 13 actor
state tensors and differs from the start. No rollback. Mean action std falls
.10117→.06785. Checkpoint SHA:
`29dd98bb98506fc9f9618b88a2cfd25a32b997e42372f2b7495fca809de071f7`.

| −.4 treatment, retained 7000 | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, worst of three | .964 | .822 | .777 | .672 | .818 | .783 |
| Final-native diagnostic | .938 | .885 | .807 | .848 | .746 | .790 |
| CPU v4 | .970 | .892 | .858 | .613 | .793 | .794 |

Native/CPU pure-yaw means are .885/1.041 rad/s for a .8 command. CPU right
stability recovers relative to the unchanged −.2 control, and CPU left improves;
but CPU yaw worsens relative to that control (.761→.613), and stage native yaw
falls from .790 at 6750 to .672. Final native has four passing buckets; CPU has
three. Scores are not success rates. The declared joint-improvement criterion
fails: **stop the treatment; do not implement/promote a −.4 taper or extend it**.
6750 remains the reference, not an accepted usable policy. Reused diagnostic
seeds are not fresh held-out acceptance.

### Resume integrity

Root causes reproduced before the fix: RSL-RL persists model/optimizer tensors,
but omits PPO entropy; the adaptive env factory inherited only relief scope,
so a saved −.4 controller was rebuilt with −.2 and rejected on full restore.
Eight targeted regression cases failed before the fix (two compatibility cases
already passed). New adaptive checkpoints record the live entropy coefficient;
full resume/rollback restores it, while actor-only load does not change it.
Legacy checkpoints without this metadata retain the configured launch value.

For a deliberate change on full resume use
`--env.adaptive-entropy-coef-override 0.0`. It is recorded and applies after
full loads, including rollback. Ordinary subsequent resumes need no override.
Fresh/legacy runs also support the existing `--agent.algorithm.entropy-coef`.
The campaign's exact-path resume reconstructs relief weight, thresholds, window
limits and scope before environment creation; explicit conflicting controller
settings still fail validation. This does not choose entropy or smoothing
adaptively. MJLab writes `params/agent.yaml` before runner restore; the effective
restored entropy is in checkpoint/result metadata and the live proof, not
necessarily that launch-config file.

Proof: 109 focused tests pass; focused Ruff/diff checks pass with no additional
findings in the broader pre-existing lint baseline. Real fresh64/5,
legacy-resume64/5 and ordinary-resume64/5 all pass, with live entropy zero,
ordinary-resume relief −.4, finite rewards/61D observations/14D actions.
The final smoke checkpoint also exports through `scripts/export.py` with baked
normalization; eight PyTorch/CPU-ONNX comparison cases pass at 1e-5 tolerance.
`/tmp/microduck-resume-settings-smoke-a408d80/summary.json` records exact commands,
checkpoint hashes and effective values. These short runs verify restoration,
not policy improvement or matched-budget quality. Source snapshot:
`/tmp/microduck-resume-settings-source-a408d80`, 681 manifest entries, SHA
`aded72b5050d10e73c7e00e44210843fa339d45c228126f6b4b5f156850a9207`.

### Automatic consolidation: completed V3 and V5 comparison

The opt-in `--entropy-consolidation` campaign flag enables a bounded runner-owned
attempt. A fresh validated native cohort must show all six ≥ .60, zero ≥ .80,
weakest < .80, and positive current entropy. On legacy resume it re-evaluates the
actual retained actor without consuming another teacher window. Historical
per-bucket best scores do not authorize the switch.

The runner saves a baseline, selects entropy zero and persists the original
coefficient, scores/path and one-window update budget. The existing gate must
preserve mastered capabilities, and the weakest score must gain ≥ .05 (or all
six master), to retain the candidate. Otherwise restore the baseline trainer.
Failed evaluation also rejects the attempt. Either outcome stops training and
runs final-native/CPU evaluation. Result manifests distinguish requested and
actual completed updates; resumes preserve deadlines, and pending-at-deadline
evaluation occurs before further PPO updates. Terminal attempts cannot extend
unchanged. No acceptance threshold changes.

V3 is complete: `/tmp/microduck-auto-consolidation-s17-v3/experiment-summary.json`
and `verification.json`. It automatically selected zero entropy for 6500→6750,
then rejected turn-right preservation (.812→.726), restored baseline entropy .01
and stopped at 6750 despite requesting 7000. Worst score .669→.702 also misses
the required .05 gain. Candidate stage scores:
.970/.748/.794/.702/.790/.726 (zero/forward/lateral/yaw/left/right).
Retained final native: .923/.879/.797/.837/.737/.820; CPU:
.940/.903/.730/.000/.696/.849. The retained actor matches baseline/known-good
and differs from the rejected candidate. Baseline actor/critic/optimizer match
the original 6500 snapshot. All 682 source files, 12 reports, 48 unique traces and
checkpoint hashes verify. Retained checkpoint SHA:
`f91a92fa1f66648adf6632c2525833b4b3ee47516217330ce510d89685a135fd`.
This proves automatic negative-decision execution, not usability or reliable
consolidation improvement. The earlier manual 6750 remains the reference.

A stale lateral exposure focus while yaw is weakest motivates a second treatment,
but has not been proven causal. V5 redirects one 25%-toward-target exposure slice
to the weakest directional bucket after saving the baseline, applies it to the
live command manager, and checkpoints the selected bucket (state version 2 with
version-1 migration). Rejection restores original teacher state as well as PPO.
The zero/nominal anchors and directional floors remain fixed. No reward or DR
change, no manual entropy value, no change to product gates or actor ABI.

176 focused tests pass, including actual manager values, teacher rollback,
partial resume, pending-at-deadline evaluation and early-stop transition budgets.
V1/v2 failed before PPO updates (logger initialization / relative-path validation)
and are covered by regressions. V4 was intentionally terminated after at least
11 updates because probabilities never reached the manager; its
`/tmp/microduck-auto-consolidation-s17-v4/invalid-intervention.json` excludes it
from treatment comparisons. Preserve these failed artifacts; do not restart.

V5 completed with training seed 17, 4096 envs, cohort 20260815–17 and reused
diagnostic seed 20260915. Fresh smoke64/5 passed. The request was 7000 total;
the controller retained the improved candidate and stopped at 6750 after exactly
250 new updates (24,576,000 transitions). Session `10932` and audit `40226` exited 0.
Baseline actor/critic/optimizer equal the exact 6500 start; V3/V5 bootstrap scores
match. Retained actor equals the evaluated candidate and known-good, differs from
baseline, and has mean action std .10010 (start .27970), with entropy zero.

| V5 retained actor | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage cohort, worst of three | .962 | .754 | .792 | .732 | .819 | .819 |
| Final native, diagnostic seed 20260915 | .937 | .886 | .810 | .851 | .802 | .811 |
| CPU v4, diagnostic seed 20260915 | .981 | .902 | .851 | .668 | .754 | .766 |

The weakest cohort score gains .0626 while mastered capabilities are preserved,
so retention follows the declared rule. One cohort seed passes all six; two do
not. The final-native diagnostic passes all six with a left-turn margin of only
.0025. Native/CPU yaw means .840/1.004 rad/s for a .8 command. CPU remains upright
but turning accuracy fails. This is evidence of one automatically selected and
retained improvement; it is not product acceptance or proof of a causal focus
mechanism. Source/trace audit verifies 682 source files, 12 reports and 48 unique
traces; live manager yaw probability was .1368 during training, from baseline
.0890. The ordinary end-of-window teacher update leaves .1726 in the checkpoint.

Contract, summary and audit:
`/tmp/microduck-auto-consolidation-s17-v5/{experiment-contract,experiment-summary,verification}.json`.
Source: `/tmp/microduck-auto-consolidation-source-d503d63-v5`, manifest SHA
`8aa8f6a42c8f46ae38023ed33687a1a31657a7e4136b8a342e9a903128a60ea6`.
Retained checkpoint SHA:
`311265f7d9258c6de29845675d1b68bcf499e247ee80d43e6749d59064167aea`.
The summary owns the full checkpoint path and retention proof. Focused Ruff and
diff checks pass; worktree code/tests match the verified source. README and the
specialist reproducibility document remain unchanged after documentation review.

V5 and the manual 6750 actor remain comparison references. Seed23 now provides
from-scratch acquisition, automatic rejection/rollback and one retained early
sensor-reset improvement. Sensor timing/coverage and command-magnitude training
comparisons are complete; the active work is the native/CPU transfer diagnosis
below. Terminal attempts must not extend unchanged. Native cohort mastery, fresh
held-out evidence, multi-training-seed replication, CPU usability and final CoM
mastery remain open. The controller remains opt-in.

### Fixed V5 robustness and bounded yaw-precision comparison

`/tmp/microduck-v5-robustness-diagnostics/summary.json` compares the retained V5
actor against the manual 6750 reference at stage distribution, seeds 20260916
(push) and 20260919 (sensor corner). All 12 native/ONNX comparisons pass, six
recorded reset fields and initial actor observations match exactly, and all
traces/source hashes verify. Both batteries remain upright. Push lateral improves
.739→.781; sensor lateral .748→.795 and right .773→.830. Sensor forward remains
.762, yaw .852. These known diagnostic seeds are not fresh held-out evidence.
All failed buckets in this probe and V5's gate/CPU reports are tracking-limited.

`/tmp/microduck-v5-cpu-visual/summary.json` records yaw/left/right videos with
900 frames. Rendering preserves every recorded trace array exactly. Eighteen
sampled frames show upright, foot-supported turning; contact logs contain only
floor/feet. Instantaneous yaw oscillation and persistent over/underspeed remain.
This is sampled-frame review, not full temporal or hardware acceptance.

### Negative yaw-precision comparison and from-scratch seed-23 proof

The bounded precision comparison at `/tmp/microduck-yaw-precision-s17-v2` is
complete. Control and treatment each resumed the exact V5 retained actor at 6750
for 250 updates, with fresh smoke64/5, seed17/4096 environments, entropy0,
relief−.2 and the same gate/CPU evaluation. Only the treatment changed signed-EMA
yaw L1 weight 1.0→2.0. Both source manifests contain 682 files and differ only
in that config line; actor, critic, optimizer and RNG at each start are equal.
The audit verifies 16 report files, 60 unique traces, finite arrays and all
penalty values ≤0. Session `16017` exited 0; `verification.json` records the
completed audit. Derived starts cleared terminal consolidation only for this
separately contracted comparison and used branch-local rollback baselines;
the original V5 checkpoint and its audit history are unchanged.

Control native-cohort minimum gain over V5 was +.0434 and CPU minimum gain −.1492;
treatment gains were +.0411 and −.1233. Neither reaches the joint +.05 retention
rule. Control final native left is .726 and CPU yaw .519; treatment CPU yaw/right
are .579/.545. This rejects reward-weight-only precision repair and stops both
arms. It does not alter product defaults or the adaptive controller.

`/tmp/microduck-adaptive-from-scratch-s23-v4` completed 1250 updates from scratch,
seed23/4096 envs, with smoke64/5 and stage cohort 20260815–17. At 1000 its scores
were .938/.766/.737/.731/.699/.777, triggering zero-entropy consolidation with
left-turn focus. At 1250 the candidate scored .977/.789/.789/.813/.667/.818.
Five scores improved, but the weakest left regressed; the controller rejected
and restored the 1000 actor/critic/optimizer, entropy .01 and teacher. Actual spent
budget remains 1250. Both CoM axes remain stage 0 ±.003.

The inherited post-return driver assertion allowed only 6750/7000 updates.
Session 85159 therefore exited 1 after the valid production checkpoint/result;
recovery session 2245 validated the actual deadline and completed final native/CPU
evaluation without retraining. `verification.json` verifies682 source files,
222 worktree-equal source/script files, 24 reports, 102 unique traces and1250 nonpositive
penalty records per term. The original wrapper failure remains in
`evaluation-recovery.json`. Its imports used the worktree, with source equality
checked before/after; this was not isolation. V1–v3 failed before any PPO update.

| Seed23 retained/candidate diagnostics | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained 1000 actor, final native | .944 | .852 | .793 | .818 | .823 | .811 |
| Retained 1000 actor, CPU | .966 | .913 | .810 | .776 | .833 | .887 |
| Rejected 1250 candidate, final native | .956 | .813 | .810 | .873 | .844 | .848 |
| Rejected 1250 candidate, CPU | .993 | .915 | .870 | .847 | .796 | .873 |

Candidate comparison: `/tmp/microduck-seed23-consolidation-candidate-diagnostic-v1/summary.json`.
All 12 traces and physical resets verify. This uses diagnostic seed 20260915 again;
the one final-native pass does not override cohort rejection or CPU failure.

Fixed-candidate probe: `/tmp/microduck-seed23-sensor-diagnostic-v1/summary.json`.
At failed gate seed 20260815, left .667→.823 with zero encoder bias, while forward
and right regress. Identity IMU alone gives .735; both nominal give .759. All 24
native/ONNX comparisons pass, physical resets/additive-noise settings match and
actual sensor realizations verify. The product trace replays exactly. Encoder
bias affects both observations and position targets, so this is evidence of
calibration sensitivity and interactions, not a sufficient removal repair.
Initial diagnostic audits mishandled the CPU ONNX hash and added `onnx_action`
column; separate auditors validate the already completed data without reruns.

The matched zero/half coverage comparison completed at
`/tmp/microduck-seed23-sensor-reset-comparison-v3`; `verification.json` passes.
Both arms restored the exact pre-consolidation 1000 trainer/RNG/physical state,
ran smoke64/5 and 250 automatic-consolidation updates with seed23/4096 envs,
stage cohort 20260815–17 and unchanged reward/inference/product DR bounds. Control
candidate minimum .702 and half-coverage minimum .745 both fail the required
+.05 gain over baseline .699, so both restore the 1000 actor. The latter's actual
gain is +.04555, not a pass. Per arm, 12 reports/48 unique traces, source/lineage,
bootstrap equality and all penalty logs verify. The live observer records 20109
sensor refreshes across 37315 reset-environment visits, with ceiling rounding
for small batches.

Because rollback returns the same actor, frozen candidate diagnostics were run
separately: `/tmp/microduck-seed23-reset-candidate-diagnostics-v1/summary.json`.
Control CPU is .974/.916/.782/.843/.817/.773; half coverage is
.968/.888/.868/.853/.819/.880, all six passing on the reused diagnostic seed.
Half-coverage final-native lateral is **.7998718476**, strictly below .80; the
other five pass. The three-seed stage minimum remains .745. All 24 traces and
matched initial physical/observation fields verify. Neither rejection is changed.

The full-coverage arm completed at
`/tmp/microduck-seed23-sensor-reset-comparison-v4`; session 98081 and audit 97119
exited 0. Candidate stage scores are .967/.777/.744/.756/.778/.758; the minimum
improved .699→.744 (+.04457), below the required +.05. The controller rejected
and restored the same 1000 trainer. All 37313 reset-environment visits across
5931 calls refreshed calibration. Source, 12 reports/48 traces, actual 250-update
budget, bootstrap equality, trainer rollback and nonpositive penalties verify.
V4 reused completed control/half coverage via exact contract paths. Its CPU
report evaluates the restored actor; no separate full-coverage candidate CPU
claim is made. Coverage alone did not pass native retention in these trials.

The early-reset-from-initialization arm completed and was retained:
`/tmp/microduck-early-sensor-reset-s23-v2/verification.json`. It applied 50%
sensor calibration reset coverage from update 0, then used the same automatic
consolidation window. The stage candidate improved the weakest score
.6617→.7399 (+.0781), satisfying the controller's retention rule. Final native
diagnostic scores were .962/.837/.790/.855/.748/.801, while CPU scores were
.980/.854/.780/.456/.728/.863. Native minimum gain over the original retained
actor was only +.0407 and CPU minimum gain was −.3206. CPU pure-yaw mean over
the final five seconds was 1.161 rad/s versus .917 for the original actor; the
early-reset actor's full six-second mean is 1.109321 rad/s. This supports early
acquisition diversity as a native training signal. Native mastery and CPU
transfer both remain insufficient; actuator causality is not established.

The acquisition-timing comparison completed and audited:
`/tmp/microduck-seed23-acquisition-comparison-v1/{experiment-summary,verification}.json`.
Same 1000 trainer/RNG/physics, 250 updates with original entropy .01/exposure,
zero reset coverage, seed23/4096 envs and unchanged evaluation seeds. Native
stage scores .945/.753/.696/.776/.739/.682; CPU .972/.927/.758/.842/.860/.903.
The ordinary preservation gate kept the candidate, but native minimum is .02077
below the matched consolidation candidate and CPU minimum .01831 below the
retained baseline. This does not support changing readiness or extending the
continued-acquisition treatment. Eight reports/30 traces, exact initial trainer/
RNG/physics/exposure and nonpositive penalties verify; no production change.

Six 300-frame CPU replays of the rejected half-reset candidate exactly reproduce
its traces: `/tmp/microduck-seed23-reset50-cpu-visual-v1/{summary,review}.json`.
Sampled frames show upright idle and stepping/turning; full traces show no fall
and maximum tilt 4.13°. Angular-rate oscillation remains. Lateral has three
sampled trunk/leg self-contact pairs, maximum penetration .258 mm. This is
sampled-frame evidence; continuous-video and hardware acceptance are not claimed.

The command-magnitude comparison completed and was rejected:
`/tmp/microduck-seed23-command-rehearsal-v1/verification.json`, session 98777.
A 100000-sample probe finds only .593%/.393% of all samples within ±20% of the
low-speed left/right command combinations. The treatment keeps the same 1000
trainer/RNG/physics, half sensor-reset coverage, automatic consolidation and
250-update budget. Half of directional samples rehearse ±20% around deployment
command magnitudes, retaining signs, unselected samples, nominal and zero pools.
An independent sampler RNG preserves the original random stream; live assertions
verify selection, bounds and untouched commands. Fresh smoke64/5 passed.

Require controller retention and ≥.05 native minimum gain over the half-reset
candidate without CPU minimum regression, or all six stage/CPU scores ≥.80.
The native minimum gain was −.04555 and CPU minimum gain was −.04299; left-turn
remained about .701. Failure rejects this unchanged exposure recipe and does not
justify a production sampler change. Product DR, rewards, .80 gates and
inference stayed fixed; reused seeds remain diagnostic.

### Fixed-policy CPU transfer diagnosis

The early-reset ONNX remains fixed. Uniform action scales .75/.85/.95/1.0/1.1
never pass all six buckets: best minimum .5748 at 1.10, with yaw .575 and left
.627. Scaling only the hip-yaw joints .5/.75/1/1.25/1.5 also has no passing
window: at 1.5 yaw reaches .8313 while left falls to .5594. Joint sensitivity
is not a root-cause identification; neither scaling becomes a runtime default.

`/tmp/microduck-cpu-transfer-causal-s23-v1/verification.json` independently
recomputes scores and ONNX actions from 66 saved traces (11 variants), checks
actual delivered targets, matched resets, correct task/source metadata and all
682 source-file hashes. Both five-scale scans reproduce their original arrays
exactly. The earlier v1 scan reports recorded requested rather than scaled
`applied_action` and had imprecise metadata; the causal audit supersedes those
limitations without treating a retrospective replay as preregistration.

Refreshing CPU sensors with `mj_forward` at the 20 ms control boundary gives
.97956/.85534/.78273/.44376/.71837/.87261. Yaw changes by −.01187, rejecting
sensor refresh alone as a sufficient explanation.

Nominal BAM M6 replacement was paired against XML PD in the same CPU scene:
`/tmp/microduck-cpu-bam-response-s23-v1/verification.json` (3 reports/18 traces).
The independent audit checks reset/first-observation equality, all 1200
physics-step delivered targets and torque limits, replays original XML traces
exactly, and recomputes every saved ONNX action and capability score.

| CPU actuator diagnostic | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XML PD | .97959 | .85440 | .78006 | .45563 | .72832 | .86311 |
| BAM M6 | .98257 | .73820 | .86140 | .57954 | .65825 | .68747 |
| BAM M6, stiff friction | .98040 | .74205 | .86900 | .56243 | .62216 | .65884 |

BAM improves yaw but regresses other buckets beyond the predeclared .02 limit;
both treatments fail. Full-six-second yaw means are 1.10932/1.01084/1.01754
rad/s respectively. The diagnostic fixes the dependency's joint-versus-DOF
friction-constraint indexing locally; it does not change the installed package.
This comparison does not replicate native scene, DR, observation or delay.

The partial live-parameter capture at
`/tmp/microduck-native-physics-capture-early-s23-v1` replays native capability
scores but cannot support exact-model parity. The partial CPU v2 verifier's
delivered-target claim was corrected: its arrays precede encoder-bias subtraction
and verify requested targets only. The original verification file is retained
as `verification.superseded-target-claim.json`. The CPU v1 import failure has no
behavioral result. External-native-observation open-loop v1/v2 replays are
explicitly invalidated for causal inference (wrong timestep/BAM cadence and
reset/termination defects); their falls are not evidence of policy failure.

### Complete native/CPU contact diagnosis

Capture: `/tmp/microduck-complete-native-s23-v2`. For each reused diagnostic seed
20260915–20260920 it saves the compiled model, every expanded live field (13),
reset/calibration, initial previous motor torque and all 1200 physics substeps'
requested/delayed targets, sampled lags, applied torques and states. Reset has
already inserted one zero target into the delay history; the older first-policy-
target startup clamping approximation was incorrect.

The native actor sees current unbiased root/link and joint state; ordinary
observation computation still consumes its RNG. Product physics/DR/delays remain
active. CPU uses the same complete model, fields, reset and lag sequence, BAM
updated every 5 ms, and the same current actor view. First observations match
within 7.2e-15; initial BAM force/friction errors are below 1e-7. Recorded push
increments are applied without forcing trajectories. Scores:

| Paired diagnostic | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | .96742 | .84734 | .82243 | .81978 | .73985 | .76863 |
| CPU original contacts | .95017 | .82833 | .74120 | .31496 | .74189 | .69148 |
| CPU native foot/plane contacts | .96718 | .84813 | .82006 | .81439 | .71768 | .76612 |

`/tmp/microduck-complete-cpu-s23-v1/verification.json` independently verifies
12 traces, scores, ONNX actions, actual delayed targets, model/field hashes and
all 682 source files. It exactly reproduces 7200 CPU one-step comparisons from
recorded native inputs, removing closed-loop drift and BAM recomputation.

Contact-state comparison: `/tmp/microduck-contact-pair-s23-v1/summary.json`.
Warp exactly reproduces recorded native next states at substeps 0/7/8/9/638.
CPU agrees before contact, but at substep 8 (40 ms) generates two foot contacts
where Warp generates eight; at substep 638 it generates one versus four.
CPU 100 versus 10 Newton iterations gives identical results; Warp sequential
line search changes qvel by at most .000102 at these states.

Replacing only the contact manifold, retaining CPU integration and forces,
reduces maximum qvel errors at 8/9/638 from .38745/.21135/.66790 to
.000102/3.12e-7/4.28e-7:
`/tmp/microduck-contact-injection-s23-v1/summary.json`.
The installed Warp `plane_convex` chooses up to four separated support points;
MuJoCo 3.10's plane-mesh path searches the deepest point's immediate graph
neighbors. The difference matters for these sole meshes.

The closed-loop intervention computes fresh Warp foot/plane contacts on CPU
from the live CPU pose, retaining other CPU contacts, the solver, BAM and all
other paired settings. It uses neither recorded contacts nor native trajectory
forcing. All six score gaps are ≤.02217 and yaw mean .73741 versus native
.73938 rad/s, passing the declared .03 score / .05 rad/s agreement criterion.
Independent audit: `/tmp/microduck-contact-closedloop-s23-v2/verification.json`.
It validates fresh contact geometry against native snapshots, every saved ONNX
action, scores, delivered targets and immutable inputs. The earlier v1 wrapper
failed on contact-frame array shape and produced no completed battery.

Blocker fingerprint is now `native_mastery_and_product_rehearsal_contract`.
The paired CPU regression has a demonstrated contact-generation cause; this
does not establish hardware correctness or validate the original XML-PD runtime.
Keep original-runtime and native-matched rehearsal verdicts separate. Do not
replace product acceptance with simulator agreement or continue scale scans.
Native steady 1–5 s moving-turn yaw is +.708/−.729 for ±.8 commands, with
further post-push degradation. Next acquisition work measures the unmodified
native actor's turn rollouts before a bounded training intervention. No production
source or dependency changed in this slice; all seeds remain diagnostic.

The sensor-coverage derived inputs initialize a fresh controller for bootstrap and
change only branch-local rollback/audit metadata plus the coverage fraction.
Original checkpoints stay unchanged. V1 preserved a waiting controller and its
live guard stopped before PPO; V2's treatment smoke completed but its observer
missed the registry-cached event function. Their invalid-start/monitor records
remain excluded. The corrected observer reads actual `_reset_idx` changes without
replacing event logic. The active capsule owns current handles and audit commands.

### Pre-consolidation reference and measured limits

Authoritative evidence:
`/tmp/microduck-adaptive-action-rate-s17-500/experiment-summary.json`.
The seed-17, 4096-env campaign consumed 6250→6750 updates after smoke64/5.
Its 6750 candidate failed turn preservation and was rolled back to the 6500
actor; all 13 actor-state tensors match the known-good snapshot and differ from
the rejected actor. A final filename of `model_6749.pt` records consumed budget,
not a newly retained 6750 policy. Both CoM axes remain at stage 0 (±3 mm).

The matched first 250 updates used identical starting checkpoint and gate seeds
20260815–17: native stage yaw .353 in the control versus .669 with relief.
Both failed mastery. The full relief campaign spent 500 updates and cannot be
reported as a matched 250-update total-budget success.

Retained final native scores (seed 20260915): zero .923, forward .879, lateral
.797, yaw .837, left .737, right .820. Corrected CPU: .940, .903, .727, .000,
.696, .849 in the same order. CPU yaw averages 1.390 rad/s for a .8 command;
native final yaw averages .843. The CPU samplewise error .655 exceeds the .6
cap. This is overspeed, not idle, and is not explained by export mismatch.

The completed stage diagnostic over seeds 20260915–24 has worst scores .914,
.696, .298, .146, .707, .767. No member passes all six; yaw and lateral each
pass 3/10. Lateral falls on seed 20260916. Worst yaw seed 20260919 starts slowly,
briefly reverses, then reaches .86 rad/s in the last two seconds. All 60
native/ONNX action comparisons pass (max absolute error 1.67e-6), with all
report/trace hashes verified. These diagnostics do not replace final-range
held-out acceptance. Their seed set is now diagnostic evidence and future
acceptance must use a fresh held-out set.

### Implemented controller and evidence contracts

`ef4734e` enables bounded action-rate relief only in the adaptive LateralDrive
recipe. Worst yaw/turn capability below .55 activates weight -.2; release
requires .80 yaw/turn mastery or four windows, followed by cooldown. The live
training weight was verified. `75f479f` ensures relief never strengthens an
earlier smaller penalty, clears stale state on legacy loads, prevents rejected
candidates from releasing relief and validates full-resume controller state.
83 focused tests plus the subsequent 6 relief tests pass; focused adaptive Ruff
and diff checks pass. Full `mdp.py` Ruff retains existing findings.

Training used `ef4734e`; the later fixes have not had long-run training proof.
The training snapshot contains 1001 verified copied-worktree files, including
the explicit pre-existing CPU seed overlay. The current diagnostic snapshot
`/tmp/microduck-adaptive-relief-source-75f479f` is `git archive 75f479f` plus
only that overlay (680 verified files). Exact manifests/hashes are in the
experiment summary. New-source training must first pass smoke64/5.

Previously proven infrastructure remains required: runner-owned gate/checkpoint/
rollback state; persisted command/transition/sensor settings; stage-matched
native gate cohorts with exact member provenance; clearing incomparable evidence
on distribution/stage changes; and final-distribution held-out checks kept
separate. `931e3c9` measures CPU velocities in the link frame/origin; old
principal-inertia-frame scores are superseded. A fresh locked non-editable
install/config/model compile passes; fresh-environment GPU and remote execution
remain unproven. Existing tests and smoke artifacts are linked from the capsule
and earlier experiment summaries.

### Pre-consolidation sensor diagnosis

Historical diagnosis; current blocker: `native_mastery_and_product_rehearsal_contract`.
Relief helped yaw but did not resolve startup/DR robustness or CPU transfer;
blindly extending global relief or direct-yaw/turn-proxy exposure is not the next
experiment. The old 6250 actor's additive-white-noise ablation did not rescue
its failed seed; persistent encoder bias and IMU misalignment were untouched.

The completed fixed-calibration diagnostic holds the actor and additive noise
fixed across nominal/zero-encoder/identity-IMU/both-nominal conditions for seeds
20260915/16/19. All 72 trace hashes pass, six recorded physical reset fields
match between conditions, and actual sensor realizations confirm the intended
interventions. Evidence:
`/tmp/microduck-relief-6500-sensor-bias/analysis-summary.json`.

Seed 20260919 yaw rises .146→.807 with zero encoder bias and mean rate
.360→.830 rad/s; identity IMU alone leaves .115. Seed 20260915 yaw rises
.744→.859 with identity IMU. Lateral still falls on seed 20260916 in all four
conditions. This confirms calibration sensitivity in the tested yaw cases,
while ruling out calibration removal as a sufficient repair for the lateral
failure. It is not product acceptance and does not explain CPU overspeed.

### Bounded pure-yaw relief comparison

The lateral fall is triggered by a seeded push at 3.9 s. Holding recorded
physical/sensor reset state and the pre-push trace fixed, half/no push changes
lateral survival from failure to full survival and scores .298→.801/.807. The
matched 6500 canonical-smoothing control also survives full push (.790), with
identical recorded initial state and actor observation. Actual qvel write
proof and the original cached-velocity logging limitation are recorded at
`/tmp/microduck-relief-6500-push-sensitivity/analysis-summary.json`.

Hypothesis: restricting relief to pure-yaw commands retains its acquisition
benefit while avoiding the lateral recovery regression caused during global
relief training. This remains a training hypothesis, not a proven repair.

Implementation is opt-in through
`MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE=pure_yaw`: positive action-rate
costs are scaled only for zero-planar/nonzero-yaw commands, with the canonical
manager weight and other command costs retained. Pure-yaw capability controls
the existing bounded window. Versioned state preserves the scope through resume
and rollback; legacy controllers remain global and mismatches fail closed.
Default LateralDrive stays global. Product pushes, sensors, commands, reward
signs, ABI and gates are unchanged. Focused tests, smoke64/5 and a 64-env live
weighted-cost probe passed. The 250-update treatment completed at
`/tmp/microduck-adaptive-pure-yaw-s17-250` on immutable `610fbc5` plus the
recorded CPU seed overlay. All 680 source files and retained actor equality
were verified; the candidate was retained without rollback, but not accepted
as usable. Its authoritative `experiment-summary.json` records the negative result.

Training contract: exact common 6250 checkpoint, seed 17, 4096 environments,
250 updates to 6500, gate seeds 20260815–17, stage distribution and existing
transition settings. The starting checkpoint has no relief controller and may
bootstrap this explicit treatment. Use a clean committed snapshot plus the
recorded CPU seed overlay and smoke64/5 before the budget. Verify actual
weighted action-rate costs in the live environment. Compare both existing
250-update controls, retained stage/final/CPU metrics and the seed-20260916
full-push rollout. Failure to preserve lateral recovery or yaw learning rejects
this treatment as sufficient. Do not extend its budget without a changed
hypothesis. Fresh held-out seeds, multi-training-seed proof and CPU transfer
remain required even if this diagnostic comparison improves.

Completed scores (zero/forward/lateral/yaw/left/right): gate cohort
.975/.718/.802/.000/.749/.719; native final diagnostic seed 20260915
.949/.868/.812/.000/.649/.788; CPU .973/.859/.789/.000/.838/.841.
Native/CPU yaw mean rates are .103/.046 rad/s for a .8 command: under-speed.
The full-push stage case at seed 20260916 survives with lateral .796, versus
global relief .298 and no-relief .790. All six recorded reset fields and the
initial raw actor observation match the no-relief control. Six native/ONNX
action comparisons pass (maximum error 6.56e-7). This rejects pure-yaw-only
relief as a sufficient treatment; do not extend it unchanged.

### Deterministic startup and consolidation comparison

The matched training windows have nearly equal command fractions. Learned
action std means differ: no-relief .185, global relief .280, pure-yaw relief
.197. The distribution uses state-independent per-action std, so local reward
relief does not isolate exploration by command. This is a mechanism clue,
not proof that a different exploration schedule will repair the policy.

Fixed-actor diagnostic:
`/tmp/microduck-yaw-exploration-dependence-v2/analysis-summary.json`.
Six 64-env rollouts compare deterministic inference, learned Gaussian action
noise for the first second, and continuous noise. All eleven recorded initial
state/observation fields match, all six trace hashes pass. For pure-yaw relief,
45/64 deterministic rollouts end near idle; a one-second pulse reduces this to
17/64 and continuous noise to 1/64. Mean final-two-second rates are .202/.609/
.805 rad/s. The global actor's corresponding means are .819/.845/.816. These
are diagnostic tail rates, not capability passes; no deployment noise is proposed.
One pulse rollout falls. Auto-reset keeps other batch members running but failed
members remain excluded. Batched Warp contact trajectories are not bitwise
replays: initially tiny differences grow even with equal noise samples.

Hypothesis: reducing the entropy incentive during late consolidation makes PPO
optimize behavior closer to deterministic inference and can retain the global
actor's acquired yaw without its late recovery/transfer regressions. Test the
standard `--agent.algorithm.entropy-coef 0.0` against a fresh .01 control,
starting from the exact global-relief 6500 known-good adaptive snapshot, not the
final 6750-budget filename. Use seed 17, 4096 envs, 250 updates, inherited
global relief, unchanged gate cohort/distribution and final/CPU diagnostics.
The launch wrapper must record the actual argv, entropy override and source
hash, run smoke64/5 first, and verify saved `params/agent.yaml`. No automatic
entropy controller or default change is justified before behavioral proof.
Improvement must be measured on the retained actor and accompanied by actual
std reduction; failure rejects this bounded consolidation treatment. Both arms
completed at `/tmp/microduck-adaptive-entropy-consolidation-s17-v2`
(session `31003` exited 0), with identical resume boundaries and immutable source.
Both smoke64/5 checks and saved entropy settings were verified. The result is
the retained improvement described above. The old uninterrupted .01 extension
is contextual evidence, not the matched control. Full
six-bucket mastery, fresh held-out seeds and all remaining acceptance still apply.

## Goal

Build an automated MJLab training system that can adjust bounded difficulty
parameters from measured policy capability and produce a policy that matches or
beats the canonical fixed-schedule policy on a held-out final distribution.

The system will use three layers with distinct responsibilities:

1. The untouched canonical task remains the reproducible baseline and final
   fine-tuning reference.
2. A deterministic adaptive runner owns evaluation cadence, stage transitions,
   checkpoint state, and rollback during a run.
3. An optional Codex advisor analyzes completed windows and proposes the next
   bounded experiment. It cannot directly mutate arbitrary training code or
   bypass controller gates.

The evaluator may run in-process, as a subprocess, or as a separate cluster
job. The adaptive runner remains the authoritative owner of the training and
curriculum state.

## Invariants

- Keep the canonical Velocity task and fixed schedule unchanged.
- Preserve the 61D actor observation ABI, 14D action ABI, command semantics,
  BAM M6 actuator model, reward signs, termination behavior, and mandatory
  normalizer-baked export path.
- Keep PPO architecture, optimizer, rollout length, and normalization fixed
  while testing curriculum behavior.
- Do not introduce action filtering, runtime adaptation, or M4/M6 interpolation.
- Adaptive changes use an explicit, checkpointed allowlist. The user's sustained
  execution authorization also covers necessary reward shaping, regularization
  pacing and evaluation-semantic repairs in the adaptive recipe, after an
  evidence-backed experiment contract. Canonical rewards remain unchanged.
- Every stage change is bounded, logged, reproducible, and reversible.
- Final claims require matched total environment steps, multiple seeds, and a
  fixed held-out battery. Training reward alone is not an acceptance signal.
- A held-out seed is evidence only when it changes a consumed reset, DR, or
  perturbation stream. Distinct labels on deterministic traces do not count as
  independent evaluation.
- A deployment-quality claim must first pass a native MJLab/BAM evaluator. The
  CPU MuJoCo/ONNX rehearsal is a separate transfer check and cannot diagnose
  native training quality by itself.

## Target architecture

```text
AdaptiveMicroduckOnPolicyRunner
  ├─ PPO rollout and update
  ├─ periodic checkpoint (policy + optimizer + RNG + gate state)
  ├─ FrozenCapabilityEvaluator interface
  ├─ CapabilityGate decision
  ├─ live EventManager stage application
  ├─ bounded command-bucket exposure feedback via live CommandManager
  ├─ explicit completed-update/result manifest for launch and resume
  └─ rollback to last-known-good checkpoint

FrozenCapabilityEvaluator
  ├─ fixed seeds and command buckets
  ├─ final-distribution anchor cases
  ├─ continuous tracking/survival/upright metrics
  └─ versioned JSON report

CodexAdvisor (optional outer loop)
  ├─ consumes immutable run manifest and reports
  ├─ returns a schema-validated proposal
  └─ chooses among bounded experiments; never bypasses runner safety rules
```

The shell and Executor scripts become thin launchers. They may submit jobs,
stage artifacts, and retry failed infrastructure, but they must not be the
source of truth for checkpoint selection, curriculum state, or rollback.

## Phase 0: Make the capability signal useful

### Work

- Replace the battery's binary pass score with continuous per-bucket values:
  command tracking error/ratio, survival fraction, episode length, root height,
  tilt percentile, angular tracking for yaw and turns, zero-command drift,
  action magnitude/rate, joint-limit proximity, and actuator saturation where
  available.
- Keep the six required buckets: `zero`, `forward`, `lateral`, `yaw`,
  `turn-left`, `turn-right`.
- Add fixed nominal seeds and a fixed final-distribution anchor set.
- Define a versioned aggregation schema: per-bucket gates plus lower-tail
  aggregate; do not allow a mean to hide a failed bucket.
- Fix axis isolation so `all_static`, `com`, `head_com`, and `composed` each
  construct exactly the intended adaptive axes. A CoM-only run must never
  advance head-CoM.
- Record the evaluation task/config/source SHA, checkpoint, seed, and metric
  schema in every report.

### Tests and evidence

- CPU tests for metric normalization, zero/yaw/turn-specific measures, missing
  or non-finite values, and per-axis isolation.
- A known good and known bad synthetic trace must produce different scores.
- Re-run the existing ONNX battery and confirm scores no longer saturate solely
  because the robot remains upright.

### Gate

Do not enable autonomous transitions until the battery distinguishes at least
one deliberately degraded policy or rollout from the known-good policy on a
continuous metric.

## Phase 1: Move control authority into the runner

### Work

- Add a typed evaluator protocol returning a report for a specific policy
  checkpoint and curriculum state.
- Extend `AdaptiveMicroduckOnPolicyRunner.learn()` with a configured evaluation
  interval. Keep normal PPO rollout/update behavior unchanged between windows.
- Snapshot policy, optimizer, iteration, RNG state, and adaptive state before
  evaluation. Use an explicit checkpoint path instead of globbing for the latest
  file.
- Feed the evaluator report into `CapabilityGate` and apply one legal stage
  transition through `EventManager.get_term_cfg(...)`.
- Store stage values, gate counters, best metrics, transition trace, evaluator
  report hash, and last-known-good checkpoint in the same checkpoint metadata.
- Make rollback restore the complete checkpoint and reapply all live manager
  terms. Record rollback as a first-class trace event.
- Keep the evaluator CLI as a standalone entry point for audit and cluster
  execution. A subprocess adapter may be used if evaluator simulation cannot
  share the training process safely.
- Move stage-file output to an audit artifact. It must not be the primary state
  used to resume training.

### Tests and evidence

- Runner unit tests with a fake evaluator: hold, advance, preservation failure,
  rollback, and resume.
- Save/load test that compares transition traces and live event ranges after
  restoration.
- 64-env, 5-iteration smoke using the adaptive task; verify 61D/14D and finite
  rewards.
- Failure injection test where the evaluator returns a failed bucket and the
  runner stays at or returns to the last good stage.

### Gate

The same checkpoint plus the same evaluator report must reproduce the same
decision and transition trace without relying on shell filename conventions.

## Phase 2A: Repair the adaptive experiment and establish a usable-policy signal

The r4 campaign is retained as historical evidence, but it does not satisfy the
experiment contract: the adaptive factory removed every canonical curriculum
term, the battery seed did not affect the executed trace, and the final battery
used XML position actuators while training used BAM M6. This phase resolves
those confounders before spending another matched-budget campaign.

### Work

- Change the adaptive factory so it removes only the canonical schedules for
  the axes owned by the adaptive controller (`com_range` and/or
  `head_com_range`). Preserve standing, action-rate, command, pose, and other
  canonical curricula exactly. Do not edit the canonical velocity factory.
- Add a native evaluator that runs the checkpoint in the MJLab environment with
  the same BAM M6 actuator, observation normalizer, command ABI, reset path,
  and six bucket definitions used for training. Keep the existing CPU
  MuJoCo/ONNX battery as a separately labelled deployment rehearsal.
- Make evaluation seeds affect an actual consumed source of variation. Use a
  fixed, recorded reset/DR/perturbation manifest so gate and held-out sets are
  disjoint and reproducible. Add a test that two seed sets produce different
  traces while rerunning one seed reproduces the same trace.
- Calibrate the curriculum thresholds, EMA, dwell, and preservation tolerance
  from native reports of canonical checkpoints. Keep the final product gate
  unchanged unless a written calibration report shows that its units or
  semantics are wrong; do not tune thresholds to rescue a failed run.
- Separate two decisions in every report: (a) whether a policy is usable under
  the deployment gate, and (b) whether adaptive matches or beats fixed. A
  usable adaptive policy remains valuable even when it does not win the
  research comparison.

### Tests and evidence

- Config test proving each adaptive mode preserves all non-owned canonical
  curriculum terms and owns only its declared axes.
- Native-vs-export observation/action parity test for one checkpoint, including
  the baked normalizer and 61D/14D ABI.
- Seed-consumption test proving gate and held-out seeds alter a consumed trace;
  same-seed replay remains byte-for-byte deterministic.
- One-checkpoint six-bucket report from native MJLab/BAM and the CPU rehearsal,
  with per-bucket raw metrics and an explicit transfer comparison.
- Single-seed checkpoint ladder at 500/1000/2000/4000 iterations for fixed and
  all-static, recording native capability, CPU rehearsal capability, reward
  terms, episode length, and videos/traces where available.

### Gate

Do not submit another 15-job campaign until the native evaluator has a valid
seed-consumption proof and the checkpoint ladder identifies whether the failure
is native learning, export/observation parity, or CPU transfer. The sensor-reset
run supplies seed-consumption evidence; the complete paired comparison now
identifies foot/plane contact generation as the main paired CPU regression.
Native mastery and the original product rehearsal still fail. Resolve those
behavioral gates before another full campaign. A
policy is **usable** only when its native six-bucket report passes the product
gate; adaptive superiority is a separate later claim.

## Phase 2: Establish the deterministic adaptive baseline

Phase 2 starts only after Phase 2A passes. The r4 matrix and artifacts remain
historical and must not be reused as proof of a valid held-out comparison.

### Work

- Calibrate thresholds, EMA, dwell, and preservation tolerance from measured
  canonical checkpoint distributions, rather than treating current values as
  universal constants.
- Run the following branches with identical PPO settings, total environment
  steps, environment count, and seeds:
  - canonical fixed schedule;
  - all-static initial adaptive slice;
  - CoM-only;
  - head-CoM-only;
  - composed best two-axis controller.
- Keep standing and action-rate diagnostics separate until their own
  controllers and metrics are validated.
- Use at least three training seeds for the final comparison, plus one fixed
  held-out battery seed set not used by the gate.
- Log learning curves, transition timing, per-bucket capability, anchor
  preservation, and rollback count.
- Finish adaptive acquisition by freezing canonical final ranges and run a
  canonical-distribution fine-tuning segment.
- Evaluate the final adaptive checkpoint with both the native product gate and
  the CPU deployment rehearsal, and report the two verdicts separately.

### Gate

Adaptive is accepted only if it reaches the canonical final distribution, has
reproducible transitions, and matches or improves the canonical held-out
battery without nominal or zero-command regression. Otherwise retain the fixed
schedule and record adaptive as negative or inconclusive evidence. A policy can
still be marked usable when it passes the product gate even if this comparison
gate is not won.

## Phase 3: Add the constrained Codex advisor

This phase is optional and starts only after Phase 2 produces informative
metrics and a reliable deterministic runner.

### Work

- Define an immutable run manifest containing source SHA, task ID, config hash,
  PPO settings, seed, checkpoint, curriculum state, battery schema, and all
  reports.
- Define a strict proposal schema with actions such as `hold`, `replicate`,
  `advance_allowed_axis`, `rollback`, `open_diagnostic`, and `stop_negative`.
- Allow only bounded values: approved axis, approved stage delta, maximum
  budget, maximum number of new seeds, and no arbitrary reward/termination/ABI
  changes.
- Require evidence references and a reason code for every proposal.
- Validate proposals deterministically before execution. Invalid or ambiguous
  proposals become `hold` or `replicate`, never an arbitrary code change.
- Run Codex at experiment-window boundaries or between jobs, not inside PPO
  rollout steps and not as the owner of live checkpoint state.
- Compare advisor-selected experiments against a deterministic scheduling
  baseline using the same experiment budget.

### Gate

The advisor is useful only if it improves experiment selection or time-to-good
policy without reducing reproducibility, violating invariants, or increasing
the false-transition rate. It is not required for the adaptive curriculum to
be considered successful.

## Phase 4: Consider stronger active teachers

Only consider ALP-GMM, PLR, DORAEMON-style constrained distribution expansion,
or SimOpt after the deterministic baseline is stable and the failure mode
requires more than a small number of bounded axes.

Each candidate must specify its state, sampling distribution, safety bound,
rollback behavior, and comparison budget before implementation. It must be
compared against the deterministic gate, not introduced as an unqualified
replacement. Real-robot data would be required before using SimOpt for BAM or
other sim-to-real parameter fitting.

## Executor experiment contract

- Use immutable JuiceFS source snapshots and versioned output prefixes.
- Keep one job manifest per branch and seed; never infer provenance from a
  mutable output directory.
- Run smoke64/5 before each new long-run source snapshot.
- Use the existing Executor/CloudML queue only for jobs whose YAML records task,
  source SHA, image, queue, seed, budget, evaluator schema, and output prefix.
- Treat infrastructure failures separately from policy failures.
- Download and verify reports before updating the active status document.

## Deliverables

- `FrozenCapabilityEvaluator` protocol and continuous battery schema.
- Runner-owned adaptive evaluation, checkpoint, rollback, and resume behavior.
- Tests for metrics, axis isolation, deterministic trace replay, and rollback.
- Matched-budget multi-seed comparison report.
- Optional validated Codex advisor schema, validator, and audit log.
- Final canonical fine-tuned ONNX, exported through `scripts/export.py`, followed
  by `scripts/infer_policy.py` deployment rehearsal.
- Updated proposal and active status record with measured results and explicit
  negative evidence where criteria are not met.

## Decision rule

The project does not need to adopt the most sophisticated method. The chosen
system is the smallest one that reliably produces a suitable policy under the
fixed held-out battery and preserves the repository's sim-to-real contracts.
The fixed schedule remains the production fallback until that result is shown.

## Executable planning loop output

The initial document is a roadmap, not an executor handoff. The planning loop
produced the two bounded, reviewable implementation plans below. They resolve
the report, axis, evaluator, checkpoint, rollback, and verification contracts
before any matched-budget compute is requested.

1. [`00-capability-signal-PLAN.md`](mjlab-adaptive-curriculum-v2/00-capability-signal-PLAN.md)
   defines the continuous capability report and end-to-end axis isolation.
2. [`01-runner-control-PLAN.md`](mjlab-adaptive-curriculum-v2/01-runner-control-PLAN.md)
   makes the adaptive runner the state authority after Phase 0 passes.
3. [`02a-diagnostic-repair-PLAN.md`](mjlab-adaptive-curriculum-v2/02a-diagnostic-repair-PLAN.md)
   repairs curriculum ownership, proves seed consumption, and establishes the
   native MJLab/BAM usability signal before new campaign compute.

Phase 2A (experiment repair and native usability diagnosis) remains the next
active work item. The early-reset arm now supplies a retained native-acquisition
signal. Scale, sensor-refresh and CPU BAM probes failed joint improvement;
complete-model/physics/timing comparison and contact replacement now establish
the paired CPU contact-generation cause. Native mastery is still below .80,
and the original product rehearsal remains unvalidated. Resolve the behavioral
gates before another multi-seed campaign. Phase 2 (matched-budget multi-seed experiments) follows
only after its gate passes. Phase 3 (constrained Codex advisor) and Phase 4
(stronger active teachers) remain parked until the deterministic baseline
produces informative, native-evaluated evidence.
