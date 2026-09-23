# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
User intent: sustained necessary changes, training and checks through
`intuitive-flow`; the status request does not cancel execution.
Project-status writer: not assigned; project-status delta: none.

## Current verdict

Phase 0/1 automation works. V5 is the current automatically retained candidate:
one final-distribution native diagnostic passes all six buckets, but the native
three-seed gate cohort and CPU deployment rehearsal still fail all-six mastery.
Both adaptive CoM axes remain stage 0, ±3 mm. Scores are capability, not success
rates. No from-scratch or multi-training-seed autonomy claim is justified.

V5 and the matched yaw-precision comparison are complete. The latter is negative;
neither arm should be extended unchanged. The current proof is the existing
from-scratch seed-23 run below. The latest user explicitly resumed after an API
interruption; the host goal's stale `blocked` label is not an external blocker.

## Live seed-23 proof

Poll session `85159` with `write_stdin`; do not restart on observation timeout.
Root: `/tmp/microduck-adaptive-from-scratch-s23-v4`. Contract/config/commands and
logs are in that root. Wrapper: `/tmp/run_microduck_from_scratch_s23_v4.py`;
driver: `/tmp/train_microduck_from_scratch_s23_v4.py`. Fresh smoke64/5 passed;
training is seed23/4096 envs from canonical initialization, budget 7000 updates,
gate every 250, opt-in entropy consolidation. `campaign-config.json` confirms
`stage` distribution, cohort seeds 20260815–17. Final diagnostic 20260915 is reused.

The 500-update cohort scores are .900/.781/.750/.606/.500/.645 (zero/forward/
lateral/yaw/left/right), up from a minimum of zero at 250. All evaluation cases
survive; tracking limits the weak turn buckets. Consolidation is still waiting,
entropy .01, both CoM axes stage0 ±.003. The teacher has shifted yaw exposure to
.1675; temporary action-rate relief is −.2. This is learning progress, not mastery.

The driver has an inherited **post-return** assertion allowing only 6750/7000
completed updates. A valid earlier controller stop will fail that assertion
after production `learn()` saves its final checkpoint and `training-result.json`.
At exit validate those artifacts and the actual controller deadline; if needed,
complete the launcher's final native/CPU evaluations separately, preserving the
wrapper failure. Do not retrain or modify immutable running scripts to hide it.

The wrapper imports campaign code from the V5 snapshot but training imports the
editable worktree. `source-equality-audit.json` verifies 222 relevant source/
script files equal the snapshot; preserve that equality until training ends and
repeat the audit at closeout. Do not claim isolated imports. V1–v3 starts failed
on inherited instrumentation assertions before any PPO update; their separate
`invalid-driver.json` records remain diagnostic artifacts, not learning evidence.

## V5 evidence and behavior

The runner automatically selected entropy .01→0 from current native capability,
then shifted one bounded command-exposure slice toward the weakest bucket, yaw.
The change reached the live command manager: yaw probability .0890→.1368 during
training, with zero/nominal anchors and directional floors preserved. The normal
end-of-window teacher update leaves .1726 in the retained checkpoint.

The weakest cohort score improved .6694→.7320 (+.0626), while the existing gate
preserved mastered capabilities. The controller retained the candidate and
stopped. Retained actor equals candidate/known-good and differs from baseline;
mean action std fell .27970→.10010. Entropy remains zero. This demonstrates one
automatic improvement from an existing actor, not a reliable complete recipe.

| Capability scores | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Starting stage cohort, worst of three | .938 | .735 | .796 | .669 | .830 | .812 |
| V5 stage cohort, worst of three | .962 | .754 | .792 | .732 | .819 | .819 |
| V5 final native, diagnostic seed 20260915 | .937 | .886 | .810 | .851 | .802 | .811 |
| V5 CPU v4, same diagnostic battery seed | .981 | .902 | .851 | .668 | .754 | .766 |

Native/CPU yaw rates average .840/1.004 rad/s for a .8 command. CPU cases remain
upright; their tracking scores fail. The final-native left margin is only .0025,
and one gate seed passes all six while the other two do not. The single final
native pass cannot establish robustness or justify product acceptance.

Summary: `/tmp/microduck-auto-consolidation-s17-v5/experiment-summary.json`.
Audit: `/tmp/microduck-auto-consolidation-s17-v5/verification.json`.
Checkpoint root:
`/tmp/microduck-auto-consolidation-s17-v5/training/logs/rsl_rl/matched_lateral-drive/2026-09-23_09-49-03_matched-lateral-drive-s17/`.
Retained `model_6749.pt` SHA:
`311265f7d9258c6de29845675d1b68bcf499e247ee80d43e6749d59064167aea`.
Source: `/tmp/microduck-auto-consolidation-source-d503d63-v5`, 682 files; manifest
SHA `8aa8f6a42c8f46ae38023ed33687a1a31657a7e4136b8a342e9a903128a60ea6`.
Audit verifies source, 12 reports, 48 unique traces and checkpoint hashes.
Baseline actor/critic/optimizer equal the original 6500 start. Training seed 17,
4096 envs, gate seeds 20260815–17; diagnostic seed 20260915 is already reused.

## Controller contract and verified implementation

Experimental opt-in `--entropy-consolidation`; no default promotion. Trigger:
all native scores ≥ .60, zero ≥ .80, some bucket < .80, existing entropy > 0.
Save the baseline before changing entropy or exposure. Try one evaluation
window; retain only with gate preservation and weakest-score gain ≥ .05 (or
all six ≥ .80). Otherwise restore baseline trainer, entropy and teacher. Either
outcome stops and runs final native/CPU evaluation. Resume preserves the original
deadline and actual update budget; a terminal attempt cannot extend unchanged.

176 focused tests pass. V5 fresh smoke64/5, live manager/controller agreement,
finite 61D/14D, entropy, relief −.2 and early-stop assertions pass. Focused Ruff
and `git diff --check` pass. Worktree code/tests match the verified V5 snapshot.
Human docs checked: README and specialist reproducibility record remain valid;
experimental evidence belongs here and in the canonical plan. No Serena memory
entries were present to update.

V3 proved automatic rejection/rollback: candidate right .812→.726, weakest gain
only .033; original actor/teacher/entropy restored. Evidence:
`/tmp/microduck-auto-consolidation-s17-v3/verification.json`. V3/V5 bootstrap
scores match exactly. The changed focus treatment helped this V5 run, but one
training seed does not prove the earlier stale-focus hypothesis causal.
V1/v2 failed before PPO; V4's missing live-manager update made its at-least-11
updates invalid treatment evidence. Preserve all artifacts; do not restart them.

## Remaining work and next decision

Blocker fingerprint: `reset_conditioned_acquisition_and_cpu_yaw_rate_transfer`.
Classification: V5 preserves acquired behavior on the known push/sensor seeds;
remaining failures are tracking precision, with no falls in these diagnostics.
`/tmp/microduck-v5-robustness-diagnostics/summary.json` verifies 12 native/ONNX
cases and exact six-field reset plus initial-observation equality against the
manual reference. Push lateral .739→.781; sensor lateral .748→.795 and right
.773→.830; sensor forward stays .762 and yaw .852. This does not prove fresh-seed
robustness. No external blocker exists.

CPU recordings: `/tmp/microduck-v5-cpu-visual/summary.json`. All arrays from the
three rendered cases exactly match the original product battery. Reviewed 18
sampled video frames: upright foot-supported turning without body/head contacts;
raw yaw oscillation and sustained speed errors remain. Videos are recorded,
but sampled-frame review is not a full temporal/hardware acceptance.

The matched yaw-precision comparison is negative. Both arms resumed the exact V5
actor/critic/optimizer/RNG and teacher for 250 updates with entropy 0 and relief
−.2; only the treatment changed signed-EMA yaw L1 weight 1.0→2.0. Both passed
smoke64/5, live 61D/14D and penalty checks. The control's native cohort minimum
gain was +.043 and CPU minimum −.149; yaw2 was +.041 and −.123. Neither meets the
joint +.05 rule; control final native left fell to .726 and CPU yaw to .519, while
yaw2 CPU yaw/right were .579/.545. No unchanged extension is justified.

Summary: `/tmp/microduck-yaw-precision-s17-v2/experiment-summary.json`.
Audit: `/tmp/microduck-yaw-precision-s17-v2/verification.json`.
The 16 report files, 60 unique traces and both immutable source manifests verify. The
failed preparation v1 stopped before training and is retained separately. This
rejects reward-weight-only precision repair; it does not reject adaptive training
or the V5 candidate. Full .80 acceptance and multi-seed autonomy remain separate.

The earlier manual 6750 reference and negative −.2/−.4 continuations are indexed
in the canonical plan. Transfer diagnostics:
`/tmp/microduck-transfer-diagnostics-6750/summary.json`. Clean native inputs or
nominal calibration help one reused seed; CPU delay/integrator/actuator changes
did not resolve transfer. Product defaults and installed dependencies stay as
specified; the temporary CPU BAM friction-DOF diagnosis is not a shipped fix.

Required acceptance: all-six native mastery at final ranges, fresh held-out
seeds, rollout/video inspection, normalized ONNX/CPU rehearsal, the same procedure
across training seeds 17/23/47, canonical final-range fine-tuning and matched-
budget fixed/axis comparisons. Reused diagnostic seeds are not held-out proof.

Preserve canonical Velocity, BAM M6, unfiltered actions, 61D/14D, reward signs,
.80 gates, zero/nominal anchors and the existing CPU seed patch in
`scripts/run_specialist_action_battery.py`. IsaacLab, generated files, `uv.lock`
and unrelated processes are outside scope. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Use unique `/tmp` directories;
worktree training logs are unwritable.
