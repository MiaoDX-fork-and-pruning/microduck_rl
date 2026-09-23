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

V5 completed; session `10932` and audit session `40226` both exited 0. No owned
training process remains running. Requested 6500→7000; the controller consumed
only 250 updates and stopped at 6750. This is 24,576,000 new transitions.

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
Classification: V5 can retain an automatic native improvement, but reset/sensor/
push robustness and CPU turning accuracy remain insufficient. There is no
external blocker. Before another training intervention, evaluate this retained
actor on the already diagnosed push/sensor seeds and compare raw tracking traces
with the previous manual-consolidation reference. This decides whether a next
bounded treatment should address native robustness or CPU transfer; it must not
be an unchanged extension of this terminal attempt.

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
