# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
The user authorized sustained necessary changes, training and checks through
`intuitive-flow`; the status request does not cancel execution.
Project-status writer: not assigned; project-status delta: none.

## Current retained policy

Phase 0/1 automation works; Phase 2A usable-policy acquisition remains open.
A matched low-entropy consolidation experiment retains a new **6750 actor**.
Every capability bucket must score **≥ .80**; scores below are not success rates.
Both adaptive CoM axes remain at stage 0 (±3 mm).

| Retained entropy-zero 6750 actor | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, worst of 20260815–17 | .961 | .782 | .783 | .790 | .838 | .777 |
| Final-native diagnostic, 20260915 | .936 | .875 | .812 | .857 | .815 | .797 |
| CPU rehearsal, 20260915 | .966 | .892 | .861 | .531 | .698 | .788 |

Native final has five passing buckets; cohort and CPU acceptance still fail.
CPU yaw mean is 1.104 rad/s for a .8 command, improved from 1.390 but still fast.
Native final mean is .817. No usable-policy, hardware or superiority claim.

Authoritative evidence:
`/tmp/microduck-adaptive-entropy-consolidation-s17-v2/experiment-summary.json`.
Checkpoint:
`/tmp/microduck-adaptive-entropy-consolidation-s17-v2/entropy0-250/training/logs/rsl_rl/matched_lateral-drive/2026-09-23_07-08-07_matched-lateral-drive-s17/model_6749.pt`.
SHA: `95d8fd11b8b7a5acc44669584981f87dbb2b8d8c48460bb6cb51e91bad49be2e`.
All 13 actor-state tensors equal the candidate/known-good snapshot and differ
from the start. The .01 matched control loses left/right preservation and rolls
back to the original 6500 actor. Both consume 250 updates from the exact same
6500 adaptive snapshot, seed 17, 4096 envs. Saved agent configs differ only in
`algorithm.entropy_coef` (.01 vs .0). Both smoke64/5 checks pass. Mean action std:
start .280, rejected control .291, retained treatment .101. Session `31003` ended.

The known full-push case (stage seed 20260916) now survives, lateral .739 vs .298;
tracking recovery is still below mastery. Encoder-sensitive seed 20260919 yaw
improves .146→.864, mean .360→.836 rad/s, with product sensor DR unchanged.
Each comparison matches six recorded reset fields and the initial observation.
Twelve ONNX action-parity cases pass (max error 7.16e-7). Those paths are linked
from the authoritative summary. Previously consumed seeds remain diagnostic.

## Running now and next decision

Blocker: `reset_conditioned_acquisition_and_cpu_yaw_rate_transfer`.
Classification changed: stochastic rollout performance masked deterministic
startup failure; low-entropy consolidation improves both startup and transfer,
while remaining native tracking deficits and CPU speed bias prevent acceptance.

A **single further 250-update window, 6750→7000**, is running in session `43041`.
Poll that handle; do not restart on timeout. Root output:
`/tmp/microduck-adaptive-entropy-continuation-s17-250` (arm `continuation-250`).
Launch wrapper: `/tmp/run_microduck_entropy_continuation.py`.
Training seed 17, 4096 envs, same source and gate seeds/distribution, inherited
global relief and adaptive command teacher (now focused on lateral). The wrapper
passed smoke64/5 with entropy zero verified, records real argv, checks saved
agent entropy, and automatically performs retained native-final and CPU
evaluations. Training was observed at update 6825/7000, action std about .09.

**Entropy zero is an explicit standard PPO CLI override, not inherited from
adaptive checkpoint metadata.** Ordinary default resumes would restore .01.
Every continuation must pass `--agent.algorithm.entropy-coef 0.0` and verify the
saved config. Do not claim an automatic entropy controller exists or change the
default before behavioral proof.

Success for this bounded follow-up: retain improvement in remaining native
buckets and CPU tracking. Rollback or stagnant/worse native and CPU results
stops unchanged budget extension; investigate transfer mismatch next. When it
ends, verify `model_6999.eval.pt` versus the retained checkpoint/known-good actor,
consumed budget, source hashes, actual angular rates and all report/trace hashes.
The previous summarizer hardcodes 6749 and is not the continuation summarizer.
Fresh held-out seeds and the full acceptance below remain mandatory.

## Evidence behind the current treatment

Pure-yaw-only smoothing relief was rejected as sufficient: after 6250→6500,
yaw scores are zero in all three gate members, native final and CPU. Full push
survives but yaw learning is lost. Do not promote or extend that treatment.
`/tmp/microduck-adaptive-pure-yaw-s17-250/experiment-summary.json`.

Fixed-actor noise diagnostic:
`/tmp/microduck-yaw-exploration-dependence-v2/analysis-summary.json`.
For that pure-yaw actor, deterministic / first-second noise / continuous noise
produce 45/17/1 idle tails out of 64; mean last-two-second rates .202/.609/.805.
Thirty previously idle cases move after the pulse; one pulse case falls. Eleven
initial fields match and all six trace hashes pass. These are diagnostic tail
rates, not capability passes. Batched Warp contact paths are not bitwise replays,
and auto-resets can change later RNG after a first failure. No deployment noise.
Plot: `.../yaw-noise-dependence.png` (PDF also available).

Original global-relief evidence:
`/tmp/microduck-adaptive-action-rate-s17-500/experiment-summary.json`.
That campaign consumed 6250→6750 but retained 6500 after rollback. Its 10-seed
stage diagnostic passed no all-six case; yaw/lateral each passed 3/10. Sensor
and push ablations located independent yaw-calibration and lateral-recovery
failures; see the canonical plan for those artifact links. Do not repeat old
exposure-only, sensor-resampling-only or unchanged global-relief budgets.

## Implementation and proof boundaries

Source: `/tmp/microduck-adaptive-pure-yaw-source-610fbc5`, `git archive 610fbc5`
plus the existing CPU seed overlay. All 680 files are verified unchanged.
Manifest file SHA: `5c2e75579942e6f46d1e8c94a6d6fb6f72de0c3eca11049553e54141948d5d58`.
Source label: `610fbc5-cpu-seed-overlay-5ae829076760`.
`610fbc5` added opt-in checkpointed pure-yaw relief; default scope remains global.
98 distinct focused tests, focused Ruff/diff checks, real smoke64/5 and separate
live weighted-cost/61D/14D/unfiltered-action proof passed before these runs.
This turn changes only task docs and explicit temporary experiment wrappers.
Earlier infrastructure: 269 relevant tests and fresh locked non-editable install/
config/model compile pass. Fresh-env GPU and remote execution remain unproven.
Full `mdp.py` Ruff has 17 pre-existing findings. CPU link-frame fix is `931e3c9`;
old principal-inertia-frame scores are superseded.

## Remaining acceptance and boundaries

Required: six-bucket retained native mastery at final ranges, fresh held-out
seeds, rollout/video inspection, normalized ONNX and CPU rehearsal, autonomous
training seeds 17/23/47, canonical final-range fine-tuning and matched-budget
fixed/axis comparisons. Goal is active; no external blocker prevents progress.

Preserve canonical Velocity, 61D/14D, BAM M6, unfiltered actions, reward signs,
.80 gates, exact-zero/nominal anchors and the existing CPU seed patch in
`scripts/run_specialist_action_battery.py`. IsaacLab, `uv.lock`, generated files
and unrelated processes are outside scope. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Use the local venv and unique
`/tmp` work directories; worktree `logs/rsl_rl` is unwritable.
