# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
The user authorized sustained necessary changes, training and checks through
`intuitive-flow`; the status request does not cancel execution.
Project-status writer: not assigned; project-status delta: none.

## Latest result and best diagnostic reference

Phase 0/1 automation works; Phase 2A usable-policy acquisition remains open.
The latest **7000 actor** was retained by the native gate, but regressed in CPU
right-turn stability. **6750 remains the better product diagnostic reference**.
No training or diagnostic process is running. Do not extend the same treatment.
Every capability bucket must score **≥ .80**; scores below are not success rates.
Both adaptive CoM axes remain at stage 0 (±3 mm).

| Retained entropy-zero 6750 actor | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, worst of 20260815–17 | .961 | .782 | .783 | .790 | .838 | .777 |
| Final-native diagnostic, 20260915 | .936 | .875 | .812 | .857 | .815 | .797 |
| CPU v4 rehearsal, 20260915 | .966 | .892 | .861 | .531 | .700 | .788 |

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

## Completed continuation and next decision

Blocker: `reset_conditioned_acquisition_and_cpu_yaw_rate_transfer`.
Classification: low-entropy consolidation improves deterministic acquisition,
but native tracking/DR robustness and CPU actuator-path transfer remain open.
Last decision delta: the extra window regresses product behavior; current-limit
alignment and a paired actuator-delay probe do not resolve yaw overspeed.

The single authorized follow-up, **6750→7000**, completed in session `43041`
after smoke64/5. Seed 17, 4096 envs, explicit entropy zero, unchanged training
source and gate distribution. All 13 retained actor tensors equal the candidate
and known-good snapshot; no rollback. Mean action std fell .101→.072.

| Retained 7000 actor | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, worst of 20260815–17 | .981 | .819 | .776 | .746 | .843 | .776 |
| Final-native diagnostic, 20260915 | .937 | .877 | .797 | .866 | .764 | .797 |
| CPU v4 rehearsal, 20260915 | .975 | .892 | .851 | .761 | .699 | .000 |

CPU right turns and survives: mean -.990 rad/s; samplewise angular error .615
exceeds the .6 cap. Its .201 filtered error alone would score .665. Do not call
it idle. Native/CPU yaw means are .824/.946 rad/s for a .8 command.
Authoritative training summary:
`/tmp/microduck-adaptive-entropy-continuation-s17-250/experiment-summary.json`.
Retained checkpoint SHA:
`94c29a191bea9bc7bdfdc7e5f93fcd68fc621845d188f56b02229219a548ee86`.

**Entropy zero is an explicit standard PPO CLI override, not inherited from
adaptive checkpoint metadata.** Ordinary default resumes would restore .01.
Every continuation must pass `--agent.algorithm.entropy-coef 0.0` and verify the
saved config. Do not claim an automatic entropy controller exists or change the
default before behavioral proof.

Next: isolate BAM versus XML position-PD response with a fixed actor and matched
physical state, then choose a bounded adaptive-training change from that evidence.
Do not change runtime delay or relax acceptance. Fresh held-out seeds and the
full acceptance below remain mandatory.

## CPU harness and delay evidence

`918bfb6` makes the battery share `infer_policy.py`'s 1.75 A default current
limit (M6 kt × current = ±.64052362 Nm) and records the XML position-PD profile.
The old battery used XML ±.96 Nm. CPU is a separate rehearsal, not native BAM.
Full normalized-ONNX v4 rechecks for both actors complete at
`/tmp/microduck-cpu-v4-recheck-6750-7000/summary.json`. All 12 cases remain finite
61D/14D with identical reset perturbations and initial observations. The 6750
traces exactly reproduce the earlier manually aligned runtime diagnostic.
Yaw and the 7000 right-turn failure are unchanged; both actors fail acceptance.

Fixed-6750 delay comparison:
`/tmp/microduck-actuator-delay-6750/analysis-summary.json`.
Native 3–6 lag units are **5 ms physics steps (15–30 ms)**; the optional CPU
inference buffer counts 20 ms control steps. CPU product default stays zero.
Native nominal/zero-delay yaw: score .857/.861, rate .817/.811 rad/s. CPU
0/15/20/30 ms yaw: score .531/.345/.291/.223, rate 1.104/1.229/1.271/1.313.
Delay mismatch is not a sufficient cause or repair for this tested overspeed.
Native zero-delay still consumes the original buffer/RNG; recorded physical and
sensor resets, first observation and sampled lag sequences match. Native nominal
exactly reproduces previous final traces. All 36 trace hashes, native/ONNX parity
and 681 source files pass. One reused diagnostic seed; no new acceptance claim.

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

Training source: `/tmp/microduck-adaptive-pure-yaw-source-610fbc5`, `git archive 610fbc5`
plus the existing CPU seed overlay. All 680 files are verified unchanged.
Manifest file SHA: `5c2e75579942e6f46d1e8c94a6d6fb6f72de0c3eca11049553e54141948d5d58`.
Source label: `610fbc5-cpu-seed-overlay-5ae829076760`.
`610fbc5` added opt-in checkpointed pure-yaw relief; default scope remains global.
98 distinct focused tests, focused Ruff/diff checks, real smoke64/5 and separate
live weighted-cost/61D/14D/unfiltered-action proof passed before these runs.
Current CPU/delay diagnostic source:
`/tmp/microduck-current-limit-source-918bfb6`, 681 verified files, manifest SHA
`a22c92f5a416f33e954e49ecc47055a0345e177e71ed870ba9267ae85e6be791`.
No training has run on this source; a new long run requires smoke64/5.
The current-limit fix has 30 passing focused tests; two failures require absent
pre-existing specialist artifacts and reproduce on pre-fix `610fbc5`.
Focused Ruff/diff pass; `infer_policy.py` has no new Ruff findings.
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
