# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: status inspection within sustained authorized intuitive-flow
execution. The root goal remains active; this checkpoint is not completion.

## Current result

Automated evaluation, command-exposure feedback, checkpoint/resume and rollback
operate. Both CoM axes remain at stage 0 (±3 mm); native six-capability usability
and CPU transfer are unproven. Formal training seeds 17/23/47 have not started.
No training or evaluation process from this slice remains running.

Latest completed comparison:
`/tmp/microduck-adaptive-com-rehearsal-s17-5000/comparison-summary.json`.
It tests 20% final-CoM rehearsal against the existing no-rehearsal control
`/tmp/microduck-adaptive-evidence-s17-5000-cadence/`. Both start from the same
cumulative-4500 checkpoint and consume 500 updates at 4096 envs, with identical
command-exposure state, gate cadence and evaluation seeds. This is one bounded
comparison, not a replicated causal study.

| Evidence (minimum six-bucket score) | Control | 20% final-CoM | Outcome |
| --- | ---: | ---: | --- |
| Candidate, final profile, gate seed 20260815 | 0.70988 | 0.71551 | both fail 4/6 buckets |
| Candidate, final profile, held-out seed 20260915 | 0.79519 | 0.0 | control passes 5/6; rehearsal passes 3/6 |
| Candidate, initial-profile diagnostic, gate seed | 0.78234 | 0.76219 | neither establishes initial mastery |
| Retained policy after rollback, held-out | 0.42439 | 0.42439 | yaw and turn-left fail |
| Retained policy, CPU/XML actuator rehearsal | 0.0 | 0.0 | transfer fails |

New candidate:
`training/logs/rsl_rl/matched_lateral-drive/2026-09-22_13-54-41_matched-lateral-drive-s17/model_4999.eval.pt`
under the latest comparison directory. `model_4999.pt` is the restored policy.
The gate triggered yaw preservation failure and returned to
`/tmp/microduck-adaptive-evidence-s17-4250-repair/training/logs/rsl_rl/matched_lateral-drive/2026-09-22_12-42-00_matched-lateral-drive-s17/model_4124.eval.adaptive.pt`.
Keep candidate learning and retained-policy verdicts separate.

The new candidate's held-out yaw score of zero is a **tracking stall**, not a
fall or nonfinite rollout: command 0.8 rad/s, actual mean 0.10769 rad/s, with
middle-second means around 0.01–0.03 rad/s. All six cases survive six seconds.
Gate turn-left improves to 0.85588, but forward/lateral/yaw regress. Held-out
lateral reaches 0.80495 while yaw/turn-left/turn-right fail. Identical gate and
held-out reset/DR fields were verified against their control traces in
`paired-reset-probe.json`. No videos or hardware usability are established.

## Proven implementation and decision

- `b2dc088` / `b3db958` / `873db35`: bounded command-exposure repair accumulates
  across automatic policy rollback and restart. Regression tests use distinct
  immutable checkpoints; explicit load/rollback still restore saved state.
- `da51889`: opt-in `--final-com-fraction` (0..0.20) wraps only adaptive-owned
  CoM events, retains stock non-accumulation and full physical recomputation,
  and keeps live stage ranges mutable. Launch/checkpoint metadata persist it;
  ordinary resume inherits it, explicit load restores it, and automatic
  preservation rollback retains the live experiment setting. Native evaluation
  disables rehearsal and freezes the requested distribution.
- 183 adaptive/config tests pass. A native physical probe proves 12/64 final
  environments, other offsets <=3 mm, 20 non-accumulating partial resets,
  untouched-environment isolation and finite 61D/14D steps. Smoke64/5 passed
  before training. The mandatory export path produced smoke ONNX with maximum
  PyTorch/ONNX action error 3.58e-7 over eight input cases.
- The run used a read-only source snapshot of `da51889`; all 675 recorded file
  hashes were checked after execution. 49,152,000 command samples were classified,
  none unclassified, with no NaN terminations or positive checked penalty terms.
  Zero and nominal command pools remain 20% each. Diff checks pass and Ruff
  adds no findings relative to the existing 100 findings in the checked files.

Decision: **do not select 20% final-CoM rehearsal for the default procedure**.
The implementation works, but this bounded setting fails acquisition/retention
acceptance. It neither identifies CoM as the cause nor rules out all rehearsal
schedules. Ordinary fresh runs still default to zero. The latest experimental
checkpoint contains 0.20, so explicitly use `--final-com-fraction 0` if resuming
it for a different mechanism. Do not silently repeat it or lower the product gate.

## Current blocker and selected repair

Blocker fingerprint remains `native_command_conditioned_acquisition_and_retention`.
The paired native diagnostic in
`/tmp/microduck-adaptive-yaw-transition-s17-v4/yaw-transition.json` changed the
classification: `forward→yaw` starts in roughly `0.06–0.28 s`, while
`zero→yaw` commonly waits about `4 s`; matched reset/DR/CoM/friction state was
identical. The failure is command initiation coverage, rather than a pure
steady-state yaw or evaluator-reset issue.

The selected repair is an opt-in transition-acquisition slice. For yaw and
moving-turn buckets, a bounded fraction starts with a short forward command and
restores the original target after `1–2 s`. `TransitionExposure` increases this
fraction by at most `0.05` after a yaw/turn frontier below `0.80`, releases it by
`0.025` after consolidation at `0.88`, and caps it at `0.40`. It is separate
from the six evaluator buckets, keeps the zero/nominal anchors, persists in
checkpoints, survives rollback, and is disabled for native evaluation.

Focused tests now pass for probability/config contracts, pre/post command
restoration, reset `dt=0` timing, partial-reset isolation, exposure state
serialization, and bounded runner integration. A real CUDA/MJLab/BAM probe in
`/tmp/microduck-adaptive-transition-native-probe-s17-v2.json` observed 26/64
initial transitions, zero elapsed time at reset, forward bootstrap after ten
steps, exact target restoration, and no untouched-environment reset changes.
The 64-env/5-update transition smoke also passed with 61D actor, 14D action,
and no NaN termination. No long training budget has been spent on this repair
yet.

Next action: run one matched cumulative-4500→5000 transition branch at initial
probability `0.20`, with final-CoM rehearsal `0`, then compare native gate,
held-out, retained-policy and CPU transfer evidence against the existing
control. Do not change product thresholds or the frozen battery. Formal seeds
17/23/47 remain blocked until this bounded comparison establishes a retained
six-capability policy.

## Remaining gates and no-touch scope

- Establish all six capabilities across native reset/DR conditions, then
  reproduce training seeds 17/23/47 with fresh held-out evidence, rollout/video
  review, normalized ONNX export and CPU deployment rehearsal. Matched-budget
  comparison follows acquisition. No current adaptive-superiority claim.
- Preserve canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward
  signs, signed-EMA metric, product thresholds and the zero anchor.
- Leave unrelated IsaacLab, uv.lock, generated files and other processes alone.
  Fresh-sync portability remains unproven; local runs use the validated venv.
  Keep artifacts under `/tmp`; repo `logs/rsl_rl` is not writable.
- Advisor, stronger teachers, generalist, IsaacLab migration and hardware
  deployment remain parked. Stop on nonfinite training, invalid provenance or
  resource exhaustion; inspect behavioral regression before another budget.

## Verification and launch inventory

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest pytest -q
 tests/test_adaptive_*.py tests/test_mjlab_adaptive_velocity_config.py`.
`scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17
--output <new-dir> --iterations <cumulative-budget> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval <explicit-interval>` runs smoke64/5, preserves
consumed-update accounting and records manifests. `status: evaluated` means
completed evidence, not a passing policy. Use immutable source snapshots.
Optional `--final-com-fraction <0..0.20>` controls rehearsal on owned axes;
omission inherits the checkpoint value (zero for legacy checkpoints).
