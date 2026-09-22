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

Latest transition-acquisition campaign:
`/tmp/microduck-adaptive-transition-s17-5000/campaign-result.json`. It resumed
the cumulative-4500 lateral-drive checkpoint for 500 updates at 4096 envs, with
initial transition probability 0.20 and final-CoM rehearsal 0. The launcher
finished smoke, training, native held-out and CPU/XML transfer. This run was
started from the shared working tree at source SHA `20e31f2`; no source edit
occurred during training, but the run is not an immutable-source experiment.

| Transition campaign evidence (minimum six-bucket score) | Score | Result |
| --- | ---: | --- |
| Candidate, native gate seed 20260815 | 0.73638 | fails lateral, yaw, turn-left, turn-right |
| Candidate, native held-out seed 20260915 | 0.59512 | fails yaw and turn-right |
| Retained policy after rollback, native held-out | 0.46801 | fails yaw and turn-left |
| Candidate, CPU/XML transfer held-out | 0.0 | fails forward, lateral, yaw and both turns |
| Retained policy, CPU/XML transfer held-out | 0.0 | fails forward, lateral, yaw and both turns |

The candidate is better than the retained policy on held-out native evidence,
but the gate still rejects it. The retained checkpoint therefore remains a
rollback artifact, not a usable policy. The transition controller ended at
probability 0.25 after one yaw/turn retention repair; it did not establish
that a zero→yaw lead-in naturally completes under a live policy.

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

The transition candidate is
`/tmp/microduck-adaptive-transition-s17-5000/training/logs/rsl_rl/matched_lateral-drive/2026-09-22_16-40-38_matched-lateral-drive-s17/model_4999.eval.pt`.
`model_4999.pt` is the restored policy. The gate triggered a yaw/turn
preservation failure and returned to the known-good actor from the cumulative
4500 run. Keep candidate learning and retained-policy verdicts separate.

The transition candidate's native held-out yaw score is 0.59512 and
turn-right is 0.75738; all six cases survive six seconds, so the failure is
tracking rather than a crash or nonfinite rollout. The native candidate gate
does clear zero and forward, but lateral and both turn directions remain below
the product threshold. CPU/XML transfer remains a separate failure even when
the native candidate survives. No video or hardware usability is established.

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
- 196 adaptive/config tests pass. The transition checkpoint contract now
  restores saved state exactly, clears live state when explicitly loading a
  legacy checkpoint, and applies a CLI override only after full restore;
  runner tests cover repeated rollback repair. Smoke64/5 passed before
  training. The mandatory export path produced smoke ONNX with maximum
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
serialization, and bounded runner integration. The existing CUDA/MJLab/BAM
probe recorded allocation and reset-isolation data, but manually forced timer
completion; it does not prove natural deadline switching under a live policy.
The 64-env/5-update transition smoke passed with 61D actor, 14D action, and no
NaN termination. The 500-update campaign above is the first long training
evidence for this repair and it remains below the usability gate.

Next action: repair the remaining yaw/turn acquisition and native-to-CPU
transfer gap using a new bounded experiment with immutable source provenance.
Do not change product thresholds or the frozen battery. Formal seeds 17/23/47
remain blocked until one retained checkpoint passes all six native gates, then
passes fresh held-out evidence, video/rollout review, normalized ONNX export and
CPU deployment rehearsal.

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
