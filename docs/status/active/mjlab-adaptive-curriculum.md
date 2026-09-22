# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: status inspection within sustained authorized intuitive-flow
execution. The root goal remains active; this checkpoint is not completion.

## Current result

The runner, checkpoint/resume, native MJLab/BAM evaluator, command-conditioned
reward evidence, rollback and bounded command-exposure repair operate. Both
CoM axes still remain at stage 0 (`0.003 m`); the six-capability product gate
has not been met consistently across native reset/DR seeds. The formal
training-seed matrix `17/23/47` has not started.

The latest completed campaign is
`/tmp/microduck-adaptive-evidence-s17-5000-cadence/`, with a compact audit in
`retention-summary.json`. It continued seed17 for 500 PPO updates at 4096 envs
from cumulative 4500, evaluating once after 500 updates. No training or
battery process from this slice remains running.

| Evidence | Lower-tail score | Outcome |
| --- | ---: | --- |
| 5000-update candidate, gate seed `20260815` | 0.70988 | lateral, yaw, turn-left and turn-right fail |
| Same candidate, held-out seed `20260915` | 0.79519 | 5/6 pass; lateral fails |
| Final retained policy after rollback, held-out seed `20260915` | 0.42439 | yaw and turn-left fail |
| Final retained policy, CPU/XML actuator rehearsal | 0.0 | transfer fails; separate from native learning |

The candidate's lateral signed-EMA error is `0.0245773 m/s`, versus `0.024 m/s`
required for a 0.80 score. Do not relax the product threshold or accept this
single near-pass seed. The gate cohort exposes much larger remaining errors.
The rejected candidate is `model_4999.eval.pt`; `model_4999.pt` is the final
retained policy. Paths and hashes are in the summary and campaign manifests.
All six candidate held-out cases survive their six-second rollouts; videos and
hardware usability have not been established.

The 4250, 4500 and 5000 final policies restore the same known-good checkpoint:
`/tmp/microduck-adaptive-evidence-s17-4250-repair/training/logs/rsl_rl/matched_lateral-drive/2026-09-22_12-42-00_matched-lateral-drive-s17/model_4124.eval.adaptive.pt`.
4250/4500 actor and normalizer tensors were compared and are identical.
A candidate can improve on held-out evidence while failing the gate cohort;
keep candidate and retained-policy verdicts separate.

## Proven controller repair

- `b2dc088`: identify regressed buckets and allocate bounded repair exposure.
- `b3db958`: at automated preservation failure, preserve live teacher exposure
  through the complete policy/gate/RNG rollback, then repair it. Plain
  checkpoint load and explicit rollback still restore saved state exactly.
- `873db35`: regression tests use distinct immutable checkpoint paths, exercise
  continuous execution and restart, and verify policy restoration alongside
  increasing repair counts. Both cases fail with the pre-fix method and pass
  with the fix. The original test reused its rollback target and masked this.

The pre-fix 3750→4250 continuation exposed repair counts resetting to 1.
After the fix, the 4250→4500 continuation preserves counts `2→3→4`, with
turn-left exposure `0.105→0.12375→0.1628125`; the 500-update window increments
it to 5. Actual per-command samples track the new mixture. The latest window
contains 49,152,000 classified samples, zero unclassified samples, and no
positive penalty terms or NaN terminations in the training log. The zero and
nominal anchors remain 20% each.

Proof: 147 adaptive tests passed after the implementation fix; after hardening
the regression test, all 28 production/rollback/resume cases pass. Ruff and
`git diff --check` pass. Each campaign ran its own LateralDrive 64-env/5-update
smoke before long training; the base adaptive smoke also passed.

## Next decision

Blocker fingerprint: `native_command_conditioned_acquisition_and_retention`.
Controller repair persistence is verified. A longer 500-update learning window
produces a better held-out candidate but still triggers preservation failure;
it has not solved retention across reset/DR conditions. Do not infer that
cadence was the sole cause or that the unchanged final policy shows no learning.

A paired diagnostic on the cumulative-4500 retained checkpoint is under
`/tmp/microduck-adaptive-stage0-diagnostic-s17-4500/`. On the same gate seed,
initial-distribution lower-tail is `0.78202` and final-distribution lower-tail
is `0.67967`; both fail. This does not support changing the gate solely on the
claim that stage 0 was already mastered.

Next use the candidate's gate/held-out raw traces and per-bucket reward mass
to distinguish command tracking under DR from retention interference. Pick a
bounded acquisition or consolidation mechanism with an explicit comparison
before spending another training budget. Do not launch another unchanged
continuation or a coefficient-only search. Preserve evaluation seed separation
and keep the held-out evidence independent of live gate decisions.

## Remaining gates and no-touch scope

- Establish all six capabilities across native reset/DR conditions, then
  reproduce the automated procedure with training seeds 17/23/47, native
  held-out evidence, video review, normalized ONNX export and CPU rehearsal.
  Matched-budget comparison follows acquisition; current evidence establishes
  neither native usability nor superiority over the fixed recipe.
- Keep canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward
  signs, signed-EMA metric, product thresholds and the zero anchor unchanged.
- Do not touch unrelated IsaacLab, uv.lock, generated files or other users'
  processes. Fresh-sync portability is unproven; local runs use the validated
  venv. Keep run artifacts under `/tmp`; repo `logs/rsl_rl` is not writable.
- Optional advisor, stronger teachers, generalist, IsaacLab migration and
  hardware deployment remain parked. Stop a run on nonfinite training,
  invalid provenance or resource exhaustion; inspect a regression before
  continuing the same mechanism.

## Verification and launch inventory

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest pytest -q
 tests/test_adaptive_*.py` exercises the adaptive suite. Changed files pass
Ruff; unrelated full-file `mdp.py` findings remain outside this slice.
`scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17
--output <new-dir> --iterations <cumulative-budget> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval <explicit-interval>` records source provenance,
runs smoke64/5, preserves consumed-update accounting and writes explicit
training/campaign manifests. `status: evaluated` denotes completed evidence,
not a passing policy. Prefer an immutable source snapshot for future launches.
