# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: sustained authorized intuitive-flow execution; the status request
does not cancel necessary changes, training, or behavioral checks.

## Current slice

The command-conditioned evidence contract is committed as `e4ac803`, with the
inference-mode buffer fix in `c5e80fa`. `BucketFeedbackTracker` records seven
command classes (`zero`, five explicit capability buckets, and `nominal`) with
sample counts, exposure fractions, and per-term signed and absolute weighted
reward mass. The state is checkpointed, attached to evaluation events, and
reset safely across rollback/resume.

The retention repair is committed as `b2dc088`; the follow-up controller-state
fix is `b3db958`. The latter preserves live command exposure across policy
rollback so repeated failures accumulate bounded repair slices instead of
restoring stale exposure from the old known-good checkpoint. Focused adaptive
tests pass (`147 passed`), and the canonical 64-env/5-update BAM smoke passes.
The 61D/14D ABI, BAM M6, unfiltered actions, and normalizer-baked export remain
intact.

The prior bounded evidence continuation is
`/tmp/microduck-adaptive-evidence-s17-3750-r2/`; it validated seven-class
feedback but ended with held-out native lower-tail `0.75130`. The first
post-repair continuation is
`/tmp/microduck-adaptive-evidence-s17-4250-repair/`; it completed 500 updates,
recorded four native windows, and ended with held-out native lower-tail
`0.42439`. It reproduced three rollback cycles and showed the pre-`b3db958`
state-reversion bug (`retention_repairs` repeatedly reset to 1). The new
250-update validation is running under
`/tmp/microduck-adaptive-evidence-s17-4500-retention/`.

## Last proven evidence

The prior final native checkpoint is
`/tmp/microduck-adaptive-corrected-s17-3500/training/logs/rsl_rl/matched_lateral-drive/2026-09-22_10-29-40_matched-lateral-drive-s17/model_3499.pt`.
Its held-out native report uses seed `20260915` and has lower-tail score
`0.78469`: zero, forward, lateral, yaw, and turn-right pass; turn-left fails
with signed-EMA error `0.12919 rad/s` (score `0.78469`). The report is
`/tmp/microduck-adaptive-corrected-s17-3500/heldout/capability.json`.

Independent native reset/DR checks on that same checkpoint are less stable:
seed `20260921` fails yaw (`0.15395 rad/s`, aggregate `0.74342`), while seed
`20260922` fails forward (`0.04297 m/s`), lateral (`0.02481 m/s`), and yaw
(`0.10184 rad/s`, aggregate `0.64188`). These are native MJLab/BAM reports,
not CPU transfer results. The CPU/ONNX rehearsal has lower-tail score `0.0`
and is retained only as separate transfer evidence using XML position
actuators. The 4250-update native continuation is
`/tmp/microduck-adaptive-evidence-s17-4250-repair/`; it completed four valid
windows and ended with held-out lower-tail `0.42439`. Its event trace shows
three rollback cycles and the pre-`b3db958` bug: every rollback restored stale
exposure, so `retention_repairs` reset to 1. The campaign's `status: evaluated`
means artifacts were produced, not that the policy passed. There is still no
usable policy.

The diagnostic traces also separate command bias from gait ripple: on the
held-out trace, mean velocity bias is about `-0.015 m/s` forward and
`-0.0025 m/s` lateral while samplewise MAE is `0.0217 / 0.1070`; yaw mean
bias is about `+0.014 rad/s` while samplewise MAE is `0.2227`. Across the two
additional seeds, the failures move between yaw and linear buckets. This makes
the remaining blocker a command-conditioned acquisition/retention problem,
not evidence that the signed-EMA metric is simply measuring a fixed DC bias.

## Next decision and experiment boundary

Blocker fingerprint: `native_command_conditioned_acquisition_and_retention`.
The remaining issue is no longer missing orchestration or evaluator parity. The
controller can alter exposure and roll back safely, but the learned policy does
not retain all six capabilities across native reset/DR seeds. Lateral remains
near the gate, yaw is seed-sensitive, and turn retention regresses. The current
slice tests whether preserving live exposure changes that failure mode.

Do not start the formal training seeds `17/23/47` matrix yet. Finish the
post-`b3db958` two-window native check first, then use the recorded per-bucket
reward mass to choose the next bounded acquisition mechanism. Keep product
thresholds, signed-EMA metric, 61D/14D ABI, BAM M6, and the zero anchor fixed.
Do not make another coefficient-only continuation.

The completed manifests are `training-result.json`, `campaign-result.json`,
`heldout/capability.json`, and `cpu-transfer/capability.json` under the campaign
directory. Any next experiment must inspect every gate report, focus/exposure
state, preservation decision, and native seed replay.

## Remaining gates and no-touch scope

- Acquire all six capabilities, then reproduce the procedure on training seeds
  17/23/47 with disjoint held-out native evidence, video inspection, normalized
  ONNX export and CPU rehearsal. No broad multi-seed campaign before acquisition.
- Keep current product thresholds and samplewise MAE fixed. Semantic calibration
  remains open: gait ripple contributes to MAE; kicked zero endpoint offset is
  not directly observed by the actor. The paired probe under
  `/tmp/adaptive-push-paired/v3/` shows 0.0671 m kicked offset versus 0.00258 m
  without a kick, both ending near 0.001 m/s. Any metric change needs a written
  independent calibration argument; failing motion must remain failing.
- Preserve canonical velocity, 61D/14D ABI, BAM M6, unfiltered actions and export
  normalization. Do not touch unrelated IsaacLab, uv.lock, generated files or
  other users' processes. Fresh-sync portability is unproven; local runs reuse
  the validated venv. Keep logs under `/tmp`; repo `logs/rsl_rl` is not writable.
- Stop on nonfinite training, invalid provenance, resource exhaustion or
  demonstrated regression. Optional advisor, stronger teachers, generalist,
  IsaacLab migration and hardware deployment remain parked.

## Verification inventory

Focused proof: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest
pytest -q tests/test_adaptive_*.py` (`147 passed`), plus the canonical 64-env
five-update BAM smoke after `b3db958`. The changed adaptive files pass Ruff;
pre-existing full-file `mdp.py` lint findings remain outside this slice.
Every training change requires smoke64/5 before a long run. Campaign launch uses
`scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17
--output <new-dir> --iterations <cumulative-budget> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval 125`, with source provenance recorded. Prefer an
immutable source snapshot for future launches. `status:evaluated` means the
experiment completed, not that the policy passed.
