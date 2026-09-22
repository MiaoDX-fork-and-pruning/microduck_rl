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

Focused adaptive tests pass (`142 passed`), the canonical 64-env/5-update
smoke passes, and the Feedback recipe smoke writes nonzero samples for all
seven classes with `unclassified_count=0`. The 61D/14D ABI, BAM M6, unfiltered
actions, and normalizer-baked export remain intact.

Seed17 then completed a cumulative 3500-update continuation at 4096 envs from
the cumulative-2500 checkpoint. The complete campaign is under
`/tmp/microduck-adaptive-corrected-s17-3500/`.

The bounded evidence continuation is
`/tmp/microduck-adaptive-evidence-s17-3750-r2/`. It resumed the cumulative
3500 checkpoint for 250 updates, completed two valid native evaluation windows
with command feedback, and recorded exposure/rollback behavior. The final
held-out native report has lower-tail `0.75130`, with forward, lateral, and
turn-right below threshold. Both adaptive difficulty axes stayed at stage 0.
This validates the measurement path but does not establish a usable policy.

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
actuators. The new bounded continuation's held-out report is
`/tmp/microduck-adaptive-evidence-s17-3750-r2/heldout/capability.json`; its
native gate-window reports are under the training run's
`adaptive_eval/model_3624.eval/` and `adaptive_eval/model_3749.eval/`
directories. Each feedback window contains roughly 12.4M samples, seven-class
fractions matching live exposure, and zero unclassified samples. Per-bucket
reward mass is now available for the next acquisition intervention; the
failure remains native capability retention.

The runner metadata shows the final preservation failure at env step `84000`
and rollback to `model_3249.eval.adaptive.pt`; the campaign's
`status: evaluated` therefore means artifacts were produced, not that the
policy passed. There is still no usable policy.

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
near the gate, yaw is seed-sensitive, and the final turn-left result regresses.

Do not start the formal training seeds `17/23/47` matrix yet. The evidence
contract is now proven, so the next intervention must use the recorded
per-bucket reward mass to change one bounded acquisition mechanism and then
repeat native held-out evaluation. Keep product thresholds, signed-EMA metric,
61D/14D ABI, BAM M6, and the zero anchor fixed. Do not make another
coefficient-only continuation.

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
pytest -q tests/test_adaptive_*.py` (`142 passed`), plus compileall and the
canonical/Feedback 64-env five-update smokes. The changed adaptive files pass
Ruff; pre-existing full-file `mdp.py` lint findings remain outside this slice.
Every training change requires smoke64/5 before a long run. Campaign launch uses
`scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17
--output <new-dir> --iterations <cumulative-budget> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval 125`, with source provenance recorded. Prefer an
immutable source snapshot for future launches. `status:evaluated` means the
experiment completed, not that the policy passed.
