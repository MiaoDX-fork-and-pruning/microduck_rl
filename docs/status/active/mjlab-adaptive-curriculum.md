# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: sustained authorized intuitive-flow execution; the status request
does not cancel necessary changes, training, or behavioral checks.

## Current slice

The seed 17 yaw-feedback continuation is complete at 3000 updates and is not
running in the background. It resumed the exact previous frontier checkpoint at
4096 envs from source `e068387`; manifests are under
`/tmp/microduck-adaptive-yaw-feedback-s17-3000/`. Smoke64/5 passed and the
training result is finite, but both the native held-out report and the
independent CPU/ONNX report have `passed: 0`.
The current branch includes the canonical-curriculum ownership repair
`920e0ec`; the focused suite is green at 54 tests.

The adaptive controller did execute its intended behavior: it retained the
zero bucket, rotated focus after stalls, increased yaw exposure, and completed
without a nonfinite event. The gate seed briefly improved yaw, but that did not
transfer to held-out seeds. The final gate state focused `turn-right`; no
difficulty stage was advanced and no usable checkpoint was established.

## Last proven evidence

| Native held-out metric | Yaw-feedback 3000 | Product requirement |
| --- | ---: | ---: |
| Zero endpoint drift (m) | 0.04456 | <=0.012 |
| Forward MAE (m/s) | 0.02133 | <=0.024 |
| Lateral MAE (m/s) | 0.11037 | <=0.024 |
| Yaw MAE (rad/s) | 0.74515 | <=0.12 |
| Left/right turn MAE (rad/s) | 0.30365 / 0.21847 | <=0.12 |

Only forward passes. All six native cases survive their 6 s rollouts. The
native held-out report is at
`/tmp/microduck-adaptive-yaw-feedback-s17-3000/heldout/capability.json`.
CPU/ONNX passes only zero and reports yaw MAE 0.76967 rad/s; it remains
separate transfer evidence using XML position actuators.

Bounded anti-stall rotation works as orchestration: yaw exposure reached about
12.3% and the final turn-right focus reached about 21.7%, with no invalid
rollback. This did not acquire the missing yaw/turn capability on held-out
seeds. CoM/head-CoM remain at +/-3 mm; no difficulty advance or reproducible
usable-policy procedure is established.

Completed repairs: runner state/rollback consolidation, command-aligned feedback,
zero retention, checkpointed frontier order, inclusive tracking-stage boundaries
(`8638d29`), anti-stall rotation (`e9f3a74`), focus preservation (`0d9567b`),
and canonical curriculum ownership (`920e0ec`). These prove orchestration, not
all-capability acquisition. Earlier lateral-drive and frontier-preserve reports
remain under their named `/tmp` campaign paths.

## Next decision and experiment boundary

Blocker fingerprint: `native_yaw_acquisition`.
Current classification: yaw stays near stationary despite increased exposure;
the instantaneous zero-linear-command cost also penalizes pure-turn gait sway.
Counterfactual reward replay:
`/tmp/microduck-yaw-acquisition-diagnostic-e068387/reward-replay.json`.
On a mature fixed-policy trace with mean yaw 0.7845 rad/s for a 0.8 command,
current weighted linear L1 averages -2.3269; at yaw-feedback-2625, the nearly
stationary trace costs -0.0152. Averaging planar error during active yaw would
reduce the former to -0.3611 while retaining a cost for sustained translation.
This is reward-conflict evidence, not a training or acceptance result.

The current bounded decision is a reward-semantics diagnostic, not a product
threshold change. The recorded traces show the pure-yaw policy remains nearly
stationary even while its yaw-specific L1 cost is substantial. Before another
long run, test a command-aligned acquisition signal that distinguishes active
turning from exact idle and preserves the canonical curriculum. Success requires
lower native yaw/turn error while retaining zero/forward/lateral; run focused
tests and smoke64/5 first, then a fresh bounded continuation. Do not start a
multi-seed campaign until all six native held-out buckets pass.

The completed manifests are `training-result.json`, `campaign-result.json`,
`heldout/capability.json`, and `cpu-transfer/capability.json` under the campaign
directory. Any next experiment must inspect every new gate report,
focus/exposure state and preservation decision.

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
pytest -q tests/test_mjlab_adaptive_velocity_config.py
tests/test_adaptive_velocity_l1_rewards.py tests/test_adaptive_command_exposure.py`.
Broader adaptive runner/checkpoint tests are required for controller changes.
Every training change requires smoke64/5 before a long run. Campaign launch uses
`scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17
--output <new-dir> --iterations <cumulative-budget> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval 125`, with source provenance recorded. Prefer an
immutable source snapshot for future launches. `status:evaluated` means the
experiment completed, not that the policy passed.
