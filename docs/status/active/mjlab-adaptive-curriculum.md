# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: sustained authorized intuitive-flow execution; the status request
does not cancel necessary changes, training, or behavioral checks.

## Current slice

The pure-yaw linear-sway exemption is committed as `0c0b50e`. Focused tests
(`55 passed`), compileall, and smoke64/5 passed with the canonical curriculum,
61D/14D ABI, and BAM M6 intact. Seed17 was resumed through cumulative 1500
updates at 4096 envs; manifests are under
`/tmp/microduck-adaptive-yaw-linear-exempt-s17-1500/`.

The adaptive controller retained the zero bucket and completed twelve valid
hold windows without rollback or nonfinite values. It switched from lateral
focus to yaw at cumulative 750 and ended at `25.33%` yaw exposure; no difficulty
stage was advanced. The run is still a bounded diagnostic, not a usable-policy
result.

## Last proven evidence

| Native held-out metric | Yaw-linear-exempt 1500 | Product requirement |
| --- | ---: | ---: |
| Zero endpoint drift (m) | 0.09445 | <=0.012 |
| Forward MAE (m/s) | 0.02248 | <=0.024 |
| Lateral MAE (m/s) | 0.10197 | <=0.024 |
| Yaw MAE (rad/s) | 0.16237 | <=0.12 |
| Left/right turn MAE (rad/s) | 0.23486 / 0.26773 | <=0.12 |

Only forward passes. All six native cases survive their 6 s rollouts. The
native held-out report is at
`/tmp/microduck-adaptive-yaw-linear-exempt-s17-1500/heldout/capability.json`.
CPU/ONNX passes only zero and reports forward/lateral MAE
`0.10695 / 0.12708 m/s` and yaw MAE `0.43787 rad/s`; it remains separate
transfer evidence using XML position actuators.

Compared with the 1000-update continuation, the extra 500 updates improved
held-out yaw MAE from `0.21180` to `0.16237 rad/s`, while lateral stayed flat and
zero drift regressed from `0.06409` to `0.09445 m`; turns did not improve. The
final checkpoint was saved just before the `1500 × 24` tracking-width boundary,
so the logged `tracking_std` remained approximately `0.16` rather than spending
an update under the final `0.12` stage. The pure-yaw conflict is therefore real,
but its repair is insufficient by itself and zero recovery remains a separate
failure.

Completed repairs: runner state/rollback consolidation, command-aligned feedback,
zero retention, checkpointed frontier order, inclusive tracking-stage boundaries
(`8638d29`), anti-stall rotation (`e9f3a74`), focus preservation (`0d9567b`),
and canonical curriculum ownership (`920e0ec`). These prove orchestration, not
all-capability acquisition. Earlier lateral-drive and frontier-preserve reports
remain under their named `/tmp` campaign paths.

## Next decision and experiment boundary

Blocker fingerprint: `native_yaw_acquisition`.
Current classification: the remaining pure-yaw planar penalty was a real
conflict, because removing it improved yaw/turn traces, but yaw is still far
from the product gate and zero recovery regressed. The old replay values in
`/tmp/microduck-yaw-acquisition-diagnostic-e068387/reward-replay.json` describe
the pre-`1780d58` implementation and must not be quoted as current reward
semantics.

The exact cumulative-1500 `model_1499.pt` checkpoint is now the diagnostic
frontier. A bounded continuation to cumulative 2000 with the same source and
gate settings is justified only to spend a full segment after the final
canonical tracking-width boundary (`std=0.12`) becomes active. It does not
change thresholds or open a multi-seed campaign. If that continuation does not
produce a material native yaw/turn or zero-recovery improvement, stop repeating
coefficient-only variants and isolate the remaining command-conditioned
acquisition/zero-recovery conflict with a new evidence contract.

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
