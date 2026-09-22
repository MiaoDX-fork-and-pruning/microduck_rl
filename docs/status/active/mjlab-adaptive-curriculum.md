# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: sustained authorized intuitive-flow execution; the status request
does not cancel necessary changes, training, or behavioral checks.

## Current slice

The pure-yaw linear-sway exemption is committed as `0c0b50e`. Focused tests
(`55 passed`), compileall, and smoke64/5 passed with the canonical curriculum,
61D/14D ABI, and BAM M6 intact. A fresh seed17 lateral-drive run completed 500
updates at 4096 envs; manifests are under
`/tmp/microduck-adaptive-yaw-linear-exempt-s17-500/`.

The adaptive controller retained the zero bucket and completed four valid hold
windows without rollback or nonfinite values. At 500 updates it was still
lateral-focused (`28%` lateral exposure); no difficulty stage was advanced.
The run is a bounded reward diagnostic, not a usable-policy result.

## Last proven evidence

| Native held-out metric | Yaw-linear-exempt 500 | Product requirement |
| --- | ---: | ---: |
| Zero endpoint drift (m) | 0.06889 | <=0.012 |
| Forward MAE (m/s) | 0.01952 | <=0.024 |
| Lateral MAE (m/s) | 0.09693 | <=0.024 |
| Yaw MAE (rad/s) | 0.53351 | <=0.12 |
| Left/right turn MAE (rad/s) | 0.31074 / 0.38136 | <=0.12 |

Only forward passes. All six native cases survive their 6 s rollouts. The
native held-out report is at
`/tmp/microduck-adaptive-yaw-linear-exempt-s17-500/heldout/capability.json`.
CPU/ONNX passes only zero and reports yaw MAE `0.61870 rad/s`; it remains
separate transfer evidence using XML position actuators.

Compared with the earlier 500-update lateral-drive evidence, removing the
remaining pure-yaw linear sway charge improved held-out yaw MAE from about
`0.73` to `0.53 rad/s` and preserved forward/lateral progress, but zero drift
regressed to `0.06889 m` and all six-bucket acceptance still fails. This changes
the blocker classification from “linear sway conflict is untested” to “the
conflict is real but insufficient by itself.”

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

Continue the exact `model_499.pt` checkpoint to cumulative 1000 updates with
the same source and gate settings. This tests whether the measured yaw gain
survives adaptive focus rotation and whether zero recovery returns; it does not
change thresholds or open a multi-seed campaign. Stop if zero/forward/lateral
preservation fails, training becomes nonfinite, or the 1000-update held-out
report shows no decision-changing yaw/turn improvement.

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
