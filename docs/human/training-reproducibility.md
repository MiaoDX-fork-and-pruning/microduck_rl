# Specialist Training Reproducibility

This document records the exact training choices for the 13 accepted Microduck
specialist policies. It is intentionally short: the source code and the
versioned artifact records are the authorities for full metrics and files.

## Fixed Environment

- Source snapshot used for the specialist waves: `652b7ce-20260829T140000Z`.
- Remediation source snapshot: `facd4f4-20260831T1152`.
- Training image: `microduck-rl-cuda128-20260829-0503`.
- Training device: 4096 environments on one GPU per job.
- PPO rollout length: 24 steps per environment.
- Common PPO defaults in the task configs: learning rate `1e-3`, entropy
  coefficient `0.01`, 5 learning epochs, 4 mini-batches, adaptive schedule,
  `gamma=0.99`, `lambda=0.95`.
- Every run first passed the required 64-environment, 5-iteration smoke test.

Reproduce a normal run with:

```bash
uv run train <TASK_ID> --env.scene.num-envs 4096 \
  --agent.max_iterations <ITERATIONS>
```

Use the generated CloudML YAML for the original job's image, source mount,
output mount, and resource details. Do not hand-copy an accepted checkpoint;
use the final inventory below.

## Actual Specialist Runs

| Policy | Actual iterations | Training configuration | Accepted checkpoint |
|---|---:|---|---|
| Velocity-Flat | 6,000 | task defaults | `model_5999.pt` |
| RollerStandUp | 15,000 | task defaults | `model_14999.pt` |
| VelStand | 6,000 initial | task defaults; later R2 continuation | `model_10750.pt` |
| SitStand | 15,000 | task defaults | `model_14999.pt` |
| GroundPick | 20,000 | task defaults | `model_19999.pt` |
| BallKick | 10,000 | task defaults | `model_9999.pt` |
| Roulade | 10,000 | task defaults | `model_9999.pt` |
| StandUp | 15,000 | task defaults | `model_14999.pt` |
| Velocity-Rollers | 6,000 | task defaults, roller config has entropy `0.03` | `model_5999.pt` |
| Swizzle | 6,000 | task defaults | `model_5999.pt` |
| RollerCrouch | 8,000 | task defaults | `model_7999.pt` |
| RollerSlope | 8,000 initial | task defaults; later R4 low-LR continuation | `model_11748.pt` |
| Spin | 8,000 | task defaults | `model_7999.pt` |

The original wave budgets are frozen in
`cloudml/specialist-a1-jobs-652b7ce.json` and
`cloudml/specialist-b1-jobs-652b7ce.json`. The corresponding submitted
commands are in `cloudml/generated/microduck-specialist-*.yaml`.

## Deviations and Continuations

### VelStand

The task config's nominal budget is 20,000 iterations
(`src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py`). The first CloudML
run deliberately used 6,000 iterations. R2 then resumed from `model_5999.pt`
with the task's original PPO settings and `--agent.max-iterations 14000`.
The accepted checkpoint is the R2 `model_10750.pt`, not the end of the still
running continuation.

- Submitted continuation: `cloudml/generated/microduck-specialist-r2-velstand-facd4f4.yaml`
- Full reproducibility parameters: JuiceFS bundle
  `/dongxu/microduck_rl/runs/specialist-r2/velstand-facd4f4-accepted-v1`
- Acceptance and hashes: `cloudml/specialist-r2-velstand-acceptance-facd4f4.json`

### RollerSlope

The original 8,000-iteration run did not generalize across slopes. Commit
`facd4f4` changed the reset sampling so all ten slope levels are represented,
matching the evaluation/play distribution. R2 and R3 standard-PPO
continuations were evaluated and rejected. The accepted R4 pilot resumed from
the R3 lineage and used:

- learning rate: `1e-4` (instead of `1e-3`)
- entropy coefficient: `0.001` (instead of `0.01`)
- 4096 environments, two 500-iteration local continuations

The exact environment and agent YAML files are in the R4 bundle; the accepted
checkpoint is `model_11748.pt`.

- Sampling change and tests: commit `facd4f4`
- Full reproducibility parameters: JuiceFS bundle
  `/dongxu/microduck_rl/runs/specialist-r4/roller-slope-facd4f4-v1`
- Acceptance and hashes: `cloudml/specialist-r4-roller-slope-acceptance-facd4f4.json`

## Final Inventory and Verification

The authoritative 13-policy selection, task IDs, remote checkpoint paths, and
SHA-256 values are in
`cloudml/specialist-final-checkpoints-remediated-facd4f4.json`.

Each accepted policy has a fixed-seed evaluation report and reviewed diagnostic
video. The completion audit proves 13/13 original CloudML jobs succeeded and
one remote checkpoint/report/video exists for every policy:
`cloudml/specialist-training-completion-audit-facd4f4.json`.

For deployment reproduction, export only through the mandatory normalizer-aware
path:

```bash
uv run scripts/export.py <TASK_ID> \
  --checkpoint-file /path/to/model_XXXX.pt \
  --onnx-file model.onnx
```

The VelStand and RollerSlope bundles also contain deterministic golden actions
and ONNX parity reports. The specialist toolchain test suite currently passes
with `41 passed`.

## Scope

This record covers the 13 flat specialist policies. Rough/Backlash variants,
the 71D generalist, hardware deployment, and left-kick expansion are separate
experiments and are not implied by this inventory.
