# MJLab Adaptive Curriculum v2

- Status: Phase 0 capability contract and Phase 1 runner-control contract implemented and verified; matched-budget comparison pending
- Source commit: `c72423f` on `origin/holy-ape`
- Source snapshot: JuiceFS `/dongxu/microduck_rl/source/adaptive-curriculum/c72423f`
- Workspace/context: CloudML workspace `10076`, Executor context
- Queue/resource: `11759`, `cloudml.ng1r49-8-8.13-107`, 1 GPU, GUARANTEED
- Current slice: strict v2 reports, runner evaluator seam, provenance validation,
  decision classification, checkpoint metadata, RNG capture, and explicit rollback
  boundary are implemented.
- Next action: complete fake-runner save/load replay tests, then execute the new
  `02-matched-budget-evaluation-PLAN.md` campaign; do not treat smoke as a
  policy-quality claim.
- Stop condition: do not compose or fine-tune until control, static, and one-axis
  results have complete metrics and reproducible artifacts.

## Jobs

| Branch | Task | Job ID | Output prefix | Initial state |
| --- | --- | --- | --- | --- |
| canonical control | `Mjlab-Velocity-Flat-MicroDuck` | `t-20260915111321-9fhk9` | `.../wave1-control-af06b34` | deploying |
| all-static | `Mjlab-Velocity-Flat-Adaptive-Static-MicroDuck` | `t-20260915111322-9jxop` | `.../wave1-static-af06b34` | deploying |
| CoM-only | `Mjlab-Velocity-Flat-Adaptive-CoM-MicroDuck` | `t-20260915111323-hg7uj` | `.../wave1-com-af06b34` | deploying |
| head-CoM-only | `Mjlab-Velocity-Flat-Adaptive-HeadCoM-MicroDuck` | `t-20260915111325-wvmsz` | `.../wave1-headcom-af06b34` | deploying |
| composed | `Mjlab-Velocity-Flat-Adaptive-MicroDuck` | `t-20260915111325-06cfm` | `.../wave1-composed-af06b34` | enqueued |

The first wave used an invalid `/ml-engine/code/src` Python path and must not be
used as experiment evidence. Those jobs are retained for audit. Corrected r2
jobs use `/mnt/cloudml/source/src` and the following IDs:

| Branch | Job ID | Output prefix | State at submission |
| --- | --- | --- | --- |
| canonical control | `t-20260915111713-zyhn4` | `.../wave1-control-r2-af06b34` | running |
| all-static | `t-20260915111714-w8bwd` | `.../wave1-static-r2-af06b34` | deploying |
| CoM-only | `t-20260915111715-offvh` | `.../wave1-com-r2-af06b34` | running |
| head-CoM-only | `t-20260915111716-5vmue` | `.../wave1-headcom-r2-af06b34` | running |
| composed | `t-20260915111718-nhoyf` | `.../wave1-composed-r2-af06b34` | running |

## Executable validation

The train/export/battery/gate chain was run on CloudML job
`t-20260915121317-8bi9e` (source `30ebb4e`). It trained 20 iterations, exported
`model_19.pt` to ONNX, and ran all six frozen buckets. Results were
`zero=1.0`, `forward=1.0`, `lateral=1.0`, `yaw=1.0`, `turn-left=1.0`, and
`turn-right=1.0`; all traces were finite with 61D observations and 14D actions.
The first window correctly held at CoM stage `0.003` because the gate requires
two consecutive passing windows. Resume job `t-20260915122019-jgdmg` restored
the state and produced the recorded transition `com_range: 0.003 -> 0.005` at
step 44 with score `1.0`.

The validation output is under
`/dongxu/microduck_rl/runs/adaptive-curriculum/20260915/pipeline-30ebb4e-r4/`.

## Completed wave-1 experiments

The five-window jobs used 100 PPO iterations per window, exported each window's
checkpoint, ran the same six-bucket battery, and resumed from the previous
checkpoint. Both jobs succeeded:

| Branch | Job ID | Output prefix | Battery result | Gate trace |
| --- | --- | --- | --- | --- |
| CoM-only windows | `t-20260915131425-nkqpf` | `.../windows-com-c72423f` | 5/5 windows score `1.0`; all finite | `com_range 0.003 -> 0.005` at step 200; `head_com_range 0.003 -> 0.005` at step 300 |
| composed windows | `t-20260915131432-mxfji` | `.../windows-composed-c72423f` | 5/5 windows score `1.0`; all finite | same two transitions at steps 200 and 300 |

The terminal adaptive stage is `com_range=0.005` and `head_com_range=0.005`.
Across all adaptive windows, every bucket (`zero`, `forward`, `lateral`, `yaw`,
`turn-left`, `turn-right`) passed with `finite_61d_14d=true`.

The completed r2 4000-iteration checkpoints were evaluated by job
`t-20260915134239-e5rt8`. Its output is
`/dongxu/microduck_rl/runs/adaptive-curriculum/20260915/wave1-battery-c72423f/`.
Canonical control, all-static, CoM-only, head-CoM-only, and composed each scored
`1.0` on all six buckets, with finite 61D/14D traces. Maximum tilt across the
five terminal policies was `0.0766 rad` (head-CoM-only); minimum final height
was `0.1159 m` (all-static).

These results validate the battery and resumable gate mechanics and show no
wave-1 held-out regression. They do not establish a matched-budget training
advantage because the adaptive window runs total 500 iterations while the r2
controls total 4000 iterations.

Monitoring command:

```bash
/home/mi/executor/exe compute cloudml cml -- custom_train describe <JOB_ID>
```

No production task or canonical schedule was modified. The standing/action-rate
diagnostic branch is parked until its independent controller is implemented.

## v2 implementation evidence

- Capability report validation recomputes bucket components, scores, validity,
  pass flags, and aggregate from raw evidence. Malformed, non-finite, missing, or
  mismatched traces fail closed.
- Runner validates report schema, checkpoint existence and SHA256, task axis mode,
  and enabled-axis allowlist. Evaluator exceptions become `evaluation_error` holds.
- Gate outcomes are typed as `hold`, `advance`, `regress`, or
  `preservation_failure`; runner records the causal event and applies only the
  allowed live EventManager axis.
- Checkpoint metadata co-locates gate state, stage values, evaluator schema and
  iteration/env-step counters, known-good checkpoint, evaluation events, and RNG
  state. Explicit rollback rejects a checkpoint other than the recorded known-good.
- Verification: 27 focused tests pass, Ruff and diff checks pass, adaptive 64-env /
  5-iteration smoke passes with 61D observations, 14D actions, BAM M6, and no NaN.

## Phase 1 replay gate and matched-budget campaign (2026-09-18)

The missing fake-runner checkpoint tests were added and passed. The replay gate
now covers PPO-like policy state, gate trace, live EventManager ranges, Python /
NumPy / Torch RNG restoration, incompatible-axis rejection, and explicit
known-good rollback. The focused suite passes 31 tests; Ruff, `git diff --check`,
the canonical-task diff guard, and the adaptive 64-env / 5-iteration smoke all
pass. The implementation is committed as `b6788e3`.

An immutable source snapshot for that commit was uploaded to
`/dongxu/microduck_rl/source/adaptive-curriculum/b6788e3`. The first campaign
submission exposed two operational issues and is retained for audit: a top-level
`--seed` was rejected (the CLI requires `--agent.seed`), and the initial CoM /
composed commands had evaluation disabled and therefore represented only the
initial static slice. Those four jobs were explicitly stopped before producing
evidence. The corrected r2 campaign uses `--agent.seed`; all-static s17 is
running as `t-20260918100626-bfpsl`, while all-static s23 is
`t-20260918100627-j3snt`. Corrected CoM/composed jobs were stopped and will be
re-submitted with the staged battery/resume harness after quota release. No
matched-budget policy result or adaptive improvement claim has been made.
