# MJLab Adaptive Curriculum Wave 1

- Status: ACTIVE; executable pipeline validated
- Source commit: `30ebb4e` on `origin/holy-ape`
- Source snapshot: JuiceFS `/dongxu/microduck_rl/source/adaptive-curriculum/30ebb4e`
- Workspace/context: CloudML workspace `10076`, Executor context
- Queue/resource: `11759`, `cloudml.ng1r49-8-8.13-107`, 1 GPU, GUARANTEED
- Current slice: pipeline validation succeeded; long Wave 1 r2 jobs remain running.
- Next action: collect terminal control/static/one-axis checkpoints and run the same battery.
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

Monitoring command:

```bash
/home/mi/executor/exe compute cloudml cml -- custom_train describe <JOB_ID>
```

No production task or canonical schedule was modified. The standing/action-rate
diagnostic branch is parked until its independent controller is implemented.
