# Specialist policy demo active run

Status: `ACTIVE`
Source plan: `docs/specialist_policy_demo_plan.md`
Control plane: primary Codex session
Latest intent: implement the approved specialist policy plan via intuitive-flow

Current slice: S0 local preflight is complete; P0 is complete and the first
8-slot A1 CloudML wave is deploying.

Last proven evidence:

- all 13 task smokes passed at 64 envs / 5 iterations on RTX 3090;
- CPU tests: 157 passed, 1 skipped;
- plain and roller ONNX exports: 61D input, 14D finite output, baked normalizer;
- container digest `sha256:246aadb0cc9c2cda0f54dba9b97f7a007d088adb59874f18b2b1395ea788aa32`;
- fresh container Velocity smoke passed.

Completed slices: deterministic scenario and manifest contract; artifact
validator and tests; full task coverage; PyPI lock repair; RollerStandUp
servo-index fix; S0 host/container proof.

Deterministic scenario slice is now executable: `scripts/specialist_scenario.py`
expands the canonical 90-second, 50 Hz scenario to exactly 4,500 frames,
zero-pads 3D twists into the 13D command ABI, and rejects discontinuous policy
chains or off-grid switch times. The scenario now includes explicit sit-hold
and rise commands at 30 s and 36 s. The validator consumes the same compiler.
Full CPU suite after this change: 165 passed, 1 skipped.

ONNX parity evidence is also executable through
`scripts/compare_specialist_onnx.py`. It compares fixed-seed PyTorch golden
observations/actions with CPU ONNX Runtime one sample at a time (the exporter
uses a fixed batch size of 1), requires zero-command and command-extreme cases,
and writes hashes plus per-case error metrics. An end-to-end check against the
S0 Velocity ONNX passed all three boundary cases. Full CPU suite after this
slice: 170 passed, 1 skipped.

The offline manifest gate now requires, per accepted policy, checkpoint, ONNX,
metadata, evaluation JSON, diagnostic video, and parity JSON. It verifies every
SHA-256, the 61D/14D ABI in metadata and parity, finite evaluation, non-positive
penalties, a numeric success rate and main-task metric, and a non-empty video
review. The example manifest remains a placeholder template until real
checkpoints exist; it is intentionally not a valid handoff.

`scripts/build_video_gallery.py --manifest ...` now consumes that same evidence
contract. It discovers diagnostic clips and renders acceptance status, success
and main-task metrics, penalties, video review, failure note, and all artifact
hashes in the standalone HTML index. Full CPU suite: 175 passed, 1 skipped.

`scripts/infer_policy.py --scenario ...` now routes the canonical schedule
through deployment-style ONNX inference. It preflights every referenced policy,
uses fixed 50 Hz timing for scenario phases, honors explicit switch times rather
than interactive auto-return timers, and supports accelerated `--no-realtime`
rehearsal. A full MuJoCo/Xvfb check using the S0 ONNX in all roles executed all
12 events and 4,500 frames, then exited at `Scenario complete`. Full CPU suite:
180 passed, 1 skipped.

Next slice: monitor and accept the P0 Velocity + RollerStandUp pilots, then
launch the remaining Track A and Track B waves.

Prepared and live P0 launch package:

- source commit: `652b7ce1b19f6c889b5683b60e439a29880605ce`;
- uploaded snapshot tar SHA-256:
  `c06bc12a138d10ff245bd6cdc4fcf469e830d357382692d96d223b15c0e4c253`;
- uploaded source destination:
  `/dongxu/microduck_rl/source/652b7ce-20260829T140000Z/`;
- shared image:
  `micr.cloud.mioffice.cn/cc-proxy/thelastfoot-openpi-g2-training:microduck-rl-cuda128-20260829-0503`;
- job specifications:
  `cloudml/microduck-specialist-p0-velocity-652b7ce.yaml` and
  `cloudml/microduck-specialist-p0-roller-standup-652b7ce.yaml`;
- CloudML context/workspace: `executor` / `10076`;
- lane: queue `11759`, guaranteed single-GPU
  `r49-24g | 13 CPU | 107 GiB`, priority 5, non-preemptible;
- read-only capacity check: 95 of 176 queue-wide slots free; quota reports
  the 176-slot guaranteed allocation with `leftNumber: 0`, so queue capacity
  is the actionable availability signal for this existing allocation;
- prior lineage `t-20260829173040-5fdmx` succeeded and is terminal; no active
  Microduck job was found during the preflight query.

Live P0 jobs:

- Velocity: `t-20260829233250-qzkbi`, output
  `/dongxu/microduck_rl/runs/specialist-p0/velocity-652b7ce-v2`;
- RollerStandUp: `t-20260829233248-jaq6f`, output
  `/dongxu/microduck_rl/runs/specialist-p0/roller-standup-652b7ce-v2`.

These are only the P0 calibration pair. The complete initial coverage is 13
independent tasks: 6 Track A teachers and 7 Track B showcase policies. After
P0 acceptance, the remaining 11 tasks will be submitted in bounded waves,
subject to fresh queue/quota checks and independent output prefixes.

A1 wave jobs (all created 2026-08-30, currently `deploying`):

- VelStand `t-20260830214918-gxien`
- SitStand `t-20260830214918-p98ty`
- GroundPick `t-20260830214918-77usp`
- BallKick `t-20260830214918-kgjma`
- Roulade `t-20260830214918-mkrjo`
- StandUp `t-20260830214918-5cmql`
- Velocity-Rollers `t-20260830214918-weych`
- Swizzle `t-20260830214918-0sllb`

The account-level quota is 8 concurrent guaranteed R49 instances. Queue-wide
capacity is ample (`107/176` free at the latest check), but that does not raise
the personal cap. B1 RollerCrouch, RollerSlope, and Spin were prepared and
their output prefixes were created, but submission was correctly rejected by
the 8-instance personal quota. They will be submitted one-for-one as A1 jobs
reach terminal states; no A1 job will be stopped just to make room.

The snapshot is a `git archive` of the recorded commit. It excludes the
unrelated untracked `tmp/` directory and the local submission-control files.
Both YAML files parse successfully. The installed CML client has no submission
dry-run, so schema validation deliberately stops before `custom_train submit`.

Next proof: verify both P0 checkpoints and acceptance reports, then record the
remaining wave job IDs, resolved mounts, resource lanes, and output prefixes.

Stop condition: launch B1 only after an A1 slot is released and a fresh quota
check passes; failed tasks remain isolated and do not overwrite completed
lineages.

No-touch scope: Generalist 71D schema, distillation, runtime repository, robot
hardware, Rough/Backlash variants.

Parked work: Track B does not block Track A; left kick remains conditional.
