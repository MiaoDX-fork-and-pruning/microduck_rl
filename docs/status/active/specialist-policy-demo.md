# Specialist policy demo active run

Status: `ACTIVE`
Source plan: `docs/specialist_policy_demo_plan.md`
Control plane: primary Codex session
Latest intent: implement the approved specialist policy plan via intuitive-flow

Current slice: S0 local preflight is complete. S1 CloudML submission is the
next boundary.

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

Next slice: upload the prepared immutable source snapshot, then submit the P0
Velocity + RollerStandUp pilots after explicit authorization.

Prepared P0 launch package:

- source commit: `104d1308f050788f949aba5747baefe7a2f564f2`;
- local snapshot directory:
  `/tmp/microduck-source-104d1308f050-20260829T133725Z` (240 files, 29 MB);
- snapshot tar SHA-256:
  `75c470b0c902417f6c9028f6500160420504a06fbd72fb014ae275abf5bcc4ea`;
- proposed immutable source destination:
  `/dongxu/microduck_rl/source/104d1308f050-20260829T133725Z`;
- shared image:
  `micr.cloud.mioffice.cn/cc-proxy/thelastfoot-openpi-g2-training:microduck-rl-cuda128-20260829-0503`;
- job specifications:
  `cloudml/microduck-specialist-p0-velocity-104d130.yaml` and
  `cloudml/microduck-specialist-p0-roller-standup-104d130.yaml`;
- CloudML context/workspace: `executor` / `10076`;
- lane: queue `11759`, guaranteed single-GPU
  `r49-24g | 13 CPU | 107 GiB`, priority 5, non-preemptible;
- read-only capacity check: 95 of 176 queue-wide slots free; quota reports
  the 176-slot guaranteed allocation with `leftNumber: 0`, so queue capacity
  is the actionable availability signal for this existing allocation;
- prior lineage `t-20260829173040-5fdmx` succeeded and is terminal; no active
  Microduck job was found during the preflight query.

The snapshot is a `git archive` of the recorded commit. It excludes the
unrelated untracked `tmp/` directory and the local submission-control files.
Both YAML files parse successfully. The installed CML client has no submission
dry-run, so schema validation deliberately stops before `custom_train submit`.

Next proof: record the uploaded snapshot URI/hash and the two CloudML job IDs,
then verify their resolved mounts, resource lanes, and writable output prefixes.

Stop condition: do not consume shared/paid CloudML capacity or upload a large
image/source artifact without explicit authorization and resolved destinations.

No-touch scope: Generalist 71D schema, distillation, runtime repository, robot
hardware, Rough/Backlash variants.

Parked work: Track B does not block Track A; left kick remains conditional.
