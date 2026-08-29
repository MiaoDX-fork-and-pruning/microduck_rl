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

Next slice: create and upload an immutable source snapshot, verify CloudML
queue/quota and output ownership, then submit the P0 gait + episodic pilots.

Next proof: record snapshot URI/hash, shared image reference, CloudML job IDs,
resource lanes, and writable output prefixes.

Stop condition: do not consume shared/paid CloudML capacity or upload a large
image/source artifact without explicit authorization and resolved destinations.

No-touch scope: Generalist 71D schema, distillation, runtime repository, robot
hardware, Rough/Backlash variants.

Parked work: Track B does not block Track A; left kick remains conditional.
