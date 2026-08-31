# Specialist policies: training, switching demo, and ONNX validation

Status: **independent execution plan**

Execution status: **S1 complete; S2 has 11 accepted policies and 2 in remediation**

Local evidence at commit `e7b4f50`:

- live registry resolved all 13 initial tasks;
- CPU suite: 157 passed, 1 hardware-dependent skip;
- every initial task completed the required 64-env, 5-iteration CUDA smoke
  test with a 61D actor and zero NaN terminations;
- the smoke matrix exposed and fixed RollerStandUp's raw-joint/servo-index
  mismatch in commit `e7b4f50`;
- representative plain and roller checkpoints exported through
  `scripts/export.py`; both ONNX graphs are `61 -> 14`, produce finite CPU
  actions, and contain the baked normalizer operations;
- clean image `microduck-rl:specialist-s0` built and passed a fresh container
  smoke test. Local digest:
  `sha256:246aadb0cc9c2cda0f54dba9b97f7a007d088adb59874f18b2b1395ea788aa32`.

S1 used immutable source snapshot `652b7ce-20260829T140000Z`. All 13 jobs
completed successfully, and every final checkpoint was verified in its unique
JuiceFS lineage. The resolved S2 inputs are recorded in
`cloudml/specialist-final-checkpoints-652b7ce.json`.

S2 accepted Velocity plus 10 policies from the `3b44c25` evaluation batch.
VelStand was rejected after 12/32 `fallen_too_long` terminations (success rate
0.625), and RollerSlope after 22/32 `fell_over` terminations (success rate
0.3125). Both remain in scope for checkpoint audit or retraining; their failed
reports and videos are retained rather than weakened by threshold changes.

This plan is deliberately separate from the Generalist distillation research.
It produces stable specialist artifacts and a long, reviewable demo. It does
not require implementing a 71D Generalist schema, collecting distillation
data, or changing the production 61D policy contract.

## Scope

Train each selected official task independently, evaluate it in MuJoCo, export
the accepted checkpoint to ONNX, and run a deterministic multi-policy episode
with legal policy switches. The same scenario drives the PyTorch (`.pt`) and
ONNX paths so differences are attributable to deployment/export rather than
to different commands or random seeds.

Initial task set:

```text
Track A (baseline/teacher):
Velocity-Flat, VelStand-Flat, SitStand-Flat, GroundPick-Flat,
BallKick-Flat, Roulade-Flat

Track B (showcase, does not block Track A):
StandUp-Flat, Velocity-Rollers, Swizzle, RollerCrouch,
RollerSlope, RollerStandUp, Spin
```

Rough and Backlash variants are a later robustness/A-B pass. Left kick is
added only if the right-kick teacher and mirror-data check justify it.

## Independent stages

### S0: local and container preflight

- Run `list-envs` and the full CPU test suite.
- Run the mandatory 64-env, 5-iteration smoke test for every task selected for
  the first wave.
- Build the Docker image from a clean `uv sync`; verify MuJoCo, CUDA, task
  registration, 61D observations, NaN guard, and ONNX export.
- Record image digest, repository commit, and source snapshot.

### S1: parallel specialist training

Use one immutable JuiceFS source snapshot and one output prefix per task.
P0 calibrates one gait and one episodic task. After P0 passes, submit Track A
and Track B in independent waves:

- target up to 8 guaranteed single-GPU R49 jobs;
- use additional best-effort capacity only when queue/quota checks pass;
- bound concurrency and monitor every 30 minutes;
- resume preempted jobs only from verified checkpoints;
- never overwrite a completed teacher lineage.

Each task has its own checkpoint, metrics, failure note, and acceptance result;
one failed task does not block unrelated tasks.

### S2: per-policy evaluation and acceptance

For each accepted checkpoint, run a fixed-seed MuJoCo battery appropriate to
the task. Record success rate, reward terms, episode length, falls/contact
failures, and representative 10--20 second diagnostic clips. A policy is
accepted only when the main task metric grows, all penalty terms are non-
positive, rollouts are finite/NaN-free, and the video matches the metric.

### S3: export and parity

- Export with `scripts/export.py`, which bakes observation normalization.
- Store `.pt`, ONNX, metadata, and SHA-256 hashes together.
- Compare PyTorch and ONNX actions on deterministic golden observations,
  including command extremes and zero-command cases.
- Run deployment-style inference with the exact 61D command-slot semantics.

### S4: integrated switching demo

Use one canonical scenario file containing seed, duration, commands, switch
times, policy IDs, and expected outcomes. Produce:

- one 60--90 second Track A reel;
- one 60--90 second Track B reel when its policies are available;
- both `.pt` and ONNX versions where practical;
- an HTML gallery with per-policy clips, integrated reels, metrics, hashes,
  and failure notes.

The scenario must test legal transitions such as stand -> locomotion -> stop,
stand -> sit/rise, ground-pick -> stand, kick -> stand, and roulade -> stand.
Unsupported transitions are listed explicitly rather than silently attempted.

## Deliverables and completion gate

The handoff is validated offline with:

```bash
python scripts/validate_specialist_artifacts.py \
  docs/specialist_artifact_manifest.example.json \
  docs/specialist_demo_scenario.json
```

Copy the example manifest, replace artifact paths and SHA-256 values after each
accepted export, then run the same command before producing the gallery. The
scenario file is the single source for seed, timing, and legal/unsupported
switches; runners must consume it rather than embedding a second schedule.

The specialist track is complete when every included task has:

1. an accepted checkpoint and immutable manifest;
2. a deterministic evaluation JSON report and diagnostic video;
3. an exported ONNX with matching metadata and parity report;
4. inclusion in the long switching demo, or an explicit unsupported reason;
5. artifacts copied to the agreed JuiceFS output tree and indexed by the HTML
   gallery.

This completion gate is independent of Generalist work. The resulting
checkpoint/ONNX manifest becomes the teacher/fallback input to
[`generalist_policy_v0_experiment_plan.md`](generalist_policy_v0_experiment_plan.md).

## Out of scope for this plan

- 71D Generalist observation/schema implementation;
- teacher rollout datasets, Behavior Cloning, or DAgger;
- multi-task PPO, MoE, or transition-curriculum training;
- hardware deployment or runtime repository changes.
