# Specialist policy demo active run

Status: `ACTIVE`
Source plan: `docs/specialist_policy_demo_plan.md`
Control plane: primary Codex session `01a04d9e-be91-7531-8041-52620c432d61`
Latest intent: continue the approved plan through intuitive-flow

Current slice: S2 accepted 11 of 13 policies. VelStand and RollerSlope
remediation jobs are running in isolated new lineages.

Last proven evidence:

- all 13 CloudML training jobs reached `succeed` with final `nan_state=0`;
- all 13 expected final checkpoint filenames have exactly one JuiceFS hit;
- every checkpoint probe completed without truncation;
- resolved task, job, output, run prefix, and checkpoint inputs are frozen in
  `cloudml/specialist-final-checkpoints-652b7ce.json`;
- evaluator/renderer contract tests pass (`21 passed`), including a real CUDA
  tensor finite/NaN check and threshold reassessment from raw episodes;
- all 13 task configs resolve their active reward terms and episode durations;
- Velocity's 32-episode report is accepted after human review: success rate
  0.8125, main metric 24.5030, all penalties non-positive, finite rollouts;
- short-task EGL smoke produced a decodable 750-frame / 15-second BallKick
  clip across two resets while retaining exactly two original metric episodes;
- immutable source `/source/3b44c25-20260831T1114` uploaded successfully
  (273 files, 28,987,837 bytes);
- all 12 output prefixes were materialized and all 12 effective CloudML jobs
  succeeded; one BallKick attempt was preempted and retained before a successful
  retry;
  job IDs and output prefixes are frozen in
  `cloudml/specialist-s2-evaluation-jobs-3b44c25.json`;
- 10 new reports are accepted after task-specific threshold reassessment and
  video review; combined with Velocity, S2 now has 11 accepted policies;
- VelStand is rejected at 0.625 success (12/32 `fallen_too_long`) and
  RollerSlope at 0.3125 success (22/32 `fell_over`); all reports remain finite
  with non-positive penalties and are published back to JuiceFS;
- reviewed thresholds and rationale are frozen in
  `cloudml/specialist-s2-thresholds-3b44c25.json`.
- formal re-evaluation rejected VelStand `model_4000.pt` at 0.6875 success;
  RollerSlope checkpoints 7250/7500/7750 reached only
  0.1875/0.3125/0.28125, so neither lineage has a defensible replacement;
- RollerSlope per-difficulty evaluation proved a narrow middle-band policy:
  success was 0.2188/0.6562/0.5312/0.1875/0.0 from difficulty 0.0 to 1.0;
- commit `facd4f4` makes RollerSlope resample all ten slope levels every
  episode, matching the S2/play distribution; focused tests pass (41 passed),
  and the 64-env/5-iteration smoke kept terrain level near the uniform mean
  with a 61D actor and zero NaN terminations;
- VelStand also passed its required 64-env/5-iteration smoke; its configured
  20,000-iteration budget was previously overridden to 6,000, so continuation
  is prepared without another reward change;
- immutable source upload dry-run for `facd4f4-20260831T1152` passed: 288
  files, 29,026,755 bytes; two isolated GUARANTEED R49 continuation YAMLs are
  ready under `cloudml/generated/`.
- source snapshot and both `_READY` output markers were uploaded; VelStand job
  `t-20260831123941-k7hts` and RollerSlope job `t-20260831124001-s8mmr` are
  both `running` on queue `11759`, with the expected read-only checkpoint,
  source, and writable output mounts; job inventory is frozen in
  `cloudml/specialist-s2-remediation-jobs-facd4f4.json`.
- subsequent log verification shows both bootstrap loads explicitly
  (`model_5999.pt` and `model_7000.pt`); latest sampled logs reached VelStand
  6108/19999 and RollerSlope 7111/12000 with mean rewards 64.20 and 53.44,
  and no traceback or NaN failure.
- the offline artifact contract, ONNX comparator, canonical scenario runner,
  and gallery builder already have focused tests.

Completed slices: S0 host/container proof; 13-task smoke matrix; immutable
source/image package; P0/A1/B1 training waves; final checkpoint inventory;
canonical scenario and manifest validator; ONNX parity helper; gallery and
deployment-style ONNX scenario route.

Next slice: monitor both remediation jobs to terminal state and verify their
bootstrap checkpoints loaded before running the fixed S2 battery.

Next proof: CloudML logs must prove the requested bootstrap checkpoint loaded;
after training, termination-aware 32-episode reports/videos for both replacement
checkpoints must each reach success rate >= 0.8.

Stop condition: do not mark a checkpoint accepted from training reward alone;
accept only finite rollouts with non-positive penalties and video review that
matches the task metric. Failed evaluations remain isolated and do not mutate
the completed training lineages.

No-touch scope: Generalist 71D schema/distillation, runtime repository, robot
hardware, Rough/Backlash variants, and left-kick expansion.

Parked work: Track A/B integrated reels and gallery wait for S2 acceptance;
Generalist teacher ingestion waits for the completed specialist manifest.
