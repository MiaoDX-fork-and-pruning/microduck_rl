# Specialist policy demo active run

Status: `ACTIVE`
Source plan: `docs/specialist_policy_demo_plan.md`
Control plane: primary Codex session `01a04d9e-be91-7531-8041-52620c432d61`
Latest intent: continue the approved plan through intuitive-flow

Current slice: Velocity is accepted and the remaining 12 S2 calibration
evaluations are running on CloudML from evaluator revision `3b44c25`.

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
- all 12 output prefixes were materialized and all 12 CloudML jobs created;
  job IDs and output prefixes are frozen in
  `cloudml/specialist-s2-evaluation-jobs-3b44c25.json`;
- CloudML read-only checks showed 72 free GUARANTEED R49 GPUs before submission.
- the offline artifact contract, ONNX comparator, canonical scenario runner,
  and gallery builder already have focused tests.

Completed slices: S0 host/container proof; 13-task smoke matrix; immutable
source/image package; P0/A1/B1 training waves; final checkpoint inventory;
canonical scenario and manifest validator; ONNX parity helper; gallery and
deployment-style ONNX scenario route.

Next slice: monitor the 12 evaluation jobs to terminal state, download their
reports/videos, derive defensible task-specific thresholds from the episode
distributions, reassess automatic gates, and review each diagnostic clip.

Next proof: 12 terminal CloudML jobs with a finite 32-episode report and valid
15-second MP4 in every output prefix.

Stop condition: do not mark a checkpoint accepted from training reward alone;
accept only finite rollouts with non-positive penalties and video review that
matches the task metric. Failed evaluations remain isolated and do not mutate
the completed training lineages.

No-touch scope: Generalist 71D schema/distillation, runtime repository, robot
hardware, Rough/Backlash variants, and left-kick expansion.

Parked work: Track A/B integrated reels and gallery wait for S2 acceptance;
Generalist teacher ingestion waits for the completed specialist manifest.
