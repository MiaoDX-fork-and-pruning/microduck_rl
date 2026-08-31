# Specialist policy demo active run

Status: `ACTIVE`
Source plan: `docs/specialist_policy_demo_plan.md`
Control plane: primary Codex session `01a04d9e-be91-7531-8041-52620c432d61`
Latest intent: continue the approved plan through intuitive-flow

Current slice: S2 evaluator is implemented and locally verified; the first
Velocity pilot exposed and fixed a mujoco-warp CUDA tensor-view bug, and the
repinned retry is ready for submission.

Last proven evidence:

- all 13 CloudML training jobs reached `succeed` with final `nan_state=0`;
- all 13 expected final checkpoint filenames have exactly one JuiceFS hit;
- every checkpoint probe completed without truncation;
- resolved task, job, output, run prefix, and checkpoint inputs are frozen in
  `cloudml/specialist-final-checkpoints-652b7ce.json`;
- evaluator contract tests pass (`16 passed`), including a real CUDA tensor
  finite/NaN check; CLI help and compilation pass;
- all 13 task configs resolve their active reward terms and episode durations;
- pilot YAML is prepared at
  `cloudml/generated/microduck-specialist-s2-velocity-pilot-e76b22f.yaml`;
- CloudML read-only checks show 114 free GUARANTEED R49 GPUs and no remaining
  workspace guaranteed quota allocation.
- local EGL smoke completed with a real checkpoint: one finite rollout, reward
  terms and `fell_over` termination captured, and a 10-frame MP4 written;
- final JuiceFS upload dry-run passed for `9dece54`: 268 files and 28,968,051
  bytes planned, zero uploaded, with the exact pilot source prefix.
- the offline artifact contract, ONNX comparator, canonical scenario runner,
  and gallery builder already have focused tests.

Completed slices: S0 host/container proof; 13-task smoke matrix; immutable
source/image package; P0/A1/B1 training waves; final checkpoint inventory;
canonical scenario and manifest validator; ONNX parity helper; gallery and
deployment-style ONNX scenario route.

Next slice: upload the immutable `e76b22f` source snapshot and submit the
bounded Velocity pilot after the CloudML confirmation gate; then review its
report/video before rendering the 12 remaining evaluation jobs.

Next proof: one live fixed-seed Velocity evaluation report/video, followed by
human review and threshold calibration before scaling to all 13 policies.

Stop condition: do not mark a checkpoint accepted from training reward alone;
accept only finite rollouts with non-positive penalties and video review that
matches the task metric. Failed evaluations remain isolated and do not mutate
the completed training lineages.

No-touch scope: Generalist 71D schema/distillation, runtime repository, robot
hardware, Rough/Backlash variants, and left-kick expansion.

Parked work: retry after the failed `t-20260831104511-dagok` pilot; Track A/B
Track A/B
integrated reels and gallery wait for S2 acceptance; Generalist teacher
ingestion waits for the completed specialist manifest.
