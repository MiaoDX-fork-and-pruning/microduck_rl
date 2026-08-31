# Specialist policy demo active run

Status: `ACTIVE`
Source plan: `docs/specialist_policy_demo_plan.md`
Control plane: primary Codex session `01a04d9e-be91-7531-8041-52620c432d61`
Latest intent: continue the approved plan through intuitive-flow

Current slice: S1 is complete; implement and run S2 fixed-seed per-policy
evaluation, acceptance reports, and diagnostic videos.

Last proven evidence:

- all 13 CloudML training jobs reached `succeed` with final `nan_state=0`;
- all 13 expected final checkpoint filenames have exactly one JuiceFS hit;
- every checkpoint probe completed without truncation;
- resolved task, job, output, run prefix, and checkpoint inputs are frozen in
  `cloudml/specialist-final-checkpoints-652b7ce.json`;
- the offline artifact contract, ONNX comparator, canonical scenario runner,
  and gallery builder already have focused tests.

Completed slices: S0 host/container proof; 13-task smoke matrix; immutable
source/image package; P0/A1/B1 training waves; final checkpoint inventory;
canonical scenario and manifest validator; ONNX parity helper; gallery and
deployment-style ONNX scenario route.

Next slice: land the real MuJoCo S2 evaluator, verify it locally, then render
and submit bounded evaluation jobs against the frozen checkpoint inventory.

Next proof: focused evaluator tests and CLI proof, followed by one live
fixed-seed evaluation report/video before scaling to all 13 policies.

Stop condition: do not mark a checkpoint accepted from training reward alone;
accept only finite rollouts with non-positive penalties and video review that
matches the task metric. Failed evaluations remain isolated and do not mutate
the completed training lineages.

No-touch scope: Generalist 71D schema/distillation, runtime repository, robot
hardware, Rough/Backlash variants, and left-kick expansion.

Parked work: Track A/B integrated reels and gallery wait for S2 acceptance;
Generalist teacher ingestion waits for the completed specialist manifest.
