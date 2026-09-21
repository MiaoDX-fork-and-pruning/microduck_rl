# MJLab Adaptive Curriculum v2

Updated: 2026-09-21. **First bounded Adaptive-Feedback run completed; native
and CPU capability checks failed. Policy acquisition and calibration remain
open; no usable adaptive policy is established.** The broader goal is a policy that matches or beats the fixed
recipe at matched budgets on a held-out final distribution.

## Current run

- Ran 2026-09-21 16:07–16:24 CST on the local RTX 3090 (17.4 minutes
  including smoke and evaluation). Launcher 616307 has exited; GPU is idle.
- Task `Mjlab-Velocity-Flat-Adaptive-Feedback-MicroDuck`, seed 17,
  4096 environments, 500 PPO updates (49,152,000 training transitions).
- All four native feedback windows at updates 125/250/375/500 completed with
  valid reports. Gate seed 20260815; held-out seed 20260915. Training had no
  traceback; the final NaN termination metric was zero. The budget ended at
  `model_499.pt`, followed by native held-out evaluation and CPU rehearsal.
- Both final reports are valid with aggregate score 0.0 and no passing bucket.
  See `/tmp/microduck-adaptive-formal-20260921-s17/review-500.json` for the
  window-by-window evidence and `observed-status.json` for completion status.
- Immutable source contents copied to
  `/tmp/microduck-adaptive-formal-20260921-s17/source/`. Verified the training
  package imports from this snapshot. Provenance:
  `snapshot-sha256:9645b5234f14ac68c2597307e39cbd9a80a728ae2ed97d23efdfe6fc5df8e6dc`.
  `source-manifest.json`, `source.patch`, `environment.json` and `launch.json`
  sit alongside it. The validated local environment is reused; this is not a
  fresh-install or CloudML portability claim.
- Logs/results: `/tmp/microduck-adaptive-formal-20260921-s17/run/`.
  `training.log`, `training-result.json` and `campaign-result.json` are complete.
  Normalizer-baked ONNX export and CPU rehearsal artifacts are under
  `run/cpu-transfer/`. No continuation has been launched.

The final native six-bucket test survived all six 6-second rollouts. Tracking
and idle behavior remain below acceptance:

| Held-out measure | Observed | Required |
| --- | ---: | ---: |
| Zero-command endpoint drift | 0.1057 m | <=0.012 m |
| Forward tracking MAE | 0.0472 m/s | <=0.024 m/s |
| Lateral tracking MAE | 0.1126 m/s | <=0.024 m/s |
| Yaw tracking MAE | 0.6560 rad/s | <=0.12 rad/s |
| Moving left-turn yaw MAE | 0.5216 rad/s | <=0.12 rad/s |
| Moving right-turn yaw MAE | 0.3984 rad/s | <=0.12 rad/s |

Native traces show mean forward velocity 0.0729 m/s and mean lateral velocity
0.0330 m/s for respective 0.12 m/s commands. These are trace measurements;
no video-based motion-quality acceptance has been made for this checkpoint.

The controller did change exposure: forward 13.33% -> 11.87%, lateral
13.33% -> 13.69%, with nominal coverage retained at 20%. Both CoM axes stayed
at their initial +/-3 mm stage. At update 125 the zero bucket scored 0.8515
(drift 0.0089 m). Subsequent gate drifts were 0.0609/0.1511/0.1070 m and
triggered three preservation failures. No rollback occurred because no
checkpoint passed all six buckets, so `last_known_good_checkpoint` stayed null.

This is direct evidence of the acquisition-stage preservation gap: degradation
is detected before a global known-good target exists, but the current policy
has no corrective rollback at that stage. Sampling feedback alone has not yet
resolved it. The reward basin and push/drift metric semantics also remain open.
Prioritize acquisition-stage capability preservation and evaluator calibration
before extending the budget. A single 500-update seed does not establish that
adaptive training is better or worse than the fixed recipe.

## Current control path

`run_adaptive_campaign_job.py` launches a fresh smoke, then a training segment,
then native held-out evaluation and a separate CPU/ONNX transfer rehearsal.
`AdaptiveMicroduckOnPolicyRunner` is the only owner of curriculum decisions,
command exposure, evaluation history, trainer RNG, resume progress and rollback.

- New task: `Mjlab-Velocity-Flat-Adaptive-Feedback-MicroDuck` (`--branch feedback`).
  It adds command exposure feedback to the composed CoM/head-CoM gate.
- `CommandExposure` allocates a floor of 10% to each of six capability buckets,
  keeps 20% nominal continuous commands, and distributes the remaining 20%
  according to `1 - bucket_score`. Each validated window moves 25% toward that
  target; no bucket changes by more than five percentage points per window.
  Equal deficits retain a balanced mixture. Pure lateral/yaw commands include
  both signs; moving turns have explicit left/right buckets. Exact zero remains
  a sampled command, and nominal samples retain backward/diagonal support.
- The live CommandManager term consumes the allocation on its next resample.
  The competing standing-fraction curriculum is removed only for Feedback.
  Other canonical reward/pose schedules remain. Reward adaptation, adaptive
  regularization/push strength and the optional advisor are not implemented by
  this refactor.
- Invalid reports do not change gate or sampler state. Mastered-bucket drops
  are checked even when the aggregate is below the pass threshold. With a
  recorded known-good checkpoint, rollback restores policy/optimizer/RNG/gate/
  sampler while retaining chronological evaluation history and consumed budget.
  Before any known-good policy exists, sampling can still respond to deficits.
- Training checkpoint metadata has one serialization path for both adaptive
  and static recipes. Resume restores completed updates and the environment
  curriculum clock. Actor-only evaluation/export does not restore trainer
  state. Episodes start fresh on training resume; physics state is not a
  bitwise continuation of an uninterrupted rollout.
- `training-result.json` names and hashes the final checkpoint and records the
  segment/cumulative update counts. The launcher consumes this manifest instead
  of guessing the last iteration label. Evaluated checkpoints remain immutable;
  post-decision state is saved separately.
- All eleven historical task IDs remain registered through one recipe table
  for artifact replay. Standing/ActionRate now record their actual task IDs.
  They are diagnostic recipes, not eleven distinct feedback strategies.
- Removed `run_adaptive_windows.sh`, `advance_adaptive_curriculum.py`, and the
  stage-file override. Submitted Phase 0 YAML evidence is archived under
  `cloudml/archive/adaptive-phase0/`, tied to its historical source snapshots.
  The obsolete `/tmp/adaptive-push-followup.sh` waiter was stopped: its training
  finished at `model_1998.pt`, not the awaited `model_1999.pt`.

## Launch and resume

One bounded pilot entry point, after review of the source revision:

```bash
MICRODUCK_SOURCE_SHA="$(git rev-parse HEAD)" uv run python scripts/run_adaptive_campaign_job.py \
  --branch feedback --seed 17 --output /tmp/adaptive-feedback-pilot \
  --iterations 500 --num-envs 4096 --gate-interval 125
```

`--iterations` is the **total** PPO update budget. To continue to 1000, use a
new output directory, `--iterations 1000`, and `--resume /absolute/path/model_499.pt`
from the prior result manifest. Preserve `--num-envs` and `--gate-seed`.
The launcher always runs the 64-env/5-update smoke before the training segment.
It refuses to mix runs in an existing training directory.

Audited launcher resume requires current runner metadata with task ID,
completed updates and environment count. Old checkpoints remain evaluable;
ambiguous historical iteration labels are not accepted as a matched-budget
resume record. The unchanged canonical `fixed` branch supports fresh baseline
runs through this launcher, not audited continuation.

`campaign-result.json` keeps native usability and CPU transfer verdicts
separate. `status: evaluated` means evaluation completed, not policy success.
Source labels for the refactor verification identify an uncommitted working
tree (`28e5372+adaptive-refactor-20260921`), not a new source commit.

## Evidence and unresolved learning questions

All historical diagnostic endpoints below are valid native schema-v2 reports
with aggregate score 0.0; those diagnostic jobs have finished.

| Recipe / endpoint | Native report | Main observation |
| --- | --- | --- |
| Repaired static, 2000 updates | `/tmp/adaptive-repaired-2000/native-1999/native_capability.json` | Lateral error 0.1193 m/s; survival alone is insufficient. |
| Lateral diagnostic | `/tmp/adaptive-lateral-pilot/native-2000/native_capability.json` | Lateral mean velocity collapsed to 0.0008 m/s. |
| Staged acquisition, 2000 updates | `/tmp/adaptive-acquisition-pilot/native-1998-resume/native_capability.json` | Forward error 0.0378 m/s; lateral error 0.1602 m/s. |
| 50% lateral acquisition, 500 updates | `/tmp/adaptive-acquisition-lateral-pilot/native-499/native_capability.json` | More lateral motion, but worse preservation; exposure alone did not solve acquisition. |
| Push diagnostic, 2000 updates | `/tmp/adaptive-push-pilot/native-1998-resume/native_capability.json` | Drift 0.0729 m; lateral error 0.1192 m/s. |

The r4 campaign completed all 15 jobs; the nine adaptive jobs each evaluated
16 windows and remained at stage 0. Its causal conclusion is inconclusive
because curriculum ownership, consumed evaluation seeds and actuator alignment
were subsequently repaired. The isolated native fixed/static ladder at
`/tmp/adaptive-native-isolated-ladder/` has 16 valid seed17/23 checkpoint reports
at 500/1000/2000/3999, all with lower-tail score zero.

Two measurement/learning issues remain material:

1. At linear tracking std `sqrt(0.1)`, standing earns about 86.6% of maximum
   tracking reward for a 0.12 m/s command. The Feedback recipe deliberately
   isolates exposure from reward changes; its tests do not prove it can escape
   that reward basin. Capability-driven reward shaping needs measured design
   and a separately reviewed experiment.
2. Zero drift includes push displacement. Paired replay under
   `/tmp/adaptive-push-paired/v3/` found 0.0671 m post-kick offset versus
   0.00258 m without the kick while both final speeds were about 0.001 m/s.
   The actor has no absolute position input. This metric is retained for
   compatibility, but is not a clean measure of ongoing idle motion.

Product thresholds are unchanged: score >=0.8 implies linear MAE <=0.024 m/s,
angular error <=0.12 rad/s, zero drift <=0.012 m and tilt <=7 degrees.
Calibration/bound manifest and a usable native policy remain prerequisites for
any claim of improvement. Do not launch another five-branch/multi-seed campaign
until bounded acquisition evidence supports it. Generalist and IsaacLab work
remain separate.

## Verification for this refactor

- Focused adaptive/capability/canonical-config tests: 126 passed. Coverage
  includes actual bucket sampling/floors, invalid-report no-op, live manager
  updates, actor-only load isolation, checkpoint/RNG restoration, preservation
  rollback and explicit launcher/resume budget accounting.
- `/tmp/adaptive-feedback-refactor-20260921-r2/`: 64-env/5-update smoke and
  a 5-update runner job completed, including a real native feedback window,
  held-out native report and normalizer-baked ONNX export via `scripts/export.py`
  followed by CPU rehearsal. The capability verdict remains negative.
- `/tmp/adaptive-feedback-refactor-20260921-resume/`: exact checkpoint resume
  from `model_4.pt` ran five further updates and produced `model_9.pt` with
  `completed_iterations=10`, `env_step=240`, two feedback windows at steps
  120/240, and cumulative training transitions 15,360 (smokes excluded).
  Native and CPU reports were valid, with negative usability verdicts.
- Six native/ONNX parity rollouts (30 steps each) retained finite 61D/14D
  tensors; every bucket passed, maximum absolute action difference 1.05e-7.
  Evidence: `.../onnx-parity/native_capability.json` in the resume directory.
- Focused Ruff and `git diff --check` pass. Shared `mdp.py`, task registrations
  and exporter retain their pre-existing lint diagnostics; no new diagnostics
  were introduced in changed lines. The first smoke caught the heading cfg
  constraint before iteration 0; the corrected fresh smoke and resume passed.

Canonical velocity source and product scoring are unchanged. Preserve the
pre-existing IsaacLab, evaluator and planning edits in this shared worktree.
