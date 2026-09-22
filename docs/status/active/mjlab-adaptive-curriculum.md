# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
The user authorized sustained necessary changes, training and verification through
`intuitive-flow`; the latest status request does not cancel execution.
Project-status writer: not assigned; project-status delta: none.

## Current state

Phase 0/1 automation is implemented; Phase 2A usable-policy acquisition remains
open. The seed-17 relief campaign completed 6250→6750 updates at 4096 envs.
The 6750 candidate lost turn preservation and was automatically rejected; the
final checkpoint retains the 6500 actor. Consumed budget remains 6750.
All 13 actor-state tensors equal the known-good snapshot and differ from the
rejected actor. Both adaptive CoM axes remain at stage 0 (±3 mm).

Scores below are normalized capability, not success rates. **Every bucket must
reach .80.** The seed sets and distributions are deliberately labelled separately.

| Retained 6500 actor evaluation | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, seeds 20260815–17 | .938 | .735 | .796 | .669 | .830 | .812 |
| Final native, seed 20260915 | .923 | .879 | .797 | .837 | .737 | .820 |
| Stage diagnostic, worst of 20260915–24 | .914 | .696 | .298 | .146 | .707 | .767 |
| Corrected CPU rehearsal, seed 20260915 | .940 | .903 | .727 | .000 | .696 | .849 |

No diagnostic seed passes all six buckets. Yaw passes 3/10; lateral passes 3/10
and falls on seed 20260916. Worst yaw seed 20260919 averages .360 rad/s for a
.8 command; the first four seconds include under-speed and a brief reverse,
then the last two seconds reach approximately .86. CPU yaw is a different
failure: average **1.390 rad/s** for the same .8 command (native final: .843),
with samplewise error .655 exceeding the .6 cap. Do not label it idle.

All 60 native/ONNX action parity cases pass (maximum absolute error 1.67e-6).
All cohort member reports, trace hashes and source-manifest files were verified.
Parity confirms the export on identical inputs; it does not explain the
native/CPU physics and reset-distribution difference.

Authoritative artifact:
`/tmp/microduck-adaptive-action-rate-s17-500/experiment-summary.json`.
It records exact checkpoint, report, trace, ONNX and source hashes, retained vs
rejected evidence, and the matched first-250-update comparison. The matched
control's stage yaw is .353 versus relief .669 on the same gate seeds. Both
fail mastery; the relief campaign consumed 500 updates in total, not 250.
This single training seed establishes neither reproducibility nor superiority.

## Implemented slice and proof boundaries

- `ef4734e`: bounded adaptive action-rate relief, enabled only in LateralDrive.
  Worst yaw/turn capability below .55 activates weight -.2; release requires
  .80 mastery or four gate windows, followed by cooldown. Training confirms
  the live -.2 override, and the campaign's smoke64/5 passed.
- `75f479f`: relief cannot strengthen an earlier smaller canonical penalty;
  legacy loads clear stale relief; rejected candidates cannot release it;
  full resumes cannot silently discard the enabled controller. Focused proof:
  83 tests, then 6 relief tests after constructor validation, adaptive Ruff and
  diff checks. Full `mdp.py` Ruff has 17 pre-existing findings.
- Training used `ef4734e`, not the later fixes. Source snapshot
  `/tmp/microduck-adaptive-action-rate-source-ef4734e` has 1001 verified files,
  including unrelated copied files and the existing CPU seed overlay. Its
  recorded manifest hash is `4dd1470ff30c1cc4f136737b8951b008606e3ed56b7b6f13933ce7a1d2bab5f4`.
- Current diagnostic source is a clean `git archive 75f479f` plus only the
  explicit CPU seed overlay: `/tmp/microduck-adaptive-relief-source-75f479f`,
  680 verified files, manifest file SHA
  `29a3eecea543830a2b6af0246f671c9eacb47f57ae6cca9a0ca1df69abdcf9bf`.
  No long training has run on this newer source; smoke64/5 is still required.
- Earlier infrastructure proof: 269 relevant tests and a fresh non-editable
  locked install/config/14-actuator model compile pass. Fresh-environment GPU
  training and remote execution remain unproven. CPU link-frame correction is
  `931e3c9`; old principal-inertia-frame tracking scores are superseded.

## Current diagnostic and next decision

Blocker: `reset_conditioned_acquisition_and_cpu_yaw_rate_transfer`.
Global smoothing relief improves native yaw acquisition but does not repair
reset robustness, lateral survival, or CPU rate transfer; do not blindly extend
it or repeat the failed direct-yaw/turn-proxy exposure budgets.

The earlier 6250 actor white-noise ablation is complete at
`/tmp/microduck-obs-sensitivity-6250/probe.json`. Removing additive white noise
did not rescue the failed seed. Fixed IMU misalignment and encoder bias were
unchanged, so the probe does not exclude persistent sensor calibration effects.

The fixed-bias comparison **completed**, terminal session `61738` exited 0.
Evidence: `/tmp/microduck-relief-6500-sensor-bias/analysis-summary.json`.
Twelve evaluations cover seeds 20260915/16/19 × nominal, zero encoder bias,
identity IMU misalignment and both nominal. All 72 trace hashes pass; six
recorded physical reset fields match, unchanged sensor fields match, and
white-noise settings are identical. Nominal scores reproduce the prior cohort.

Seed 20260919 yaw rises from .146 to .807 without encoder bias (mean rate
.360→.830 rad/s); identity IMU alone leaves .115. Seed 20260915 yaw rises
.744→.859 with identity IMU. Seed 20260916 lateral still falls in all four
conditions. These are diagnostic interventions, not product acceptance.

The push ablation is complete:
`/tmp/microduck-relief-6500-push-sensitivity/analysis-summary.json`.
The failing lateral rollout receives a world-frame velocity increment
(+.284, -.295) m/s at step 195 (3.9 s). Half/no push survives with scores
.801/.807, compared with .298 at full push. Initial sensor/physical fields and
pre-push trace prefixes match. The matched 6500 canonical-smoothing control
also survives full push (.790), with identical recorded initial state and raw
actor observation. The earlier probe's immediate root-link velocity delta was
cached and invalid; the separate `push-write-proof` qvel trace supplies the
actual write. This supports testing narrower relief, not weakening product pushes.

## Current implementation and experiment

The next treatment is **pure-yaw-only action smoothing relief**. The opt-in
`MICRODUCK_ADAPTIVE_ACTION_RATE_RELIEF_SCOPE=pure_yaw` keeps canonical action-rate
costs for idle, forward, lateral and mixed turns; only zero-planar/nonzero-yaw
commands receive the bounded relief. Trigger/release use pure-yaw capability.
Scope is checkpointed, inherited by direct/campaign resumes, and checked on
load. Legacy v1 controllers retain global scope; evaluators strip the launch
setting. Default LateralDrive remains global until behavioral evidence supports
promotion. No observation, action, DR, push or acceptance change is included.

Proof: 97 focused adaptive/config/cohort/campaign tests passed, then 13 campaign
tests passed after adding the scope-inheritance regression. Focused Ruff passes.
New-source smoke and training are next; no training process is currently running.

Bounded comparison: resume the exact original 6250 `model_6249.eval.adaptive.pt`
from the stage-gate campaign, seed 17, 4096 envs, 250 additional updates to 6500,
stage gate seeds 20260815–17 and diagnostic final seed 20260915. The source has
no relief controller, so this is an explicit treatment bootstrap, not a silent
scope change on a saved controller. Compare existing no-relief and global-relief
250-update controls, then the seed-20260916 full-push case and retained native/
CPU traces. A useful treatment must retain yaw learning without the lateral
recovery regression; full product acceptance still requires all six buckets.
Use an immutable source plus the explicit CPU seed overlay, smoke64/5 before
long training, and verify live per-command weighted costs. Do not simply
extend a failing treatment. Sensor robustness and CPU overspeed remain open.

## Remaining acceptance and boundaries

Required: retained six-bucket native mastery at final ranges, a fresh held-out
seed set after diagnostics, rollout/video inspection, normalized ONNX and CPU
rehearsal, the same autonomous procedure across training seeds 17/23/47, canonical
final-distribution fine-tuning and matched-budget fixed/axis comparisons.
No hardware or adaptive-superiority claim exists. No external blocker prevents
the next required diagnostic; the goal is neither complete nor blocked.

Preserve canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward
signs, .80 thresholds and exact-zero/nominal anchors. Keep native and CPU proof,
and rejected and retained actors, separate. Preserve the pre-existing CPU seed
patch in `scripts/run_specialist_action_battery.py`. IsaacLab, `uv.lock`,
generated files and unrelated processes are outside this task. Advisor,
stronger teachers, generalist and hardware deployment remain parked.
Use the validated local venv and unique `/tmp` working directories; the
worktree's `logs/rsl_rl` is not writable. Do not change its ownership.
