# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: continue authorized changes, training and checks via intuitive-flow.
Execution is local in the main session; other worktree changes are not owned.

## Current evidence and decision

Runner evaluation, bounded command exposure, checkpoint/resume and retention
rollback operate. Both CoM axes remain at ±3 mm. Formal training seeds 17/23/47
remain gated on a retained six-capability policy; no video/hardware usability or
adaptive superiority is established.

`e86a1f4` isolates training transition overrides from evaluators. The validated
zero-bootstrap campaign is `/tmp/microduck-adaptive-zero-transition-s17-5000-e86/`.
Its same-policy variance battery has yaw scores
`[.809, .000, .776, .000, .000, .595, .555, .842, .000, .000]` for base seeds
20260815–20260824: five zero scores and only two passes at 0.80. Earlier status
prose claiming six failures was imprecise. These are diagnostic reset/DR seeds,
not independent training seeds. Scalar friction correlation was about 0.13 and
does not isolate a causal DR field.

The matched 5000→5500 reward comparison is in
`/tmp/microduck-adaptive-yaw-planar-s17-5500/clean-control/` and
`clean-treatment-v2/`. Control native held-out scores were
zero .931 / forward .842 / lateral .795 / yaw .442 / left .754 / right .754.
The treatment did not produce a retained usable policy and its held-out yaw
score was zero. Its retained-policy score must not be confused with a direct
measurement of the pre-rollback candidate. `ac21204` reverted the `1f17e81`
yaw-planar reward treatment; canonical reward behavior is restored.

`b942d44` replaces the training transition's bootstrap hold/jump with a linear
bootstrap→target command ramp. It attributes the ramp to the bootstrap bucket until
completion and preserves reset `dt=0` and partial-reset isolation. Evaluators still
use direct frozen commands. The bounded experiment is
`/tmp/microduck-adaptive-slew-s17-5500/experiment-contract.json`.

| Slew experiment (5000→5500, seed 17, 4096 envs) | Minimum score | Outcome |
| --- | ---: | --- |
| Pre-rollback candidate at 5250, native gate | .71733 | forward retention failure |
| Pre-rollback candidate at 5500, native gate | .72894 | forward/left retention failure |
| Retained policy, native held-out | .00000 | original 5000-update actor restored |
| Retained policy, CPU/XML transfer | .00000 | transfer fails |

Actor state including normalizer is tensor-identical to the starting checkpoint.
No retained learning gain is established. The campaign source is read-only
`/tmp/microduck-adaptive-slew-source-b942d44/`; 676 tracked file hashes are recorded.
`campaign-result.json` means evidence completed, not policy accepted.

## Current slice and next decision

Blocker fingerprint: `direct_command_yaw_under_reset_DR_robustness`.
Classification: command-initiation/retention depends strongly on reset/DR state;
continuous acquisition commands alone have not cleared the retained-policy gate.

The paired candidate/control probe is complete. Slew improved mean yaw from
about `.477` to `.646`, but only 2/10 reset/DR seeds passed yaw and 0/10 passed
all six buckets. The candidate was rolled back at both gate windows; no retained
learning gain is established. Startup sensor probes show a nonlinear failure
interaction between IMU misalignment and encoder bias.

The current implementation slice adds
`scripts/run_adaptive_native_cohort_battery.py`. It keeps the old single-seed
battery as the primitive, while an explicit cohort size runs fixed native seeds
and selects the worst score and raw evidence independently for each bucket.
The campaign launcher now opts into a three-seed gate cohort; held-out CPU and
native checks remain separate. Cohort metadata records all member reports and
the selected seed map, and the runner persists the cohort size in checkpoint
state. Member config/provenance and every member trace are checked before
aggregation, and the runner rechecks the returned seed map and worst-member
selection. Legacy checkpoints require an explicit migration flag; migration
preserves PPO state and cumulative budget while clearing old single-seed
mastery and rollback evidence.

The actual native proof at `/tmp/microduck-adaptive-cohort-proof/` produced a
schema-v2 report and passed the native trace validator. Its conservative
lower-tail score was `.72200` and `passed=false`, correctly exposing the current
policy's lack of robust mastery. A fresh real three-member native run at
`/tmp/microduck-adaptive-native-cohort-smoke/capability.json` also passed the
trace validator, with lower-tail `0.0` as expected from the 2-update smoke
policy. No training remains running.

Next action: use the cohort gate for the next changed training hypothesis and
then address the causal sensor-DR robustness failure. Do not lower product
thresholds, reuse a single lucky seed, or repeat the same 500-update slew budget.

## Proof and remaining gates

- Full adaptive/config suite: 207 passed, including cohort report, explicit
  migration, and runner wiring contracts. The fresh real native cohort smoke
  and the existing trace validator both passed; its policy score is negative
  evidence, not a usability result.
  Ruff retains
  the 17 pre-existing findings in mdp.py. No new findings in the changed slice.
- Transition-enabled smoke64/5 passed using TensorBoard in `/tmp`; the new
  cohort-size configuration also passed a fresh 64-env/5-iteration smoke. Actor 61D,
  action 14D, no NaN terminations. Repo `logs/rsl_rl` is not writable; W&B has no
  local key. Neither prevents local training/evaluation.
- `runtime-slew-proof.json`: 26/64 sampled transitions reached their exact target
  naturally in 52–96 environment steps; actor command parity held for 110 steps.
  This proves the ramp executes, not that the policy is usable.
- Still required: retained six-bucket native gate, fresh held-out reset/DR proof,
  rollout/video inspection, normalized ONNX and CPU deployment rehearsal,
  three independent training seeds 17/23/47, then matched-budget comparison.
- Fresh-sync portability remains unproven; current campaigns use the validated
  local venv and explicit `PYTHONPATH` pointing to immutable source.

## Boundaries and verification inventory

Preserve canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward signs,
0.5 s signed-EMA metrics, product thresholds and exact-zero/nominal anchors.
Leave IsaacLab, uv.lock, generated files and other processes untouched.
Advisor, stronger teachers, generalist, IsaacLab migration and hardware deployment
remain parked. The rejected 20% final-CoM rehearsal remains opt-in, not a default.

Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest pytest -q
 tests/test_adaptive_*.py tests/test_mjlab_adaptive_velocity_config.py`.
Campaign: `scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17
--output <fresh-dir> --iterations <cumulative-updates> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval 250 --gate-cohort-size 3`. Use a read-only source
snapshot and source manifest. Rehearsal/transition settings inherit the
checkpoint unless explicitly overridden; frozen evaluators disable acquisition
aids.
