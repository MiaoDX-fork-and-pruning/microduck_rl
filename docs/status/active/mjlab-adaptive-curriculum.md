# MJLab Adaptive Curriculum v2

- Status: **ACTIVE**, Phase 2A diagnostic repair. No usable adaptive policy has
  been established; the overall goal remains active.
- Owner: this adaptive curriculum session (`/root`). Latest user intent:
  continue all directly executable experiments without unnecessary pauses.
- Scope: [Phase 2A plan](../../plans/mjlab-adaptive-curriculum-v2/02a-diagnostic-repair-PLAN.md).
  Keep the canonical task and unrelated IsaacLab work unchanged.
- Current slice: validate native schema-v2 evaluator, regenerate the existing
  fixed/all-static checkpoint ladder, prove seeded replay and ONNX parity.
- No external blocker for these local experiments. Available local RTX 3090;
  r4 checkpoints are in `/tmp/r4-monitor/`.

## Evidence boundary

The old native v1 ladder, final reports, calibration and manifest are
**superseded**. Their boolean thresholds were not the canonical continuous
CapabilityReport v2 gate; seed labels did not prove applied reset noise.
The previous “3/4 native endpoints pass” and “manifest valid” claims are
withdrawn. The strict audit artifacts are `/tmp/adaptive-phase2a-audit-manifest.json`
and `/tmp/adaptive-phase2a-audit-calibration.json`, both incomplete.

Adaptive config ownership is repaired: only the selected CoM curricula are
removed; canonical standing/action-rate/command/pose curricula survive.
Ownership/runner tests pass 27/27; native/evidence tests pass 26/26.
Ownership repair is committed as `e22532e`.
Canonical source task is unchanged. These implementation checks alone do not
prove a working policy.

Native evaluator now uses the actual training MJLab/BAM task and baked actor
normalizer, consumes seeded reset/DR, preserves terminal states, and builds
canonical v2 scores from trace metrics including zero drift and tilt. It
asserts the 61D/14D ABI, frozen command slots and live reference CoM ranges.
Fresh-environment CUDA replay passes exactly across all 19 trace fields in
all six 300-step buckets (`/tmp/adaptive-native-isolated-replay/a` and `b`).
ONNX parity passes on both streams. Disjoint gate-seed run `c` is available
for consumed-state comparison. Targeted reset probe confirmed stale BAM
qfrc state and reward peak-height carryover; per-bucket environment construction
removes both without changing the canonical task. Evidence:
`/tmp/adaptive-bam-reset-probe/compute.json`.

Authoritative ladder is `/tmp/adaptive-native-isolated-ladder/` and is complete: 16/16
reports are valid, but every lower-tail score is 0.0. Fixed and all-static
old checkpoints therefore do not meet the native gate at any tested index.
Earlier cross-bucket v2 traces are diagnostic only; validator now rejects them,
noncanonical command amplitudes and seed hashes inconsistent with actual state.
Focused suite: 63 passed; ownership JUnit:
`/tmp/adaptive-native-isolated-replay/ownership.xml`.


Initial 300-step CUDA diagnostic endpoints at `/tmp/adaptive-native-v2-final/`
(fixed17/static17, evaluation seed 17) both finish all buckets without falling
but fail the canonical gate. These are diagnostic seeds, not held-out evidence.
Fixed17 forward error is 0.1187 m/s for a 0.12 m/s target; static17 lateral
error is 0.1473 m/s. Full disjoint-seed ladder is the next required evidence.

The old CPU/XML-position-actuator ladder has lower-tail 0 throughout and is
only a transfer rehearsal. CPU failure does not establish native failure, and
finite native traces do not establish a CPU actuator root cause.

## Next proof and continuation

1. Same checkpoint/seed replay: compare every array in all six traces; different
   seeds must change consumed reset/DR fields. Reports include actual seed and
   consumed-state hashes. Use disjoint gate/held-out seed ranges.
2. Evaluate fixed and all-static training seeds 17/23 at checkpoint indices
   500/1000/2000/3999, 300 steps per bucket. `model_3999.pt` is the zero-indexed
   endpoint of 4000 training iterations. Validate every report against raw traces.
3. Export via `scripts/export.py`; compare ONNX/native actions on the same six
   observation streams. Treat native learning, export parity and CPU actuator
   transfer as distinct questions.
4. Use validated curves and temporal gate replay to assess EMA/dwell/preservation
   and curriculum thresholds; preserve product thresholds unless a measured
   semantic/unit error is demonstrated. Do not invent calibration from quantiles.
5. If native fails, run bounded recipe/command/reward diagnostics before another
   campaign; if native passes but CPU fails, isolate the transfer discrepancy.

Verification inventory: focused adaptive config, checkpoint battery, production
runner, checkpoint resume, rollback, native battery and Phase 2A evidence tests;
Ruff on owned scripts; `git diff --check`; actual native runs and ONNX parity.

No new five-branch multi-seed campaign until the Phase 2A evidence gates pass.
That restriction is not a reason to stop existing-checkpoint diagnostics.
A bounded corrected all-static pilot completed: seed 17, 4096 environments,
500 iterations, `/tmp/adaptive-repaired-pilot/pilot.log`. Its endpoint
`model_499.pt` is evaluated at `/tmp/adaptive-repaired-pilot/native-final/` and
still fails the native gate: zero drift 0.082 m, lateral error 0.141 m/s,
yaw error 0.637 rad/s; forward improved to 0.054 m/s. This is evidence of
partial learning, not a usable policy or a matched-budget result. The pilot's
64-env/5-iteration smoke and standard normalizer-baked export passed.

A bounded 2000-iteration all-static diagnostic is still running: seed 17, 4096
environments, `/tmp/adaptive-repaired-2000/pilot.log`, with checkpoints under
`/tmp/logs/rsl_rl/adaptive_repaired_static_2000/2026-09-20_17-59-03_ownership-repair-s17/`.
The held-out native final-distribution reports remain below the product gate:
model 500 has zero drift 0.087 m, lateral error 0.136 m/s, and turn-right
error 0.767 rad/s; model 1000 improves zero drift to 0.033 m, forward to
0.053 m/s, yaw to 0.255 rad/s, and turn-right to 0.252 rad/s, but lateral
error is 0.148 m/s; model 1500 regresses to zero drift 0.079 m, forward
0.099 m/s, and lateral 0.119 m/s. All three aggregate scores are 0.0.
The same model 1000 at the training initial CoM distribution (±3 mm) still
fails (zero drift 0.089 m, lateral 0.136 m/s), so the final evaluator's
larger ±15/±10 mm distribution is not the sole cause. Training is live around
iteration 1600; the run completed at model 1999. Its native final-distribution
report is `/tmp/adaptive-repaired-2000/native-1999/native_capability.json`:
zero drift 0.072 m, forward error 0.103 m/s, lateral error 0.119 m/s, yaw
error 0.301 rad/s, turn-left 0.257 rad/s, and turn-right 0.257 rad/s. All
episodes survived and tilt stayed below 0.132 rad, but aggregate remains 0.0
because zero drift and lateral tracking miss the continuous six-bucket product
gate. The bounded recipe test therefore confirms partial native learning but
does not produce a usable policy; next work should be a single-axis diagnostic
(push robustness or explicit lateral command sampling), still before any
multi-seed campaign.

The explicit-lateral single-seed pilot is implemented as
`Mjlab-Velocity-Flat-Adaptive-Lateral-MicroDuck` (commit `edf6a3f`). It keeps
the canonical standing/action-rate/pose curricula and adds a 20% pure-lateral
command bucket; a 1024-env reset probe measured 18.75% pure-lateral samples.
The required 64-env/5-iteration smoke and focused config tests passed. Its
500-iteration seed-17 checkpoint is under
`/tmp/logs/rsl_rl/adaptive_lateral_pilot/2026-09-20_19-16-59_lateral-bucket-s17/`.
Native held-out evaluation is valid but still fails: zero drift 0.103 m,
forward error 0.055 m/s, lateral error 0.138 m/s, yaw error 0.629 rad/s,
turn-left 0.731 rad/s, and turn-right 0.673 rad/s. The lateral bucket did
increase mean lateral velocity from 0.001 m/s in the static model-1999 trace
to 0.040 m/s, but with large oscillation and no gate improvement. Resume this
same seed to 2000 iterations before rejecting the recipe; do not start a
multi-seed campaign from the 500-iteration result.

The lateral pilot has a sampling confound: its 20% lateral override replaces
some turn and standing samples. Read-only replay measured effective turn at
about 12% instead of 15% and standing at about 68% of the configured value;
with `rel_lateral_envs=0`, the old command tensors and Torch RNG state remain
bit-for-bit unchanged. The pilot is therefore diagnostic evidence only. Its
resume was verified from checkpoint state: model 499 had env step 12000 and
model 1000 had env step 24048; the current resume command is continuing toward
2499 total update labels, not a fresh matched 2000-iteration run. A corrected
mutually-exclusive sampler or a fresh from-scratch pilot is required before
attributing any improvement to lateral exposure.

The runner end-to-end smoke also passed: native held-out and separately named
CPU transfer reports were produced under `/tmp/adaptive-native-campaign-smoke-final/`;
both are negative valid reports. The first smoke exposed and fixed the live
EventManager reference-range overwrite, then the rerun completed successfully.
Stage only owned changes; this shared worktree contains unrelated external edits.
