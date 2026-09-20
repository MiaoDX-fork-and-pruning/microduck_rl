# MJLab Adaptive Curriculum v2

- Status: **ACTIVE**, Phase 2A. No usable adaptive policy is established.
- Owner: `/root`; latest user intent is to continue executable experiments.
- Contract: [Phase 2A](../../plans/mjlab-adaptive-curriculum-v2/02a-diagnostic-repair-PLAN.md).
- No-touch: canonical velocity factory, product thresholds, unrelated IsaacLab edits.
- No external blocker. Local RTX 3090 is available.
- No five-branch/multi-seed campaign before Phase 2A evidence is complete.

## Current experiment and next decision

Lateral sampling did not solve native tracking. `edf6a3f` adds a diagnostic
20% lateral override. Its 500-update pilot completed; the same seed 17 resumed
from model 499 and is currently running as PID 3106039, with log
`/tmp/adaptive-lateral-pilot/resume3.log` and run directory
`/tmp/logs/rsl_rl/adaptive_lateral_pilot/2026-09-20_19-41-08_lateral-bucket-s17-resume/`.
Authoritative progress is the process/log, not this capsule. Resume requested
2000 ADDITIONAL updates: final label will be 2498, actual total 2500 updates.
Model 2000 has env step 48048 (2002 updates); use env step / 24 for budgets.
Environment counters/curricula restore, but episodes/startup DR restart.
This is diagnostic evidence, not a matched-budget campaign.

Next: finish this run and evaluate its endpoint. Then run one fresh seed-17,
4096-env, 500-update `Mjlab-Velocity-Flat-Adaptive-Tracking-MicroDuck` pilot.
It changes only linear tracking std from sqrt(0.1) to 0.12 m/s compared with
Adaptive-Static; command sampling and all other curricula/rewards remain fixed.
Hypothesis: broad velocity reward makes standing at a nonzero lateral command
cheaper than developing a stable gait. Do not infer success from total reward.
Require raw native forward/lateral velocity and error, falls, and all six scores.
Keep goal active until native usability, calibration, and required export/transfer
proofs are complete; do not stop because a diagnostic report was produced.

## Latest diagnostic evidence

- Repaired static seed 17: 2000 updates completed under
  `/tmp/logs/rsl_rl/adaptive_repaired_static_2000/2026-09-20_17-59-03_ownership-repair-s17/`.
  Final native report `/tmp/adaptive-repaired-2000/native-1999/` has drift
  0.0722 m, forward 0.1033 m/s, lateral 0.1193 m/s, yaw 0.3008 rad/s,
  left/right 0.2566/0.2571 rad/s. All buckets survive; aggregate is zero.
- Lateral diagnostic reports `/tmp/adaptive-lateral-pilot/native-{499,1000,1500,2000}/`
  also have aggregate zero. Model 2000: drift 0.0711 m, forward 0.0946 m/s,
  lateral 0.1192 m/s, yaw 0.2141 rad/s, left/right 0.2235/0.1546 rad/s.
  Lateral mean velocity falls from 0.0399 m/s at model 499 to 0.0008 at 2000.
  The improved error is collapse to standing, NOT learned lateral motion.
- Reward diagnosis `/tmp/adaptive-lateral-pilot/reward-replay.json`: actual
  weighted total rewards favor the stationary model 2000 (8.480/step) over
  model 1000 (8.364/step). Re-scoring just linear std at 0.12 reverses this
  ranking slightly (7.494 vs 7.519). This is a fixed-trajectory hypothesis
  test, not evidence PPO will learn. Transform the recorded exponential term;
  post-step velocities differ from the reward's pre-forward physics state.
- Sampling confound: lateral overrides turn/standing. Effective turn is 12%
  instead of 15%; standing is 68% of its configured fraction. Do not claim an
  isolated lateral exposure effect. Disabling lateral preserves old command
  tensors and Torch RNG exactly in the 200k-draw CPU audit.
- All-static trains CoM at +/-3 mm; final evaluator uses +/-15/10 mm.
  Initial-distribution model-1000 diagnostic also failed, but used a different
  evaluation seed, so it is not a paired CoM intervention.
- Zero endpoint drift includes push displacement: fixed23 recovers to final
  1-second mean speed 0.0010 m/s, local displacement 0.745 mm, but retains
  67 mm spawn-relative offset. Actor has no absolute position input. The old
  no-push test changed RNG consumption; it is not a clean causal comparison.
  Preserve thresholds; future push diagnostics must retain RNG draws/timers.

## Proven foundations and remaining gates

- `e22532e` repairs curriculum ownership: preserve every non-owned canonical
  standing/action-rate/pose schedule. Canonical task factory is unchanged.
- Native schema-v2 evaluator uses training MJLab/BAM M6, normalizer, 61D actor,
  14D action, terminal traces, live EventManager reference ranges, and fresh
  environment per bucket. `/tmp/adaptive-bam-reset-probe/compute.json` confirms
  stale qfrc and reward peak-height carryover in the old reused-env evaluator.
- `/tmp/adaptive-native-isolated-replay/{a,b,c,parity}`: same-seed CUDA traces
  exact across 19 fields/six buckets; different seeds change consumed state;
  ONNX action parity max error about 5.37e-7. Gate/held-out seeds disjoint.
- `/tmp/adaptive-native-isolated-ladder/`: 16 fixed/static seed17/23 checkpoint
  reports at 500/1000/2000/3999 all valid, every lower-tail score zero.
  Old native-v1 pass claims/calibration/manifest are superseded.
- Calibration and bound manifest remain incomplete. Never fabricate thresholds
  from quantiles of failing policies. Product pass requires score >=0.8, i.e.
  linear MAE <=0.024 m/s, angular <=0.12 rad/s, drift <=0.012 m, tilt <=7deg.
- CPU/XML position-actuator battery is a separate transfer rehearsal; its
  failure does not establish native learning failure. Export via scripts/export.py.

## Verification

Prior focused adaptive suite: 74 passed. Latest tracking/config/resume/native
checks: 38 passed; lateral sampler behavioral checks: 2 passed. Both diagnostic
64-env/5-update smokes passed (61D/14D, no NaN). Tracking smoke additionally
exported through scripts/export.py to `/tmp/adaptive-tracking-smoke.onnx` with
`/tmp/adaptive-tracking-smoke-golden.npz`. Focused config/tests Ruff and diff
checks passed; full mdp.py/__init__.py Ruff has existing baseline violations.
Stage only owned changes; shared worktree contains external edits.
