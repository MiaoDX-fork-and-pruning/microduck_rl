# MJLab Adaptive Curriculum v2

- Status: **ACTIVE**, Phase 2A. No usable adaptive policy is established.
- Owner: `/root`; latest user intent is to continue executable experiments.
- Contract: [Phase 2A](../../plans/mjlab-adaptive-curriculum-v2/02a-diagnostic-repair-PLAN.md).
- No-touch: canonical velocity factory, product thresholds, unrelated IsaacLab edits.
- No external blocker. Local RTX 3090 is available.
- No five-branch/multi-seed campaign before Phase 2A evidence is complete.

## Current experiment and next decision

The push robustness diagnostic is the active bounded experiment. Its fresh
500-update seed-17 pilot completed at
`/tmp/logs/rsl_rl/adaptive_push_pilot/2026-09-20_21-37-28_push-curriculum-s17/`;
the held-out native report is `/tmp/adaptive-push-pilot/native-499/`. It is
valid and NaN-free but fails the usability gate: zero drift 0.0837 m, forward
error 0.0638 m/s, lateral error 0.1716 m/s, yaw error 0.2027 rad/s, and
turn-left/right errors 0.2351/0.1884 rad/s. The push curriculum had reached
only +/-0.15 m/s at that checkpoint, so a single 500-update failure does not
identify the endpoint behavior.

The same checkpoint is now resuming for 1500 additional updates (total
2000), with explicit run directory
`/tmp/logs/rsl_rl/adaptive_push_pilot/2026-09-20_22-07-55_push-curriculum-s17-resume1500/`.
The native endpoint battery is queued by `/tmp/adaptive-push-followup.log` and
will write `/tmp/adaptive-push-pilot/native-1999-resume/` after
`model_1999.pt` exists. This is diagnostic evidence, not a matched-budget
campaign. Require raw native forward/lateral velocity and error, falls, and all
six scores before deciding whether push robustness is useful.

The continuation completed at `model_1998.pt` (the resume starts from
iteration 499, so the final label is 1998). Its final native report is
`/tmp/adaptive-push-pilot/native-1998-resume/`. It is valid and stable in the
six-bucket replay, but still fails the current schema-v2 gate: zero drift
0.0729 m, forward error 0.0942 m/s, lateral error 0.1192 m/s, yaw error
0.2760 rad/s, and turn-left/right errors 0.2585/0.1921 rad/s. The aggregate is
0.0. The old schema-v1 reports that passed a looser historical gate are
superseded and do not establish a usable policy.

If the 2000-update push endpoint still fails, continue with one bounded recipe
diagnostic selected from the measured failure mode; do not start a five-branch
or multi-seed campaign. Keep the goal active until native usability,
calibration, and required export/transfer proofs are complete; do not stop
because a diagnostic report was produced.

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
- Independent paired push replay `/tmp/adaptive-push-paired/v3/` used two
  processes with identical reset and pre-push traces. Applying the sampled kick
  left 0.0671 m endpoint offset while applying zero kick left 0.00258 m; both
  final local speeds were about 0.001 m/s. Endpoint drift therefore measures a
  persistent post-kick displacement and does not imply continued idle motion.
- All-static trains CoM at +/-3 mm; final evaluator uses +/-15/10 mm.
  Initial-distribution model-1000 diagnostic also failed, but used a different
  evaluation seed, so it is not a paired CoM intervention.
- Zero endpoint drift includes push displacement: fixed23 recovers to final
  1-second mean speed 0.0010 m/s, local displacement 0.745 mm, but retains
  67 mm spawn-relative offset. Actor has no absolute position input. The old
  no-push test changed RNG consumption; it is not a clean causal comparison.
  Preserve thresholds; future push diagnostics must retain RNG draws/timers.
- Push pilot report `/tmp/adaptive-push-pilot/native-499/` is valid but still
  fails all-static usability; the total-2000 continuation is in progress.

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
