Status: ACTIVE
Phase: P2 walk conditioned BC baseline

Completed slice:
- Frozen `generalist-v0` schema v2 adapter with 71D input and 14D raw action.
- G0 teacher collection from immutable `velstand_flat` and `velocity_flat` P0 traces.
- Offline BC MLP smoke completed with 3,900 samples (3 epochs, seed 7).

Evidence:
- Contract tests: `tests/test_generalist_schema.py` (6 passed).
- BC smoke output was written to the untracked `/tmp/p2-walk-bc-smoke` directory.
- Smoke metrics: train MSE `0.0067272`, validation MSE `0.0043456`.
- 30-epoch deterministic reruns produced identical model hashes and metrics:
  train MSE `0.00125484`, validation MSE `0.00111708`.
  Per-behavior validation MSE: stand `0.00005858`, locomotion `0.00120211`.
- The dataset contains 3,900 finite samples and is serializable as NPZ with
  explicit 71D/14D shapes.
- A 100-epoch baseline reached validation MSE `0.00017076` (stand
  `0.00000327`, locomotion `0.00018422`). Outputs were finite with maximum
  absolute action `0.6105` and no values outside `[-1, 1]`.
- Independent evaluator `scripts/evaluate_generalist_bc.py` reproduced finite
  outputs and reported MSE `0.00021248` (stand `0.00005215`, locomotion
  `0.00022584`).
- Student MuJoCo rollout (`scripts/rollout_generalist_bc.py`, 120 ticks) is
  intentionally recorded as a failed behavior gate: stand max tilt `2.004 rad`
  and locomotion max tilt `1.306 rad`, both above the `65 deg` (`1.134 rad`)
  threshold. Outputs remained finite and within range, so this is a control
  quality failure rather than an ABI failure. Report:
  `artifacts/generalist-v0/p2-walk-bc-rollout.json`.
- A first DAgger smoke collected 240 student-state samples with `beta=0.5`,
  relabeled by the correct immutable teachers (`velstand_flat` for stand and
  `velocity_flat` for locomotion). The retrained candidate passes locomotion
  for 120 ticks (max tilt `0.384 rad`, displacement `40.4 mm`) but stand still
  fails (max tilt `1.635 rad`). This confirms a behavior-specific gap rather
  than a general ABI or rollout failure.
- The pooled data is imbalanced (420 stand vs 3,720 locomotion samples). The BC
  trainer now uses deterministic behavior-balanced sampling by default so the
  offline objective cannot hide the stand condition.

Remaining P2 work:
- Freeze a versioned teacher manifest outside Git with checkpoint/ONNX hashes.
- Add Rust-readable golden vectors and export metadata for the 71D student contract.
- Diagnose the rollout stability gap (teacher initialization, action scale, or
  covariate shift) and rerun the MuJoCo G0 battery before any new behavior is
  added.
- Continue DAgger/initialization work for the stand condition; do not advance
  to sit/kick/roulade until both G0 behavior gates pass.
- A VelStand-expanded actor initialization experiment and a teacher-normalized
  input experiment were both run. Neither passed the G0 rollout gate; the
  normalized candidate also regressed locomotion. These are retained as failed
  diagnostics, not as accepted baselines.
- A raw-input DAgger candidate with separate immutable teachers and a bounded
  output head passed stand (max tilt `0.090 rad`) but failed locomotion (max
  tilt `1.365 rad`). The prior unbounded small-model DAgger candidate showed the
  opposite tradeoff. This is evidence of shared-actor behavior interference;
  the next experiment should use explicit per-behavior heads or a continuity
  objective, not more untracked hyperparameter sweeps.
- Added an explicit `g0_multihead` actor with a shared 71D trunk and separate
  stand/locomotion heads. Its routing test passes; the bounded candidate passes
  stand (max tilt `0.037 rad`) but locomotion remains outside the gate
  (`1.308 rad`). The unbounded control also fails and exceeds action range.
  This validates the behavior-interference hypothesis but does not qualify the
  model for rollout.
- Do not add sit/kick/roulade until the G0 baseline and transition gates are reviewed.

No-touch: specialist 61D ABI, production runtime defaults, official scheduler, and roller track.
