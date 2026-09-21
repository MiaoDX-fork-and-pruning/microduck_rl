# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — acquisition and acceptance calibration remain open;
no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: status update during the authorized sustained intuitive-flow
objective (necessary changes, training and behavioral checks).

## Current slice

Runner/exposure consolidation is committed as `8b2f28a`. Command-aligned
shaping `86ee5be` is included in `bb37a29`, which also protects the 20% zero
anchor and uses a 0.80 focus mastery threshold. The zero20 seed17 campaign
completed 1000 cumulative updates from the 500-update checkpoint. Smoke,
resume, native gates, normalized ONNX export, and CPU transfer all completed
with valid manifests; product gates still fail. The adaptive controller did
preserve `zero` and `forward`, then moved focus to `lateral` after forward
mastery.

Next hypothesis: lateral acquisition remains the limiting skill after the
adaptive focus switch. The native gate lateral error is still about 0.131 m/s
at model_999 while the other directional skills improve. Continue the exact
checkpoint with lateral focus to test whether exposure is sufficient; if the
lateral error plateaus, inspect its reward/command semantics before changing
the acceptance metric. The samplewise MAE and product thresholds remain fixed.

The Feedback-only L1 terms now average signed velocity error before magnitude
with tau 0.5 s. Shared state updates once per step and resets on episode or
command changes using current error (no free startup grace period). Exact-zero
linear commands retain instantaneous idle-speed cost. Weights remain 0.5/0.35;
normalization floors remain 0.12 m/s and 0.8 rad/s. No action/observation
filtering, canonical recipe, PPO, or product-threshold changes.

Proof: focused tests and smoke64/5 passed, with nonpositive penalties, zero
NaN term, rollback-state coverage, and ONNX export. The 1000-update artifact
is `/tmp/microduck-adaptive-zero20-s17-1000/`; its final native gate is still
not a usable-policy result. Continue acquisition only from its exact
`model_999.pt` checkpoint and judge lateral progress, MAE, air-time, and
preservation together; do not infer capability from aggregate reward.

## Latest behavioral evidence

| Held-out metric | Aligned L1 500 | Zero20 1000 | Product requirement |
| --- | ---: | ---: | ---: |
| Zero endpoint drift (m) | 0.1050 | 0.0484 | <=0.012 |
| Forward MAE (m/s) | 0.0436 | 0.0319 | <=0.024 |
| Lateral MAE (m/s) | 0.1184 | 0.1252 | <=0.024 |
| Yaw MAE (rad/s) | 0.5504 | 0.2741 | <=0.12 |
| Left/right turn yaw MAE (rad/s) | 0.3529 / 0.3288 | 0.2601 / 0.2659 | <=0.12 |

At native gate model_999, the scores are zero0.848, forward0.849,
lateral0.000, yaw0.369, turn-left0.465, and turn-right0.480. Zero and
forward are now stable enough to be retained, while lateral remains the
acquisition bottleneck. Held-out final-DR still fails every bucket despite
better yaw/turn tracking. The controller recorded eight hold windows, no
rollback, and focus `lateral`; CoM/head-CoM remain +/-3 mm.

Evidence: `/tmp/microduck-adaptive-zero20-s17-1000/campaign-result.json`, with
exact checkpoint/report hashes and adaptive event provenance. Source:
`bb37a29` (including `86ee5be`).
The250 source snapshot is `/tmp/microduck-adaptive-aligned-s17/source/`;
500 continuation is `/tmp/microduck-adaptive-aligned-s17-500/run/`.
First unshaped pilot: `/tmp/microduck-adaptive-formal-20260921-s17/`.
Componentwise L1: `/tmp/microduck-adaptive-objective-s17[-500]/`.
All native/CPU verdicts remain negative; no video/hardware acceptance.

Diagnostics completed:
- Initial DR at250 still gives forward/lateral MAE0.1190/0.1185, so final
  CoM width alone does not explain the failure.
- A small nonzero head/body command intervention does not rescue tracking.
  This reduces confidence in that specific hypothesis, without ruling out
  all command-distribution effects.
- At250, forward MAE varies from0.0342 (gate seed20260815) to0.1191
  (held-out seed20260915); extra seed20260916 gives0.0338. This extra set
  overlaps held-out bucket seeds and is diagnostic only. Lateral fails all.
- Aligned500 with diagnostic diagonal command(0.20,0.12,0) makes0.0552 m/s
  lateral progress vs0.0034 for pure lateral. A separate fresh250-update
  diagonal-bucket training experiment still gives pure-lateral MAE0.1195;
  no further diagonal-only continuation. Source patch and manifest reside
  under `/tmp/microduck-adaptive-diagonal-s17/` and were not adopted in repo.

## Remaining gates and boundaries

- Acquire all six capabilities, then reproduce the automated procedure across
  training seeds17/23/47 with disjoint held-out native evidence, video
  inspection and normalized ONNX/CPU deployment rehearsal.
- Calibrate gate semantics: samplewise velocity MAE includes gait ripple;
  zero endpoint drift includes push displacement despite no absolute-position
  observation. Paired kick evidence under `/tmp/adaptive-push-paired/v3/`
  shows0.0671 m offset vs0.00258 without kick, both settling near0.001 m/s.
  Any metric change needs written semantics and calibration; failed policies
  must remain failed. Do not lower thresholds to make reports pass.
- Preserve partially learned skills before a complete last-known-good policy
  exists. The zero20 continuation now records a valid partial known-good
  checkpoint containing `zero` and `forward`; it still does not constitute a
  usable policy.
- Native battery uses MJLab/BAM with DR/noise/delay. CPU/XML-position-actuator
  rehearsal is independent transfer evidence, not proof of native quality.
- Stop on nonfinite training, invalid provenance, resource exhaustion or
  demonstrated regression. No five-branch/multi-seed superiority campaign
  until acquisition is informative. Fresh-sync portability is unproven;
  local runs reuse the validated venv.
- No-touch: canonical velocity recipe,61D/14D ABI,BAM M6,action filtering,
  unrelated IsaacLab/uv.lock/generated-file changes, other users' processes.
- Parked: optional advisor/stronger teachers, generalist, IsaacLab migration,
  hardware deployment. They do not substitute for native usable-policy proof.

## Verification commands

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest pytest -q
 tests/test_adaptive*.py tests/test_capability_metrics.py
 tests/test_mjlab_adaptive_velocity_config.py tests/test_mjlab_velocity_flat_config.py`

`run_adaptive_campaign_job.py --branch feedback --seed 17 --output <new-dir>
 --iterations <cumulative-budget> --num-envs 4096 --gate-interval 125`
uses explicit result manifests and exact checkpoint paths. Set
`MICRODUCK_SOURCE_SHA` and import the matching immutable snapshot. All logs go
under `/tmp`; repo `logs/rsl_rl` is not writable. A `status:evaluated` manifest
is completion of the experiment, not a passing policy.
