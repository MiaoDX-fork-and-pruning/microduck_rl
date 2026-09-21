# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — acquisition and acceptance calibration remain open;
no usable policy established. Updated: 2026-09-21.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: status update during the authorized sustained intuitive-flow
objective (necessary changes, training and behavioral checks).

## Current slice

Runner/exposure consolidation is committed as `8b2f28a`. Command-aligned
shaping `86ee5be` completed 250 updates plus a checkpoint continuation to 500.
The framework runs smoke, training, native gates, normalized ONNX export and
CPU transfer automatically. All result manifests are valid; product gates fail.
No unchanged aligned-shaping continuation is planned.

Next hypothesis: the extra L1 penalty still taxes same-axis gait sway. A
measured lateral gait makes 0.0488 m/s progress toward 0.12 m/s but has
0.1413 m/s instantaneous MAE, worse than standing's 0.12. A 0.5 s average of
signed error gives 0.0721 m/s after settling, correctly ordering that partial
progress. This is a reward-hypothesis test, not a new acceptance metric.

The Feedback-only L1 terms now average signed velocity error before magnitude
with tau 0.5 s. Shared state updates once per step and resets on episode or
command changes using current error (no free startup grace period). Exact-zero
linear commands retain instantaneous idle-speed cost. Weights remain 0.5/0.35;
normalization floors remain 0.12 m/s and 0.8 rad/s. No action/observation
filtering, canonical recipe, PPO, or product-threshold changes.

Proof: 146 focused tests and smoke64/5 passed, with nonpositive penalties,
zero NaN term and an ONNX export (`/tmp/microduck-adaptive-mean-error-smoke/`).
Next: fresh seed17,4096 envs,500 updates,native gate every125 and unchanged
held-out/CPU batteries.
Artifact root: `/tmp/microduck-adaptive-mean-error-s17/`. Judge lateral mean
progress, MAE, air-time and preservation together; do not infer capability
from aggregate training reward. No automatic extension without a decision
supported by native traces.

## Latest behavioral evidence

| Held-out metric | Feedback 500 | Aligned L1 500 | Product requirement |
| --- | ---: | ---: | ---: |
| Zero endpoint drift (m) | 0.1057 | 0.1050 | <=0.012 |
| Forward MAE (m/s) | 0.0472 | 0.0436 | <=0.024 |
| Lateral MAE (m/s) | 0.1126 | 0.1184 | <=0.024 |
| Yaw MAE (rad/s) | 0.6560 | 0.5504 | <=0.12 |
| Left/right turn yaw MAE (rad/s) | 0.5216 / 0.3984 | 0.3529 / 0.3288 | <=0.12 |

Aligned reward air_time is 0.763 at250 and 0.999 at500 (mean air-time is a
separate statistic). The first componentwise-L1 experiment fell to 0.0039
reward at500; aligned shaping prevents that collapse, but has not acquired
lateral control. Four native gates are valid. At375 the zero bucket regresses
and reports preservation_failure; no rollback is possible because no complete
last-known-good checkpoint exists. CoM/head-CoM remain +/-3 mm.

Evidence: `/tmp/microduck-adaptive-aligned-s17/review-500.json`, with exact
checkpoint/report hashes and diagnostic labels. Source: `86ee5be`.
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
  exists. Current gate reports the loss but has no effective early rollback.
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
