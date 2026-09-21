# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — acquisition and acceptance calibration remain open;
no usable policy established. Updated: 2026-09-22.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: continue the authorized sustained intuitive-flow objective
(necessary changes, training and behavioral checks). The 1500-update feedback
continuation, 500-update strictification probe, and 500-update
acquisition-feedback slice are finished; none establishes a usable policy.

## Current slice

Runner/exposure consolidation is committed as `8b2f28a`. Command-aligned
shaping `86ee5be` is included in `bb37a29`, which also protects the 20% zero
anchor and uses a 0.80 focus mastery threshold. The zero20 seed17 campaign
completed 1500 cumulative updates. Smoke, resume, native gates, normalized
ONNX export, and CPU transfer completed; product gates still fail. The
controller retains `zero` and `forward` on its gate seed set and focuses on
`lateral`. This partial retention does not generalize to the held-out product
gate. At env step 33000, forward regression caused a real
`preservation_failure -> rollback` to `model_1249.eval.adaptive.pt`; training
continued and recovered the two retained gate buckets. DR remains at stage 0.

Lateral plateaued: held-out MAE is 0.1252 m/s at 1000 updates and 0.1244 m/s
at 1500. A separate strictification bootstrap (lighter pose/action costs,
stronger tracking and 25% pure lateral sampling) failed at 500 updates with
0.0028 m/s mean lateral velocity against a 0.12 m/s command. The combined
acquisition-feedback slice improved forward MAE to 0.0287 m/s at 500 updates,
but held-out lateral MAE remained 0.1188 m/s and yaw MAE 0.7655 rad/s; zero
drift was 0.0767 m. Its native six-bucket gate still failed, and the CPU
transfer report also failed all buckets. Do not promote either checkpoint as
a usable policy.

The acquisition-feedback slice exposed a controller-specific confounder:
initial lateral focus was overwritten by the default frontier order because
forward was also below mastery. The final exposure was forward 21.7% and
lateral 14.3%, so the experiment did not maintain lateral priority. Next
hypothesis: a checkpointed lateral-first frontier must hold the acquisition
focus until lateral mastery, while retaining the zero anchor and adaptive
rollback. Keep product thresholds and samplewise MAE fixed; evaluation
semantic calibration remains separate.
Blocker fingerprint: `native_lateral_acquisition`; classification: stationary
or low-progress solution under tested recipes, with the previous slice also
confounded by frontier order; decision delta: reject direct adoption and test
a lateral-priority frontier before changing physics or thresholds.

The Feedback-only L1 terms now average signed velocity error before magnitude
with tau 0.5 s. Shared state updates once per step and resets on episode or
command changes using current error (no free startup grace period). Exact-zero
linear commands retain instantaneous idle-speed cost. Weights remain 0.5/0.35;
normalization floors remain 0.12 m/s and 0.8 rad/s. No action/observation
filtering, canonical recipe, PPO, or product-threshold changes.

Proof: focused tests and smoke64/5 passed, with nonpositive penalties, zero
NaN term, rollback-state coverage, and ONNX export. Latest feedback artifact:
`/tmp/microduck-adaptive-zero20-s17-1500/campaign-result.json`, checkpoint
`model_1499.pt`, SHA256
`4a93635a1f86f74ec859acfb58335de891ee3e162fbb449280af33b6ee665119`.
Judge mean progress, MAE, air-time and preservation together.

Latest acquisition-feedback artifact:
`/tmp/microduck-acquisition-feedback-s17-500/campaign-result.json`, checkpoint
`model_499.pt`. Native held-out and CPU reports are bound to the exact
checkpoint and source SHA `b99c6e0`; all six product buckets fail. The four
native gate windows were valid `hold` decisions with no rollback or evaluator
error. The 64-env/5-iteration smoke and 51 focused tests passed before this
run.

## Latest behavioral evidence

| Held-out metric | Zero20 1000 | Zero20 1500 | Product requirement |
| --- | ---: | ---: | ---: |
| Zero endpoint drift (m) | 0.0484 | 0.0624 | <=0.012 |
| Forward MAE (m/s) | 0.0319 | 0.0368 | <=0.024 |
| Lateral MAE (m/s) | 0.1252 | 0.1244 | <=0.024 |
| Yaw MAE (rad/s) | 0.2741 | 0.3541 | <=0.12 |
| Left/right turn yaw MAE (rad/s) | 0.2601 / 0.2659 | 0.1891 / 0.2646 | <=0.12 |

Final gate scores at model_1499: zero 0.870, forward 0.857, lateral 0.000,
yaw 0.421, turn-left 0.548, turn-right 0.371. Held-out final-DR still fails
all six buckets. CoM/head-CoM remain +/-3 mm; no successful difficulty
advance or autonomous acquisition procedure is established. Source: `bb37a29`.

Strictification diagnostic artifact:
`/tmp/microduck-strictification-s17/native-499/capability.json`. Seed 17,
4096 envs, 500 updates, gate evaluation seed 20260815, final DR. It survives
all six 6 s cases but passes none: zero drift 0.0143 m, forward/lateral MAE
0.1203/0.1178 m/s, yaw/left/right MAE 0.2734/0.2177/0.2560 rad/s. Mean
forward/lateral velocity is -0.00035/0.00278 m/s; lateral late velocity is
0.0000945 m/s. Its 35 focused tests and smoke64/5 passed before training;
those prove execution, not capability. This is a negative diagnostic, not a
held-out or transfer acceptance result. Report `source_sha` is `unknown`;
the two runtime patches match the saved training `git/holy-ape.diff` exactly.
Checkpoint SHA256:
`38a8f15fbaac63a6a26f21fd5fb1b512226bb15ae5115014120a5b947f5f959c`.
All native/CPU product verdicts remain negative; no video/hardware acceptance.

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
- Partial preservation and real rollback are established on the gate set;
  held-out preservation and all-six capability acquisition remain required.
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
