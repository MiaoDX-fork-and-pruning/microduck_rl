# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — autonomous acquisition repair; no usable policy established.
Updated: 2026-09-21. Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b`
(`/root`, worktree `holy-ape`).
Canonical scope: [plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest user intent: check status while the authorized sustained intuitive-flow
objective continues (necessary changes, training and behavioral checks).

## Current slice

The runner/exposure refactor is committed as `8b2f28a`; 126 focused tests passed
in this turn. A bounded worker owns adaptive-only normalized L1 velocity shaping
in `mdp.py`, `microduck_adaptive_velocity_env_cfg.py` and its focused tests. An
independent read-only worker audits evaluator semantics from existing traces.
Main owns integration, training, plan and this capsule. No new adaptive training
has launched yet. Another user's `gr00t_manipulation.visual_harness.g1_runner`
(PID 730702 at last check) occupies ~3.5 GiB of the 24 GiB RTX 3090; do not stop it.

Experiment hypothesis: the broad Gaussian tracking reward gives too little
benefit for acquiring lateral motion; adding a conservative, nonpositive L1
tracking penalty can improve the local objective without tightening the Gaussian
into a sparse exploration target. Only Feedback receives it; linear weight 1.0,
yaw 0.75, deadbands 0.01 m/s / 0.05 rad/s, normalization floors 0.12 m/s / 0.8
rad/s. Canonical rewards, PPO and product thresholds remain unchanged.

Proof route: focused tests -> smoke64/5 -> fresh seed17, 4096 envs, 250 updates,
gate every125 -> native held-out plus separately labelled ONNX/CPU rehearsal.
Compare equal-update gate traces with the frozen first pilot, and held-out
errors with its 500-update result. Continue only on measured capability progress;
otherwise diagnose before extending training. A new run uses an immutable source
snapshot and completion manifest. Next artifact root:
`/tmp/microduck-adaptive-objective-20260921-s17/` (250-update pilot).

## Latest behavioral evidence

The first Feedback pilot finished 16:24 CST, seed17, 4096 envs, 500 updates
(49,152,000 transitions); all four gate windows were valid. No continuation is
running. Reports are valid **negative** results, not an infrastructure failure.
All six native 6-second rollouts survived; none passed the whole product gate.

| Held-out metric | Actual | Required |
| --- | ---: | ---: |
| Zero endpoint drift | 0.1057 m | <=0.012 m |
| Forward MAE | 0.0472 m/s | <=0.024 m/s |
| Lateral MAE | 0.1126 m/s | <=0.024 m/s |
| Yaw MAE | 0.6560 rad/s | <=0.12 rad/s |
| Moving left/right yaw MAE | 0.5216 / 0.3984 rad/s | <=0.12 rad/s |

Artifact root: `/tmp/microduck-adaptive-formal-20260921-s17/`.
`review-500.json` summarizes the run; `run/training-result.json` identifies and
hashes the exact checkpoint; `run/campaign-result.json` has separate native and
CPU verdicts. Source hash:
`snapshot-sha256:9645b5234f14ac68c2597307e39cbd9a80a728ae2ed97d23efdfe6fc5df8e6dc`.
No visual motion-quality or hardware acceptance is established.

At update125 zero drift was 0.0089 m (score0.8515). Later drift increased to
0.0609/0.1511/0.1070 m, causing three preservation failures. No rollback occurred:
`last_known_good_checkpoint` requires all buckets passing and remained null.
CoM/head-CoM stayed +/-3 mm. Sampling changed modestly (forward13.33->11.87%,
lateral13.33->13.69%); preserving partial skills remains unresolved.

Latest reward audit: measured lateral mean velocity was0.033 m/s for0.12 command,
with0.112 m/s velocity standard deviation. Weighted linear reward still averaged
1.599/2, while angular reward averaged0.811/2, pose0.792/1 and air_time1.19.
Gaussian at a stationary0.12 command awards86.6% of maximum. This supports testing
objective shaping; it does not prove that shaping will solve acquisition.

## Verification and command surface

`run_adaptive_campaign_job.py --branch feedback --seed 17 --output <new-dir>
--iterations <cumulative-budget> --num-envs 4096 --gate-interval 125` runs smoke,
training, native held-out, normalized ONNX export and CPU rehearsal. Set
`MICRODUCK_SOURCE_SHA` to the immutable snapshot identity. Exact resume uses
`--resume <checkpoint>` and retains environment count/gate seed. Never glob for
trainer state. `status: evaluated` does not imply usable.

Focused tests:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest pytest -q
 tests/test_adaptive*.py tests/test_capability_metrics.py
 tests/test_mjlab_adaptive_velocity_config.py tests/test_mjlab_velocity_flat_config.py`.
Previous real smoke and5->10 resume are under `/tmp/adaptive-feedback-refactor-20260921-r2/`
and `/tmp/adaptive-feedback-refactor-20260921-resume/`. Native/ONNX action parity
passed all six30-step traces (max error1.05e-7). This is export proof, not policy
quality. Validated local dependencies are reused; fresh-sync portability remains
separate.

## Remaining gates and boundaries

- Acquisition-stage correction must preserve learned skills without repeatedly
  rewinding to a standing-only checkpoint.
- Audit instantaneous tracking errors versus gait oscillation, and zero-command
  displacement versus kick recovery. Paired native replay under
  `/tmp/adaptive-push-paired/v3/` gave0.0671 m kick offset versus0.00258 m without
  kick although both settled near0.001 m/s; actor observes no global position.
  Metric changes require written semantic evidence, not relaxed pass thresholds.
- Reproduce a usable procedure over seeds17/23/47 with held-out native traces,
  video inspection and deployment rehearsal; superiority to fixed additionally
  requires a matched-budget comparison.
- Stop a run on nonfinite training, invalid provenance, resource exhaustion or
  demonstrated regression; do not mark the root goal complete at a tests-only
  boundary. Do not launch the five-branch matrix until acquisition is informative.
- No-touch: canonical velocity recipe,61D/14D ABI, BAM M6, action filtering,
  unrelated IsaacLab/uv.lock changes and other users' processes.
- Parked: optional Codex advisor/stronger teachers, generalist, IsaacLab migration,
  hardware deployment. These do not substitute for native usable-policy evidence.
