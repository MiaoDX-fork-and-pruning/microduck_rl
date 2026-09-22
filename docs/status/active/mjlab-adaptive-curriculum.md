# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy accepted. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
The user authorized sustained necessary changes, training and verification through
`intuitive-flow`; the latest status request does not cancel execution.
Project-status writer: not assigned; project-status delta: none.

## Current state

Phase 0/1 automation is implemented; Phase 2A usable-policy acquisition remains
open. Both CoM axes remain at stage 0 (±3 mm). No acceptance threshold has changed:
**every capability bucket must score ≥ .80**; scores are not success rates.

The latest pure-yaw-only smoothing-relief campaign completed 6250→6500 updates,
seed 17, 4096 envs. The candidate was retained without rollback (all 13 actor
state tensors equal the candidate/known-good snapshot), but fails usable-policy
acceptance. It is not promoted and must not be extended unchanged.

| Pure-yaw relief evaluation | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, worst of 20260815–17 | .975 | .718 | .802 | .000 | .749 | .719 |
| Final native diagnostic, 20260915 | .949 | .868 | .812 | .000 | .649 | .788 |
| CPU rehearsal, 20260915 | .973 | .859 | .789 | .000 | .838 | .841 |

Yaw fails in all three gate members. Native/CPU average yaw rates are .103/.046
rad/s for a .8 command: under-speed/idle. The same full-push stage case that
felled global relief now survives (lateral .796 vs .298; no-relief .790), but
still falls below .80. Six reset fields and the initial actor observation match
the no-relief control. All six ONNX action comparisons pass, max error 6.56e-7.

Authoritative latest evidence:
`/tmp/microduck-adaptive-pure-yaw-s17-250/experiment-summary.json`.
Checkpoint: its `training/.../2026-09-23_06-29-27_matched-lateral-drive-s17/model_6499.pt`.
SHA: `fc061a0587d25784bb118e9174aec0b6d71445f1d26839cc33132ffb6ed01cd7`.
All 680 immutable source files match the manifest. Session `29482` completed.

## Previous retained global-relief actor

The global-relief campaign consumed 6250→6750 updates; its 6750 candidate was
rejected and the final `model_6749.pt` retains the 6500 actor. All actor tensors
were checked. Evidence:
`/tmp/microduck-adaptive-action-rate-s17-500/experiment-summary.json`.

| Retained global 6500 evaluation | Zero | Forward | Lateral | Yaw | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage gate, worst of 20260815–17 | .938 | .735 | .796 | .669 | .830 | .812 |
| Final native diagnostic, 20260915 | .923 | .879 | .797 | .837 | .737 | .820 |
| Stage diagnostic, worst of 20260915–24 | .914 | .696 | .298 | .146 | .707 | .767 |
| CPU rehearsal, 20260915 | .940 | .903 | .727 | .000 | .696 | .849 |

No diagnostic seed passes all six. CPU yaw is overspeed (1.390 rad/s vs .8),
not the latest policy's idle failure. All 60 cohort native/ONNX parity cases
pass (max error 1.67e-6). Export parity does not establish physics transfer.
The matched no-relief 6250→6500 control scores yaw .353; global relief .669.
Do not describe the entire 500-update relief campaign as a 250-update comparison.

Sensor ablation: seed 20260919 yaw .146→.807 without encoder bias, mean rate
.360→.830; identity IMU alone leaves .115. Seed 20260915 improves with identity
IMU. All 72 traces verified. Lateral still falls under all four sensor variants.
`/tmp/microduck-relief-6500-sensor-bias/analysis-summary.json`.
Push ablation locates that lateral fall after a (+.284, −.295) m/s push at 3.9 s;
half/no push survives (.801/.807). Actual qvel writes supply valid push evidence;
the earlier cached root-link velocity delta was invalid.
`/tmp/microduck-relief-6500-push-sensitivity/analysis-summary.json`.
Neither diagnostic authorizes weaker product DR or pushes.

## Current decision and next experiment

Blocker: `reset_conditioned_acquisition_and_cpu_yaw_rate_transfer`.
New classification: stochastic training can mask deterministic yaw startup
failure. This changes the next experiment from reward-scope tuning to late
consolidation; no blind exposure-only or unchanged relief extension.

`/tmp/microduck-yaw-exploration-dependence-v2/analysis-summary.json` covers six
64-env fixed-actor rollouts. For pure-yaw relief, deterministic / first-second
noise / continuous noise produce mean final-two-second rates .202/.609/.805.
Idle counts are 45/17/1 out of 64; one pulse rollout falls. Of the 45 originally
idle cases, 30 move ≥ .4 rad/s after the pulse. Global relief's corresponding
means are .819/.845/.816. All eleven recorded reset/observation fields match and
six trace hashes pass. The noise samples match, but batched Warp contact paths
are not bitwise replays. Auto-reset can also change RNG after first failure;
only each initial episode's survival is counted. These are diagnostic tail
statistics, not capability passes. No deployment noise is proposed.

Next bounded hypothesis: lower entropy during consolidation reduces reliance
on stochastic rollouts while retaining the global actor's acquired yaw.
Resume the **exact global 6500 adaptive snapshot**, not the retained actor's
6750-budget filename; seed 17, 4096 envs, 250 updates, inherited global relief,
same gate seeds/distribution, standard PPO entropy coefficient .01→0.0. Use the
immutable `610fbc5` source and an explicit hashed launch wrapper; smoke64/5
before training. Record real argv and verify `params/agent.yaml`. Compare with
the existing .01 extension; source/version differences must remain labelled.
Inspect candidate versus retained actor, learned std, native rates, CPU rates
and recovery before any controller/default promotion. No new run is launched
at this checkpoint; exact output/session must be recorded when started.

## Proven implementation and verification boundaries

`610fbc5` adds checkpointed `pure_yaw` scope to bounded action-rate relief.
Only pure-yaw commands get effective weight −.2; other commands keep canonical
costs. Trigger/release use yaw only; legacy/global scope stays supported.
97 focused tests plus 13 campaign tests after one added regression (98 distinct),
focused Ruff and diff checks passed. Real 64-env/5-update smoke passed; separate
live weighted-cost/finite-ABI/unchanged-action proof:
`/tmp/microduck-pure-yaw-610fbc5-live-proof/runtime-proof.json`.
Source: `/tmp/microduck-adaptive-pure-yaw-source-610fbc5`; manifest file SHA
`5c2e75579942e6f46d1e8c94a6d6fb6f72de0c3eca11049553e54141948d5d58`.
Source label: `610fbc5-cpu-seed-overlay-5ae829076760`.
Earlier infrastructure: 269 relevant tests and fresh locked non-editable install/
config/14-actuator model compile pass. Fresh-env GPU and remote execution remain
unproven. Full `mdp.py` Ruff has 17 pre-existing findings. CPU link-frame fix is
`931e3c9`; old principal-inertia-frame scores are superseded.

## Remaining acceptance and boundaries

Required: retained six-bucket final-range native mastery, fresh held-out seeds,
rollout/video inspection, normalized ONNX and CPU rehearsal, autonomous training
seeds 17/23/47, canonical final-range fine-tuning and matched-budget fixed/axis
comparisons. No usable-policy, hardware or adaptive-superiority claim exists.
No external blocker prevents the next experiment; goal remains active.

Preserve canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward signs,
.80 gates, exact-zero/nominal anchors and the existing CPU seed patch in
`scripts/run_specialist_action_battery.py`. IsaacLab, `uv.lock`, generated files
and unrelated processes are outside scope. Advisor, stronger teachers,
generalist and hardware deployment remain parked. Use the validated local venv
and unique `/tmp` working directories; worktree `logs/rsl_rl` is unwritable.
