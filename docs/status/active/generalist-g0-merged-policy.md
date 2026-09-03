# Generalist G0 Merged Policy

Status: ACTIVE

Source plan: `docs/plans/generalist-g0-merged-policy-plan.md`

Control plane: root Codex session

Latest user intent: implement the full plan through `intuitive-flow`.

Current slice: behavior-conditioned reset coverage, durable bucket labels, and
anchored hybrid PPO configuration are repaired. SITSTAND resets now independently cross the
validated seated/standing physical buckets with stand/sit posture goals;
conditioning and command buffers agree. The hybrid anchor weight is a declared
dataclass field and therefore survives `dataclasses.asdict()` runner setup.

The G0 termination contract now allows arbitrary tilt from step zero; inherited
VelStand's 70-degree bootstrap termination is removed so sit/rise and recovery
traces are not truncated. The fallen-time backstop remains enabled.

Discovery economics now delay action-rate, body-angular-velocity, and
torque-rate penalties until `1200 * 24` environment steps, then restore their
validated baseline weights.

Reset curriculum now seeds 20% of fresh episodes into uniformly sampled legal
transition edges at a phase in the first 80% of dwell, with source/destination
and phase labels preserved. Hold-state buckets remain behavior-conditioned.

Measured stage promotion is now online: completed episodes accumulate finite
per-behavior success rates, require 64 samples and 90% success, and unlock
behaviors/edges monotonically. Stateful reward terms are wrapped with delegated
reset semantics so behavior masks apply to class-based terms as well as plain
functions.

Last proven evidence:

- Focused G0 suite: `49 passed`.
- Reset/anchor suite: `18 passed`.
- Required 64-env, 5-iteration hybrid smoke completed at
  `logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_11-48-19_g0_reset_anchor_smoke`.
- Smoke emitted 71D actor observations, no NaN termination, 384 held teacher
  samples/update, per-behavior anchor weights `0.1`, and nonzero anchor loss.
- Reset bucket labels (`upright`, `seated`, `recovery`) and transition phase
  progress are exposed on the environment; focused config/graph tests: `18 passed`.
- Long-horizon termination smoke completed at
  `logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_12-02-46_g0_long_horizon_smoke`:
  `fell_over=0`, finite rewards, and no NaN terminations.
- Discovery-tax smoke completed at
  `logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_12-12-15_g0_discovery_tax_smoke`;
  all three staged tax terms reported zero at step zero and training remained
  finite.
- Transition-reset smoke completed with 71D observations, finite rewards, and
  no NaN or reset-router failures.
- Measured-stage and stateful-mask smoke completed with finite rewards; stage 0
  reports zero SITSTAND reward mass and no NaN/reset failures.
- Full focused G0 regression suite: `54 passed`.

Transition-reset diagnostic: the fresh 300-iteration anchored run completed
finitely at `logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_12-28-27_g0_transition_reset_300`.
Corrected Track A still failed all behaviors and legal edges (stand tilt
`1.823 rad`, locomotion `1.718 rad`, sit/stand `1.843 rad`; several edge
tilts reached `pi`). Reverse-curriculum transition exposure alone did not
improve the merged policy.

Completed batch: P0-P4 tooling, frozen manifests/graph, 71D ABI correction,
hybrid teacher-action storage/loss, corrected 50 Hz Track A evaluator, and
fail-closed acceptance gate exist. Prior 300/600-iteration candidates remain
diagnostic failures; no G0 candidate is accepted.

Latest diagnostic: the fresh 300-iteration anchored hybrid run completed
finite at `logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_11-52-52_g0_reset_anchor_300`.
The corrected Track A battery is still a diagnostic failure: stand final height
`0.043 m`/tilt `1.268 rad`, locomotion displacement `0.123 m`, and all four
legal edges fail destination-specific gates. Reset coverage and anchor wiring
are therefore proven, but behavior learning remains unresolved.

Staged-tax diagnostic: the fresh 600-iteration run under delayed motion taxes
completed finitely at `logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_12-14-00_g0_discovery_tax_600`.
Corrected Track A still failed all behavior and legal-edge gates (stand tilt
`1.450 rad`, locomotion `1.749 rad`, sit/stand `1.501 rad`). Reset routing,
termination horizon, and tax timing interventions have now each been tested;
none restores merged-policy behavior.

Schema audit: evaluator conditioning, PPO actor ordering, and frozen-teacher
reconstruction agree on behavior offset 48, command offset 54, and the frozen
71D layout. The remaining failure is not an ABI or normalization-slot mismatch.

Termination experiment: removing the 70-degree bootstrap termination produced
longer finite episodes, but the 300-iteration checkpoint still failed all
behavior and edge gates. The hypothesis that truncation alone caused failure is
falsified. The evaluator's prior 0.18 m stand threshold was also invalid for
this model and is now derived from `STAND_Z=0.115` and `SIT_Z=0.060` with 10%
tolerance; the rerun remains a diagnostic failure.

Implementation contracts are now complete: schema/ABI, frozen teachers,
balanced collection, direct/hybrid PPO, teacher anchor, behavior masks,
behavior/state/transition reset buckets, measured stage unlocks, delayed
motion taxes, evaluator gates, parity, latency, fallback, and focused tests.
The remaining stop gate is empirical P4 behavior acceptance; current candidates
still fail the corrected Track A battery.

Latest gate: `/tmp/g0-final-gate.json` is `DIAGNOSTIC_FAIL` for all three
behaviors and four legal edges; parity/latency/fallback infrastructure passes.

Anchor-strength experiment: the hybrid bootstrap coefficient is now `1.0`
(previously `0.1`) because the PPO task stack overwhelmed the weak imitation
signal before VELSTAND could pass its measured unlock. Required smoke reports
nonzero anchor loss, weights `1.0`, episode length rising `64 -> 82`, and zero
bootstrap fall terminations. This is a training candidate, not P4 evidence.

Stage scalar audit: the stage-aware run recorded `success_0=0.0` through all
299 updates, so it correctly remained at stage 0. The upright classifier uses
the repository convention (`projected_gravity_b.z < -cos(65°)`), and the zero
rate reflects genuine failed stand holds rather than an orientation-sign bug.
Locked-behavior exposure was therefore intentional, not a silent curriculum
deadlock.

Stage-aware diagnostic: the 300-iteration run at
`logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_12-45-47_g0_stage_aware_300`
completed finitely, but measured stage progression did not improve behavior.
Corrected Track A still failed all behavior and edge gates (stand tilt
`2.144 rad`, locomotion `1.768 rad`, sit/stand `2.134 rad`). The fixed-router,
transition-reset, and measured-stage hypotheses have now each been tested.

Next proof: a future experiment must change the learned-policy intervention
itself (reward/initialization architecture or actor objective), then rerun
corrected Track A and the fail-closed gate.

Stop condition: P4 passes behavior, all legal transition, parity, latency, and
fallback gates, or a repeated experiment falsifies the current hypothesis and
identifies the next bounded root-cause class.

No-touch scope: specialist 61D ABI/runtime/roller/hardware behavior; `uv.lock`;
generated checkpoints, datasets, reports, and videos remain outside Git.

Parked work: generated long-run checkpoints/reports remain outside Git; no
additional skill is in scope until a G0 candidate passes P4.
