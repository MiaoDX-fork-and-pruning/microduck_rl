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

Next hypothesis: the remaining failure is policy/task learning, specifically
the still-unimplemented transition-phase reset/state curriculum, rather than
schema,
reset, anchor, or termination plumbing. Any next experiment must change the
reward curriculum or transition-phase coverage and be compared with this
baseline.

Next proof: run a bounded fresh hybrid diagnostic under the long-horizon
termination contract, evaluate its final checkpoint with corrected Track A, and
run the fail-closed gate with existing parity/latency/fallback artifacts.

Stop condition: P4 passes behavior, all legal transition, parity, latency, and
fallback gates, or a repeated experiment falsifies the current hypothesis and
identifies the next bounded root-cause class.

No-touch scope: specialist 61D ABI/runtime/roller/hardware behavior; `uv.lock`;
generated checkpoints, datasets, reports, and videos remain outside Git.

Parked work: transition-phase reset snapshots and trainer-level bucket metrics
remain a plan P2 coverage gap; labels now exist for the next curriculum slice.
