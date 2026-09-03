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

Last proven evidence:

- Focused G0 suite: `49 passed`.
- Reset/anchor suite: `18 passed`.
- Required 64-env, 5-iteration hybrid smoke completed at
  `logs/rsl_rl/generalist_g0_hybrid_ppo/2026-09-03_11-48-19_g0_reset_anchor_smoke`.
- Smoke emitted 71D actor observations, no NaN termination, 384 held teacher
  samples/update, per-behavior anchor weights `0.1`, and nonzero anchor loss.
- Reset bucket labels (`upright`, `seated`, `recovery`) and transition phase
  progress are exposed on the environment; focused config/graph tests: `18 passed`.

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

Next hypothesis: the remaining failure is policy/task learning rather than
reset or anchor plumbing. Do not claim P4 acceptance; any next experiment must
change reward/curriculum or transition-phase coverage and be compared with this
baseline.

Next proof: run the fail-closed gate on `/tmp/g0-reset-anchor-300-eval.json`
with the existing parity/latency/fallback artifacts, then choose the next
bounded learning intervention.

Stop condition: P4 passes behavior, all legal transition, parity, latency, and
fallback gates, or a repeated experiment falsifies the current hypothesis and
identifies the next bounded root-cause class.

No-touch scope: specialist 61D ABI/runtime/roller/hardware behavior; `uv.lock`;
generated checkpoints, datasets, reports, and videos remain outside Git.

Parked work: transition-phase reset snapshots and trainer-level bucket metrics
remain a plan P2 coverage gap; labels now exist for the next curriculum slice.
