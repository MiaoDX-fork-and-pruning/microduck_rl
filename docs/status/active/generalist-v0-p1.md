Status: ACTIVE
Source plan: docs/plans/generalist-v0-execution-plan.md
Current slice: official Rust runtime replay evidence

Proven:
- P0 specialist battery and pinned official fall contract pass.
- Fork branch `test/official-fall-replay` has test-only replay coverage for the
  fall chain and independent long walk/roller command sequences.
- Official controller owns policy selection, command smoothing, gain, fall
  recovery, and handback; no second scheduler was added.

Next proof:
- Add a persistent MuJoCo-to-Rust sensor bridge and run separate walk and
  roller long-sequence batteries, recording labels, gains, targets, tilt,
  displacement, contacts, fall/recovery, and reset/latency metrics.

Stop gate:
- P1 is not complete until both profile reports prove deterministic
  controller state across a MuJoCo transition and the exact official commit,
  bridge version, model hashes, and unsupported edges are recorded.

No-touch: production runtime defaults, specialist ABI, cross-profile fallback,
and any second policy scheduler.
