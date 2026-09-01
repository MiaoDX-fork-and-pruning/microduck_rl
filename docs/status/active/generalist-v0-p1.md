Status: ACTIVE
Source plan: docs/plans/generalist-v0-execution-plan.md
Current slice: persistent MuJoCo-to-official-Rust closed loop

Proven:
- P0 specialist battery and pinned official fall contract pass.
- Fork branch `test/official-fall-replay` has test-only replay coverage for the
  fall chain and independent long walk/roller command sequences.
- Fork commit `66d4fa8` adds the explicit `robotd --replay-stdin` NDJSON
  transport. It uses the real `control_loop`, preserves its state across
  frames, and emits official targets, gains, torque writes, and state labels.
- Official controller owns policy selection, command smoothing, gain, fall
  recovery, and handback; no second scheduler was added.

Next proof:
- Add the MuJoCo-side driver with an explicit 14-policy-joint to 15-runtime-
  joint mapping, then run separate walk and roller long-sequence batteries,
  recording labels, gains, targets, tilt, displacement, contacts,
  fall/recovery, and reset/latency metrics.

Stop gate:
- P1 is not complete until both profile reports prove deterministic
  controller state across a MuJoCo transition and the exact official commit,
  bridge version, model hashes, and unsupported edges are recorded.

No-touch: production runtime defaults, specialist ABI, cross-profile fallback,
and any second policy scheduler.
