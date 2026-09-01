Status: ACTIVE
Source plan: docs/plans/generalist-v0-execution-plan.md
Current slice: persistent MuJoCo-to-official-Rust closed loop

Proven:
- P0 specialist battery and pinned official fall contract pass.
- Fork branch `test/official-fall-replay` has test-only replay coverage for the
  fall chain and independent long walk/roller command sequences.
- Fork commit `66d4fa8facd4c564f8346dc54f361ceaa5e28d59` adds the explicit `robotd --replay-stdin` NDJSON
  transport. It uses the real `control_loop`, preserves its state across
  frames, and emits official targets, gains, torque writes, and state labels.
- Official controller owns policy selection, command smoothing, gain, fall
  recovery, and handback; no second scheduler was added.
- `scripts/run_official_runtime_mujoco.py` passes a 20-tick smoke for walk and
  rollers with finite 15D targets and zero hidden resets.
- The corrected 240-tick full report keeps the command active after homing and
  reaches `homing -> walk -> stand` on the walk profile and `homing -> walk` on
  rollers. Both exceed the 65-degree fall gate; only walk sees the official
  fall flag clear, and neither proves physical recovery. The report is
  intentionally failed rather than accepting transport activity as parity.
- The current 300-active-tick report is more precise: rollers passes; walk stays
  upright and switches `walk -> stand`, but fails the existing displacement gate
  at 3.4 mm. This is a behavior failure, not a bridge or ABI failure.

Next proof:
- Diagnose the walk command/handoff behavior demonstrated by the failed 300-
  active-tick run, then rerun the
  separate full walk and roller batteries. Do not weaken the fall gate.

Stop gate:
- P1 is not complete until both profile reports prove deterministic
  controller state across a MuJoCo transition and the exact official commit,
  bridge version, model hashes, and unsupported edges are recorded.

No-touch: production runtime defaults, specialist ABI, cross-profile fallback,
and any second policy scheduler.
