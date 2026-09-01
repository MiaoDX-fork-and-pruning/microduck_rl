Status: COMPLETE
Source plan: docs/plans/generalist-v0-execution-plan.md
Completed slice: persistent MuJoCo-to-official-Rust closed loop

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
- Full evidence is stored in `artifacts/generalist-v0/p1-official-runtime-
  mujoco-full.json`; the smoke evidence is stored beside it with `smoke` in
  the filename. The full run uses 300 active controller ticks after frozen
  homing and records target discontinuity, contacts, tilt, displacement, labels,
  gains, fall/recovery flags, and unsupported cross-profile edges.
- The corrected 240-tick full report keeps the command active after homing and
  reaches `homing -> walk -> stand` on the walk profile and `homing -> walk` on
  rollers. Both exceed the 65-degree fall gate; only walk sees the official
  fall flag clear, and neither proves physical recovery. The report is
  intentionally failed rather than accepting transport activity as parity.
- The current 300-active-tick report passes both profiles at the accepted
  `0.20 m/s` walk command. Walk reaches `homing -> walk -> stand`, records an
  official fall/recovery indication, and ends with 96 mm forward displacement;
  rollers reaches `walk` and ends with 847 mm forward displacement. The lower
  `0.12 m/s` diagnostic run remains retained as a failed displacement case.

Next phase:
- P2 walk conditioned BC baseline, using this P1 package as the composition
  gate. Keep specialists immutable and available as fallback.

Stop gate:
- P1 complete: both profile reports prove persistent official-controller state,
  exact commit/model hashes, and explicit unsupported edges.

No-touch: production runtime defaults, specialist ABI, cross-profile fallback,
and any second policy scheduler.
