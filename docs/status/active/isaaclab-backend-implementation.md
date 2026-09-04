status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: reproduce mjlab Velocity-Flat semantics in IsaacLab via intuitive-flow
current_slice: forced self-contact runtime proof and seeded reset/DR/command contracts complete; remaining P0 dynamics/source rows open
blocker_kind: implementation_and_parity_evidence
blocker_fingerprint: P0 asset-friction dynamics, command resampling, full reward/termination/critic fixtures remain open
last_proven_evidence: >-
  `velocity_flat_home_smoke_1.json` runs the real IsaacLab 3.0.0 / Isaac Sim
  6.0.1 articulation: policy-ordered default HOME has max absolute error 0.0,
  right_hip_yaw is 0.0, actor/action shapes are 61D/14D, critic wiring is 76D,
  and all 15 contact-body tensors are finite. The one-environment contact report
  proves finite filtered self-contact matrices, nonzero foot-ground force, all
  16 finite reward terms, and finite termination terms. A direct mjlab-kernel
  regression proves equal self-contact point-count semantics. Concrete raw
  PhysX views now initialize for four environments with trunk/leg count buffers
  `(4,2)` and `(4,1)`; `contact_runtime_forced_direct_4.json` writes a known
  MuJoCo-valid collision pose with zero write error and observes baseline 0 ->
  16 raw self-contact points in one PhysX step, all finite. The seeded
  `velocity_flat_parity_probe_16_contract_v2.json` passes four reset cycles
  across 16 envs with bounded relative root pose/yaw, zero joint-default error,
  61D actor observations, BAM
  delay/voltage/sag/friction ranges, and finite 3/4/6D command terms. The
  previous filtered ContactSensor path remains disabled because its nested USD
  expansion is not multi-env safe. A concrete 4-env sensor-delay runtime probe
  injects a changing yaw-rate signal and observes finite delayed gyro values
  matching the previous control step (`sensor_delay_runtime_probe_4.json`).
completed: >-
  Isolated IsaacLab 3.0.0 runtime and launchers; generated walk USD and asset
  reports; named 61D actor / 76D privileged critic / 14D action contracts;
  explicit BAM actuator with action clipping, 3..6-step delay, per-env voltage,
  previous-motor-effort sag, and friction-scale state; reset/DR, sensor
  corruption, commands, turn bucket, rewards, terminations, and curricula are
  wired with CPU contracts. Subtree angular momentum is computed from body
  COM/mass/inertia/velocity tensors. `rsl-rl-lib 5.0.1` was attempted and is
  incompatible with the IsaacLab 3.0 entrypoint; the supported 5.4.1 stack is
  recorded as a delta. The incorrect IsaacLab-only right-hip-yaw HOME value and
  simulator clamp are removed and protected by a direct mjlab-source test.
  Foot height/slip/clearance now use the canonical MJCF sites. Production
  self-collision reward uses concrete raw PhysX views and a 4-env task smoke is
  finite; direct and historical probe startup failures were traced to incomplete
  PYTHONPATH rather than simulator semantics.
next_action: >-
  Complete P0 source/effect fixtures for BAM friction dynamics, command
  resampling, and reward/termination/critic values.
  Do not start the unified command battery until those rows are closed; then
  run the fresh 64-env/5-iteration PPO smoke.
next_proof: >-
  Seeded runtime effect reports for friction/asset dynamics, followed by
  command-resampling statistics and numerical
  reward/termination/critic fixtures. The final training gate is the unified
  command battery followed by a fresh 64-env/5-iteration smoke.
stop_condition: >-
  Do not start VelStand or a new 4096-env/6000-iteration run while any P0 ledger
  row is BLOCKED/NOT_STARTED, or before the unified battery and subsequent smoke
  both pass. Old long runs are diagnostic only and never a strict baseline.
no_touch_scope: existing mjlab behavior, uv.lock, production runtime, old long-run artifacts
parked_todos: >-
  MuJoCo/PhysX solver behavior and unavailable same-step solved external-load
  torque remain explicit backend limitations. The old long-run checkpoints are
  retained only for diagnosis. VelStand and all later task ports remain deferred.
