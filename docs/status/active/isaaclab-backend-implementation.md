status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: reproduce mjlab Velocity-Flat semantics in IsaacLab via intuitive-flow
current_slice: forced self-contact runtime proof and seeded reset/DR/command contracts complete; remaining P0 dynamics/source rows open
blocker_kind: implementation_and_parity_evidence
blocker_fingerprint: P0 foot-material boundary, contact numerical fixture and fixed command battery remain open
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
  The asset dynamics probe also observes finite reset-time armature/CoM
  changes, stable startup mass/inertia, and zero runtime joint friction across
  four environments (`asset_dynamics_runtime_probe_4.json`).
  The same probe confirms BAM runtime joint damping and friction tensors are
  exactly zero, separating the imported USD nominal fields from the actuator
  model.
  The command runtime probe observes the configured 2..5 s pose-command and
  3..8 s velocity-command intervals over 64 environments, plus a 13.11%
  turn-in-place sample fraction with zero linear velocity and bounded yaw
  (`command_runtime_probe_64.json`).
  The reward/termination fixture confirms the exact 16 reward and 4
  termination source sets, finite non-positive penalties, <=1e-5 direct-kernel
  agreement for stateless terms, and finite 61D/76D actor/critic outputs
  (`reward_termination_runtime_fixture_4.json`).
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
  Complete P0 source/effect fixtures for the remaining asset boundaries and
  contact force/slip numerical behavior; then run the fixed command battery.
  Do not start the unified command battery until those rows are closed; then
  run the fresh 64-env/5-iteration PPO smoke.
next_proof: >-
  Seeded runtime effect reports for the remaining asset boundaries, followed by
  contact numerical fixtures and the fixed command battery. The final training gate is the unified
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
