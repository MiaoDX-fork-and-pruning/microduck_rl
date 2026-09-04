status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: reproduce mjlab Velocity-Flat semantics in IsaacLab via intuitive-flow
current_slice: concrete multi-env self-contact adapter wired; seeded runtime parity closure next
blocker_kind: implementation_and_parity_evidence
blocker_fingerprint: P0 asset-friction, reset/DR, actor-sensor, command evidence, and forced-contact runtime evidence remain open
last_proven_evidence: >-
  `velocity_flat_home_smoke_1.json` runs the real IsaacLab 3.0.0 / Isaac Sim
  6.0.1 articulation: policy-ordered default HOME has max absolute error 0.0,
  right_hip_yaw is 0.0, actor/action shapes are 61D/14D, critic wiring is 76D,
  and all 15 contact-body tensors are finite. The one-environment contact report
  proves finite filtered self-contact matrices, nonzero foot-ground force, all
  16 finite reward terms, and finite termination terms. A direct mjlab-kernel
  regression proves equal self-contact point-count semantics. Concrete raw
  PhysX views now initialize for four environments with trunk/leg count buffers
  `(4,2)` and `(4,1)`; a 4-env task smoke and raw adapter count unit fixture
  pass. The previous filtered ContactSensor path remains disabled because its
  nested USD expansion is not multi-env safe.
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
  finite; a forced-contact runtime numerical fixture is still required.
next_action: >-
  Add a forced-contact runtime numerical fixture for the concrete raw PhysX
  adapter, then run seeded IsaacLab reset/DR/noise/delay/command probes and
  update each ledger row only when its runtime observable matches. After all P0
  rows close, run the unified command battery and a fresh 64-env/5-iteration PPO
  smoke.
next_proof: >-
  A seeded runtime parity report following the completed focused CPU/contact
  runtime contracts. The final training gate is the unified
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
