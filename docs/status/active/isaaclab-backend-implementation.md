status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: reproduce mjlab Velocity-Flat semantics in IsaacLab via intuitive-flow
current_slice: fixed-root same-state BAM dynamics proof and accepted-policy IsaacLab A/B diagnosis completed; strict trained-policy battery still blocks forward/lateral response
blocker_kind: implementation_and_parity_evidence
blocker_fingerprint: corrected strict policy fails forward/lateral command-response gates while accepted mjlab policy passes the same IsaacLab battery
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
  (`reward_termination_runtime_fixture_4.json`). The seeded contact numerical
  fixture covers 4-env ground and airborne-to-contact cases with non-zero foot
  forces, finite air-time/site-speed values, and <=1e-5 manager-vs-captured-
  tensor formula parity (`logs/contact_numerical_fixture.json`). Production
  PhysX material writes/readback for both ankle bodies are finite, in-range,
  selective, and differ across fixed low/high cases
  (`foot_material_runtime_probe.json`).
  The production push event proof covers a fixed-seed selected subset, finite
  exact velocity deltas, untouched environments, and reset non-accumulation
  (`push_runtime_probe.json`). The bounded directional trace records exact
  command observation tails for zero/forward/lateral/yaw, raw policy actions,
  policy-order processed actions, simulator-order BAM targets, delayed targets,
  applied efforts, and body-frame root velocities across four environments and
  24 control steps (`velocity_flat_directional_trace_4x24.json`). It shows no
  command-slot or action-boundary mismatch; forward/lateral remain a trained
  behavior failure under the current PhysX/BAM dynamics path.
  The reset-fix regression battery remains finite with zero resets and the
  same forward/lateral-only response failures (`velocity_flat_command_battery_resetfix_strict.json`).
  The backend-neutral fixture now compares the IsaacLab primitives directly
  with mjlab's `DelayBuffer` and BAM TorchBackend over a seeded 10-step,
  4-env sequence with an env-subset reset: target lag is identical step for
  step, target max error is 0, voltage-sag max error is 2.73e-7 V, and motor
  torque max error is 8.61e-8 N-m (`velocity_flat_backend_neutral_fixture_4x10.json`).
  The fixed-root same-state dynamics proof compares neutralized IsaacLab PhysX
  against MuJoCo BAM-solver and motor-only references for step and sine targets
  at dt=0.005, fixed HOME/root, three-step delay, 7.5 V supply, and 0.1 sag
  gain. It is finite, aligns delayed targets to <=2.98e-9, and shows IsaacLab
  qdot closer to motor-only than BAM-solver (step 0.0818 vs 0.4020 rad/s;
  sine 0.0060 vs 0.1503), confirming the explicit friction timing and solver
  `BACKEND_DELTA` rather than an action/target mismatch
  (`fixed_root_dynamics_parity_1x12.json`).
  The accepted mjlab `model_5999.pt` was then loaded into the same IsaacLab
  runner and six-case battery: all cases pass, with forward mean velocity
  `(0.118,0.011) m/s` and lateral `(-0.040,0.068) m/s`; this separates the
  remaining strict-checkpoint behavior failure from command/action wiring or
  BAM target semantics (`velocity_flat_command_battery_mjlab_policy_p0.json`).
completed: >-
  Isolated IsaacLab 3.0.0 runtime and launchers; generated walk USD and asset
  reports; named 61D actor / 76D privileged critic / 14D action contracts;
  explicit BAM actuator with mjlab-unclipped action boundary, 3..6-step delay, per-env voltage,
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
  PYTHONPATH rather than simulator semantics. The production mjlab runner's
  `clip_actions=None` boundary is now matched in IsaacLab; the prior strict
  checkpoint's Isaac-only `clip_actions=1.0` was invalidated. Corrected
  64-env/5-iteration smoke and `[1,61] -> [1,14]` ONNX export pass in
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-04_18-31-53/`.
  The corrected strict run completed `4096` environments and `6000`
  iterations in `.../2026-09-04_18-52-35_strict_unclipped/`; its final
  checkpoint is finite and uses `clip_actions: null`. The fixed six-case
  battery is finite with zero resets and tilt <=0.375 rad, but forward and
  lateral command response fail while zero/yaw/turn cases pass:
  `.cache/isaaclab-assets/velocity_flat_command_battery_strict_unclipped.json`.
next_action: >-
  Keep the strict trained-policy forward/lateral gate blocked. Do not tune
  rewards/PPO or launch another long run; the accepted-policy A/B proves the
  runtime can express the directional behavior. Any further work must inspect
  strict-run training provenance or reproduce the accepted policy's training
  recipe, while retaining solver/friction as explicit backend delta.
next_proof: >-
  Completed: accepted-policy A/B battery in IsaacLab, with finite all-case
  response and zero resets, in
  `.cache/isaaclab-assets/velocity_flat_command_battery_mjlab_policy_p0.json`.
stop_condition: >-
  Do not start VelStand. Do not launch another long run while the corrected
  strict policy fails the forward/lateral battery; do not tune reward/PPO to
  hide behavior differences. MuJoCo/PhysX solver and same-step external-load
  friction timing remain explicit backend limitations.
no_touch_scope: existing mjlab behavior, uv.lock, production runtime, old long-run artifacts
parked_todos: >-
  MuJoCo/PhysX solver behavior and unavailable same-step solved external-load
  torque remain explicit backend limitations. The old long-run checkpoints are
  retained only for diagnosis. VelStand and all later task ports remain deferred.
