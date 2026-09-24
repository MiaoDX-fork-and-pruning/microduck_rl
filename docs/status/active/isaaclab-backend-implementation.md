status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: reproduce the historical IsaacLab adapted Velocity-Flat walking profile while retaining the strict MJLab-parity task
current_slice: T26 deployment validation completed after current IsaacLab train/restore/evaluate verification, T25 RSL-RL probe, T24 sensor trace, T22 strict acceptance, and T23 bridge rejection; model_999 passes both the IsaacLab live-manager and CPU MuJoCo deployment batteries
blocker_kind: none
blocker_fingerprint: independent strict-from-scratch gate closed by T22 model_500.pt and retained by later strict checkpoints
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
  A simulator-free audit confirms strict checkpoints `model_1250`, `model_5000`,
  and `model_5999` have finite 61D actor / 76D critic normalizers, identical
  tensor schemas and parameter counts, and the expected serialized normalizer
  fields (`velocity_flat_checkpoint_audit_p0.json`). A two-way normalizer/weight
  cross-evaluation under the same IsaacLab six-case battery shows that replacing
  only normalizer statistics does not recover forward/lateral response:
  strict weights with the accepted normalizer produce exploding actions/resets,
  while accepted weights with the strict normalizer remain finite but fail those
  two cases (`velocity_flat_normalizer_weight_cross_eval_p0.json`).
  A bounded official RSL-RL resume test loaded adapted `model_750.pt` into the
  unchanged strict task and continued 250 iterations at 4096 environments.
  The resulting `model_999.pt` passes the standard six-case battery for 300
  steps and 16 environments with zero resets, forward/lateral means
  `0.153/0.081 m/s`, yaw `0.314 rad/s`, and max tilt `0.570 rad`:
  `velocity_flat_command_battery_warmstart_adapted750_strict_999_300.json`.
  Its 61D/76D schema and finite normalizers pass
  `velocity_flat_checkpoint_audit_warmstart_model999.json`; official play
  exports a finite `[1,61] -> [1,14]` ONNX graph under the warm-start run.
  The same ONNX passes the finite headless CPU MuJoCo six-case battery with
  zero resets, body-frame forward/lateral/yaw response, and max tilt below
  `0.08 rad` (`mujoco_onnx_battery_warmstart_model999.json`).
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
  Treat T22 `model_500.pt` as the first independently strict-trained
  all-six-passing checkpoint, with `model_750.pt` and `model_999.pt` as later
  strict-stage passes. Preserve the historical adapted and warm-start
  checkpoints as separate provenance. No additional long run is justified
  unless a new bounded hypothesis is recorded. T23's one-step-lag PhysX
  friction bridge passed the fixed-root and 100-step probes but failed the
  canonical 300-step `turn_left` case with a tilt reset, so it is rejected for
  production and must remain diagnostic-only. Simulator-side deployment
  validation is complete; the next gate is a controlled hardware-transfer
  review. Matched frame/order signals are closed; foot velocity, subtree
  momentum, and trainer package version remain explicit backend deltas.
next_proof: >-
  Completed: adapted run `microduck_isaaclab_velocity_flat_adapted/2026-09-08_09-08-31`
  finished normally. Fixed six-case battery passed for `model_500`, `750`,
  `1000`, `1250`, and `1500`; `model_2000` and `model_5999` failed only the
  translation response cases. Detailed artifacts and hashes are recorded in
  `docs/isaaclab_velocity_flat_adapted_profile.md`. Follow-up alignment probes
  completed 2026-09-11: the four-env native damping/friction audit remains
  exactly zero under BAM, and fixed-root solver points `4/1` and `8/2` show no
  calibration improvement. Fresh evidence is recorded in
  `.cache/isaaclab-assets/asset_dynamics_runtime_probe_handoff_4.json`,
  `fixed_root_dynamics_handoff_4x1.json`, and
  `fixed_root_dynamics_handoff_8x2.json`. The current checkout also passed the
  64-env/5-step finite smoke and a five-iteration official IsaacLab RSL-RL
  training run; checkpoints `model_0.pt` and `model_4.pt` were written under
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_09-47-26/`.
  Reloading `model_4.pt` stayed finite with zero resets in the short battery;
  its early-training command failures are diagnostic only. A fresh strict
  `model_750.pt` directional trace shows correct command tails, nonzero raw
  actions/BAM efforts, and repeated falling resets; this classifies the
  translation failure as learned stabilization/dynamics behavior rather than
  an ABI or command/action suppression issue. As a cross-profile control,
  adapted `model_750.pt` passes all six 100-step battery cases inside the
  strict task with zero resets, forward `0.139 m/s`, lateral `0.059 m/s`, and
  tilt below `0.11 rad`. A bounded strict warm-start from that checkpoint
  completed at `model_999.pt`; its 300-step six-case battery passes all cases
  with zero resets, forward/lateral `0.153/0.081 m/s`, and official ONNX export.
  The independent strict-from-scratch `model_250.pt` was then evaluated with
  the same six-case battery and failed all cases: it remains finite but resets
  at about 0.16 per environment step and drifts negatively under zero command.
  Evidence is `velocity_flat_command_battery_strict_model250.json` (SHA256
  `ba017cc70e3e4605048bc0e469a1c9ad7c7948c761dd292872d6fcc9f94a807e`).
  A bounded strict training calibration changed only `air_time.weight` from
  3.0 to 1.0 for 750 iterations at 4096 environments. Its `model_749.pt`
  remains finite and stable but fails forward/lateral response and settles near
  0.071 m root height; zero/yaw/turn cases pass. Evidence:
  `velocity_flat_command_battery_strict_airtime1_model749.json`.
  A second bounded calibration disabled only `env.terminations.root_height`
  for 750 iterations, then restored the guard for evaluation. Its finite
  `model_749.pt` fails all six cases: zero command has 776 resets (0.1617 per
  env-step) and -0.199 m/s drift, while commanded translation/yaw also fail.
  Evidence:
  `velocity_flat_command_battery_strict_rootbootstrap_model749.json` (SHA256
  `3d764508b657d657346c990edd5f7cb882b8d6b28a31262c53e182773118c256`).
  T13's three canonical strict battery artifacts are
  `velocity_flat_command_battery_t13_action_rate_flat_model250.json`,
  `velocity_flat_command_battery_t13_action_rate_flat_model500.json`, and
  `velocity_flat_command_battery_t13_action_rate_flat_model749.json`.
  T14 then changed only strict `track_ang_vel.weight` from `2.0` to `6.0` for
  750 iterations at 4096 environments. Its finite checkpoints were evaluated
  under the canonical task: all pass zero/yaw/turn-left/turn-right but fail
  forward/lateral response. Evidence is
  `velocity_flat_command_battery_t14_angvel6_model250.json`,
  `velocity_flat_command_battery_t14_angvel6_model500.json`, and
  `velocity_flat_command_battery_t14_angvel6_model749.json`; the run is
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_13-55-29_t14_angvel6_750/`.
  T15 then changed only strict `track_lin_vel.weight` from `2.0` to `4.0` for
  750 iterations at 4096 environments. Its finite checkpoints pass
  zero/forward/lateral/turn-left but fail yaw/turn-right. Evidence is
  `velocity_flat_command_battery_t15_linvel4_model250.json`,
  `velocity_flat_command_battery_t15_linvel4_model500.json`, and
  `velocity_flat_command_battery_t15_linvel4_model749.json`; the run is
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_14-34-25_t15_linvel4_750/`.
  T16 then changed only the tracking pair to `track_lin_vel=4.0` and
  `track_ang_vel=6.0` for 750 iterations. Its checkpoints pass
  zero/forward/lateral, with positive yaw absent and turn-left present from
  `model_500.pt` onward. Evidence is
  `velocity_flat_command_battery_t16_tracking_pair_model250.json`,
  `velocity_flat_command_battery_t16_tracking_pair_model500.json`, and
  `velocity_flat_command_battery_t16_tracking_pair_model749.json`; the run is
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_15-09-43_t16_tracking_pair_750/`.
  T17 retained T16 and added the 61D left-right symmetry transform/data
  augmentation. CPU contracts and the five-iteration smoke passed; its
  750-iteration run remained finite. `model_500.pt` passes zero/forward/
  lateral/turn-left but fails yaw/turn-right; `model_749.pt` passes zero/yaw/
  turn-left/turn-right but fails lateral. Evidence is
  `velocity_flat_command_battery_t17_symmetry_model250.json`,
  `velocity_flat_command_battery_t17_symmetry_model500.json`, and
  `velocity_flat_command_battery_t17_symmetry_model749.json`; the run is
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_symmetry/2026-09-11_15-51-08_t17_symmetry_tracking_pair_750/`.
  T18 retained the symmetry task but its manifest used 2.0/2.0 tracking
  weights, so its dense scan is invalid for the T17 pair. T19 corrected this
  to 4.0/6.0 with the same 50-iteration checkpoint cadence; its pass vectors
  were `FFFFFF`, `PPFFFF`, `PPPFFP`, `PPPPFP`, `PPPFFF`, `PPFPFF`, `PPPFPF`,
  `PPPFPP`, and `PPFPPP`, with no all-six checkpoint. Evidence is
  `.cache/isaaclab-assets/velocity_flat_command_battery_t19_exact_pair_model*.json`;
  the run is `logs/rsl_rl/microduck_isaaclab_velocity_flat_symmetry/2026-09-11_17-26-30/`.
  T22 then completed a 1000-iteration 4096-env from-scratch
  adapted-to-strict curriculum. The required smoke passed, no NaN terminations
  occurred, and battery vectors were `PPFPFP` (250), `PPPPPP` (500),
  `PPPPPP` (750), and `PPPPPP` (999). Evidence is in
  `.cache/isaaclab-assets/velocity_flat_smoke_t22_strictification_64.json`,
  `.cache/isaaclab-assets/velocity_flat_command_battery_t22_strictification_model{250,500,750,999}.json`,
  and `logs/rsl_rl/microduck_isaaclab_velocity_flat_strictification/2026-09-11_19-26-41/`.
  The diagnostic-only lagged PhysX friction probe then completed six finite
  fixed-root cases with zero reset-buffer leakage, zero first-step external
  effort, and ordered 14-joint projected-minus-actuation loads:
  `.cache/isaaclab-assets/lagged_friction_bridge_probe.json` (SHA256
  `73a70df7303402e6a1ce78a79b57aa9881df0c97250fda3a8d19fb3deab99542`).
  T23's 100-step lagged walking battery passed all six cases, but the canonical
  300-step battery failed `turn_left` with one tilt reset and
  `max_tilt_rad=1.08818`; the other five cases passed. The rejection artifact
  is `.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_one_step_lag_300.json`
  (SHA256 `ac4702d6b84b010149d1457a8f52df5cadfdf8d1a2de3e493aad3de0e1440263`).
  T26 then reran the official current-code IsaacLab entrypoint at 64 envs for
  5 PPO iterations, writing `model_0.pt` and `model_4.pt` under
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_21-07-34/`.
  The fresh smoke stayed finite with no `nan_state` terminations; its 100-step
  reload battery is intentionally motion-quality-failing as an untrained
  checkpoint. The current-code replay of T22 `model_999.pt` passes all six
  300-step cases with zero resets and max tilt `0.1784 rad` in
  `.cache/isaaclab-assets/velocity_flat_command_battery_current_strict_model999_300.json`.
stop_condition: >-
  Do not use adapted `model_5999.pt` or the historical warm-start checkpoint
  as substitutes for the independently strict result. T22 `model_500.pt`,
  `model_750.pt`, and `model_999.pt` pass the current six-case battery under
  the strictified task; `model_999.pt` also passes the formal ONNX and CPU
  MuJoCo deployment rehearsals. Hardware-transfer review and a controlled
  robot test remain outstanding. Do not launch another long run without a new
  bounded hypothesis.
  MuJoCo/PhysX solver and same-step external-load friction timing remain
  explicit backend limitations.
no_touch_scope: existing mjlab behavior, uv.lock, production runtime, old long-run artifacts
parked_todos: >-
  T22 strict `model_999.pt` ONNX export and finite headless MuJoCo directional
  battery are complete; optional video capture needs an image with
  `omni.replicator`. Preserve all generated battery/audit JSON under `.cache/`
  as local evidence. MuJoCo/PhysX solver behavior and unavailable same-step
  solved external-load torque remain explicit backend limitations. VelStand and
  all later task ports remain deferred.
