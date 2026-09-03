status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: implement the IsaacLab backend plan via intuitive-flow
current_slice: Task H Velocity-Flat battery and forward-only mjlab comparison complete; long-run decision pending
blocker_kind: evaluation_battery
  blocker_fingerprint: mjlab comparison and external-load friction parity are not implemented
last_proven_evidence: >-
  The formal `microduck-isaaclab:3.0.0-isaacsim6.0.1` image runs the pinned
  IsaacLab `release/3.0.0` source at commit
  `c7fd163736878a4a348a63880ff6001ea8b3143e` with Isaac Sim 6.0.1, Torch
  2.10.0+cu128, Warp 1.16.0, and Newton 1.5.1. The repository RL launcher
  pre-registers Microduck tasks before dispatching the official
  `run_train_cli`/`run_play_cli`. A 5-iteration PPO smoke passes (7,680 steps,
  finite 61D observations and 14D actions). Official playback loads
  `model_4.pt`, exports JIT/ONNX, completes rollout, and exits 0. With `--viz
  kit`, the recorder writes 32 frames to
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/2026-09-03_13-02-22/videos/play/clip_0000.mp4`.
  The warning about `/World/envs/env_0/Robot/Geometry/trunk_base/trunk_base`
  remains but does not block reset, policy execution, or video capture.
  The converted USD still reports `drive_configured=false`: BAM is explicit
  effort control, not implicit PhysX PD.
completed: >-
  Added isolated runtime and Docker launchers, pinned IsaacLab source checkout,
  lazy package/task registry, 61D/14D policy ABI with golden fixture, MJCF
  mechanical report, current Isaac Sim MJCF conversion/diagnostic path, a
  machine-readable USD inspection script, and a Torch BAM XL330/M6 numerical
  core and an explicit IsaacLab `BamActuatorCfg`/`BamActuator` wrapper. The
  225-point voltage/torque/friction grid matches the reference BAM
  implementation to floating-point precision (max error 1.4e-16). The wrapper
  imports and the Microduck asset config resolves in the actual 6.0.1 runtime;
  a 2-environment `BamActuator.compute()` smoke also returns finite efforts.
  CPU/runtime contracts: 30 passed. InteractiveScene probing identified that
  IsaacLab 3.0 beta also requires the bundled
  `/workspace/IsaacLab/source/isaaclab_contrib` path on PYTHONPATH. PhysX
  articulation loading additionally probes `isaaclab_newton`, so the runtime
  command now includes the IsaacLab `newton`, `ov`, and `ovphysx` source paths.
  The standard IsaacLab `AppLauncher` path reaches `sim_reset`,
  `articulation_ready`, and `step_ok` for both bundled Cartpole and the
  Microduck USD. The real Microduck articulation resolves 14 joints with
  `BamActuator` and produces finite efforts. The dynamic BAM bench report
  `.cache/isaaclab-assets/bam_dynamic_bench.json` covers 20-step 7.5V and
  6.5V target steps plus a 7.5V sinusoid; all samples are finite. The report
  explicitly records `friction_bridge=not_applied_to_physx`.
  The deterministic battery now runs on the real USD with a post-clone ground
  plane workaround required by Isaac Sim 6.0.1. Its 200-step subset covers
  home settle, free fall, target step, and NaN soak; all samples are finite.
  It applies the motor-only friction bridge and records the limitation as
  `motor_only_external_effort_unavailable`. All four cases are finite. With
  the canonical HOME target, `home_settle` and `nan_soak` settle at root z
  about 0.113 m with max tilt about 0.222 rad; free fall reaches z about
  0.106 m and max tilt about 0.669 rad. The quaternion check uses IsaacLab's
  xyzw layout.
  The imported USD root discovery issue is fixed by explicitly configuring
  `articulation_root_prim_path="/Geometry/trunk_base"`; the probe reaches
  `sim_reset`, `articulation_ready`, and `step_ok` with 14 joints. Policy-to-
  PhysX joint ordering is explicitly mapped and covered by a CPU golden test.
  The fixed-root sweep shows monotonic response across friction scales
  0.5/1.0/1.5: static friction 0.0181/0.0362/0.0543 N-m and peak speed
  1.217/1.071/0.884 rad/s.
  Added an IsaacLab ManagerBasedRLEnv Velocity-Flat task with canonical HOME
  action offsets, named PhysX-to-policy joint ordering, native velocity
  commands, and a 61D policy observation group. The task-only simulator spawn
  HOME clamps right_hip_yaw to its authored 0.436 rad limit while preserving
  the hardware HOME in the policy ABI. The staged smoke harness passes real
  Isaac Sim 6.0.1 reset and random-action stepping for one robot and 64
  environments; both reports are finite with 61D observations and 14D actions.
  The 64-env run records 4 controlled episode resets in 20 steps. Isaac Sim's
  entity ground-plane clone issue is handled by injecting the plane after scene
  cloning, matching the physics battery workaround. The official RSL-RL
  trainer then completes 5 PPO iterations / 7,680 steps with finite metrics and
  writes model_0.pt and model_4.pt under the ignored logs path.
  A first 4096-environment / 1000-iteration Velocity-Flat run also completed
  in about 586 seconds (98,304,000 environment steps) and produced
  `logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/2026-09-03_12-06-46/model_999.pt`.
  The run remained finite and reached about 841-step mean episodes with 0.183
  fallen fraction, but XY/yaw tracking errors remained about 0.399 m/s and
  1.11 rad/s and the task success rate stayed zero. These are training-chain
  signals, not evidence of a reliable gait or simulator parity. The fixed-seed
  command battery now completes on the same smoke checkpoint with 16
  environments and 250 steps for zero, forward, lateral, and yaw commands.
  All tensors are finite. Reset fractions are 0.0205, 0.0213, 0.0208, and
  0.0208; mean XY errors are 0.0911, 0.1421, 0.2493, and 0.0938 m/s. The
  lateral case is the weakest tracker and maximum tilt is about 0.85 rad in
  every case. See `docs/isaaclab_velocity_flat_command_battery_report.md`.
next_action: >-
  Decide whether the measured smoke-checkpoint behavior justifies another
  IsaacLab training run. The forward-only comparison is recorded in
  `docs/isaaclab_velocity_flat_backend_comparison.md`; it shows the accepted
  mjlab rollout staying upright while the IsaacLab checkpoint resets about
  2.1% of environments per step and reaches about 0.85 rad tilt. Keep the USD
  `drive_configured=false` finding visible: BAM is explicit effort control, not
  implicit PhysX PD.
next_proof: >-
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest
  tests/test_isaaclab_velocity_flat_contract.py tests/test_isaaclab_policy_joint_mapping.py
  tests/test_isaaclab_asset_cfg.py tests/test_isaaclab_friction_sweep_contract.py`,
  plus `.cache/isaaclab-assets/velocity_flat_smoke_1.json`,
  `.cache/isaaclab-assets/velocity_flat_smoke_64_prestartup.json`, and the
  smoke run under `logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/`.
  The container dependency probe is green with `torch==2.10.0+cu128`,
  `tensordict==0.10.0`, and `rsl_rl==5.0.1`.
stop_condition: >-
  Do not claim simulator parity or reliable walking until the fixed command
  battery, mjlab comparison, and PhysX drive/BAM behavior are explicitly
  documented. Do not start another long PPO run without a new Task H decision.
no_touch_scope: existing mjlab code, dependency lock, production runtime
parked_todos: >-
  External-load friction parity remains unresolved; a longer IsaacLab run is
  pending an explicit Task H resource/experiment decision; DCMotorCfg is not
  accepted as BAM parity.
