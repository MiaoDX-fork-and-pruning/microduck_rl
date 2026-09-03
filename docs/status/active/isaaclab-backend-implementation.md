status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: implement the IsaacLab backend plan via intuitive-flow
current_slice: Velocity-Flat direct-RL task and staged runtime smoke
blocker_kind: isaaclab_rsl_rl_dependency_unavailable
  blocker_fingerprint: the pinned Isaac Sim/IsaacLab image has no rsl_rl module
  or rsl-rl-lib distribution; PPO smoke cannot start until the training
  dependency is installed in the image or an approved alternate backend is
  selected
last_proven_evidence: >-
  The derived `microduck-isaaclab:3.0.0-beta2.patch1-isaacsim6.0.1` image
  starts Isaac Sim 6.0.1 headless with CUDA, Python 3.12.13, Torch 2.10.0+cu128,
  and Warp 1.13.0. SimulationContext steps five frames. The current MJCF
  importer API converts both bundled `nv_ant.xml` and the Microduck walk MJCF
  reproducibly. Microduck inspection reports 169 active prims, 15 rigid bodies,
  75 collision geometries, one articulation root at
  `/microduck/Geometry/trunk_base`, all 14 policy joints with limits, and no
  missing joints. It also reports `drive_configured=false`: every converted
  joint has zero PhysX drive stiffness and damping, despite maxForce metadata.
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
  cloning, matching the physics battery workaround.
next_action: >-
  Install or mount a pinned rsl-rl-lib package in the IsaacLab runtime, then run
  the required 64-env / 5-iteration PPO smoke against the new task. Do not
  launch long training. Keep the USD `drive_configured=false` finding visible:
  BAM is explicit effort control, not implicit PhysX PD.
next_proof: >-
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest
  tests/test_isaaclab_velocity_flat_contract.py tests/test_isaaclab_policy_joint_mapping.py
  tests/test_isaaclab_asset_cfg.py tests/test_isaaclab_friction_sweep_contract.py`,
  plus `.cache/isaaclab-assets/velocity_flat_smoke_1.json` and
  `.cache/isaaclab-assets/velocity_flat_smoke_64.json`. The next proof is a
  64-env / 5-iteration PPO smoke after rsl-rl-lib is available.
stop_condition: >-
  Do not claim simulator parity or start long PPO until PhysX drive/BAM behavior
  is explicitly mapped, the deterministic asset/actuator acceptance is green,
  and the 64-env / 5-iteration PPO smoke passes.
no_touch_scope: existing mjlab code, dependency lock, production runtime
parked_todos: >-
  External-load friction parity remains unresolved; plan Tasks H-I (walking
  run and continuation decision) remain pending; DCMotorCfg is not accepted as
  BAM parity; rsl-rl-lib image installation is pending.
