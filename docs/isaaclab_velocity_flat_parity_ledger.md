# IsaacLab Velocity-Flat Parity Ledger

Status: T22 strict semantic closure accepted.  `MATCHED` means the
implementation and its acceptance evidence exist; `BACKEND_DELTA` is reserved
for solver or same-step PhysX limitations.  Hardware validation remains a
separate release gate.

| Surface | mjlab source | IsaacLab source | Status | Tolerance / evidence | Owner |
| --- | --- | --- | --- | --- | --- |
| Actor ABI 61D / action 14D | `microduck_velocity_env_cfg.py`, policy ABI | `tasks/velocity_flat.py`, `policy_abi.py` | MATCHED | ABI fixture and shape smoke | task |
| HOME + action scale | mjlab `HOME_FRAME`, JointPositionActionCfg | `policy_abi.HOME_POSITION`, `ActionsCfg` | MATCHED | direct cross-source HOME test, raw target fixture, and `velocity_flat_home_smoke_1.json` (`max_abs_error=0`) | actuator |
| Action clipping | `RslRlVecEnvWrapper` default `clip_actions=None`; no env-side clip | `RslRlVecEnvWrapper(..., clip_actions=None)`, action term `clip=None` | MATCHED | live mjlab cfg/source proof plus opt-in clip helper fixture; prior `model_5999.pt` was trained with an Isaac-only clip and is diagnostic-invalid | actuator |
| BAM target delay | `DelayBuffer(min_lag=3, max_lag=6, update_period=0)`: per-env lag sampled on each actuator command, reset rows cleared and first command backfilled | `BamActuator` + `ControlStepDelay(sample_lag_each_push=True)` | MATCHED | `.cache/isaaclab-assets/velocity_flat_backend_neutral_fixture_4x10.json`: seeded 10-step target/lags match mjlab step-for-step (`max_target_error=0`), including env-1 subset reset at step 5; CPU contract test covers reference `DelayBuffer` | actuator |
| BAM voltage DR | `vin_range=(6.5,8.2)` | `BamActuator` per-env supply | MATCHED | seeded range/floor test | actuator |
| BAM voltage sag | `vin_drop_gain_range=(0,0.2)`, `vin_min=6.0` | `effective_supply_voltage` | MATCHED | pure math fixture | actuator |
| BAM friction scale | `randomize_bam_friction`, friction budget | `randomize_bam_friction` reset event + actuator scale hook | MATCHED | seeded range evidence plus `friction_sweep_contract.json`: scale 0.5/1.0/1.5 writes finite PhysX coefficients and changes velocity response monotonically; external-load torque remains explicit delta | actuator |
| External-load friction timing | same-step solved torque in MuJoCo | PhysX force getters refresh post-step | BACKEND_DELTA | `force_timing_probe.json`; `lagged_friction_bridge_probe.json` proves finite 14-joint projected-minus-actuation loads, reset zeroing, and one-step warm-up across motor-only/lagged fixed-root cases. T23's 100-step lagged and production probes both pass all six cases, but the canonical 300-step lagged battery fails `turn_left` (`max_tilt_rad=1.08818`, one reset; artifact `.cache/isaaclab-assets/velocity_flat_command_battery_t22_model999_one_step_lag_300.json`, SHA256 `ac4702d6b84b010149d1457a8f52df5cadfdf8d1a2de3e493aad3de0e1440263`). Production remains motor-only; the lagged bridge is diagnostic-only and rejected for production parity | backend |
| Asset joint order / limits | walk MJCF | converted `microduck_walk.usd` | MATCHED | MJCF and `microduck_walk.usd.report.json` agree on 14 names/order and limits; runtime HOME is within all limits | asset |
| Asset damping/friction | MJCF damping 0.053; BAM zeroes dof friction | USD import + explicit BAM metadata | MATCHED | `asset_dynamics_runtime_probe_4.json` and fresh `asset_dynamics_runtime_probe_handoff_4.json`: concrete runtime joint friction and damping tensors are finite and exactly zero under BAM; solved external-load timing remains the separate `BACKEND_DELTA` row | asset |
| Reset height / HOME | reset z 0.12..0.13, x/y ±0.5, yaw ±3.14, joint offsets (0,0) | `reset_velocity_flat_state`, `EventsCfg` | MATCHED | `velocity_flat_parity_probe_16_contract_v2.json`: 4 seeded resets, relative x/y within ±0.5, z within 0.12..0.13, yaw within ±3.14, and joint default error 0; explicit BAM reset/target seeding prevents cross-episode FIFO leakage (`velocity_flat_directional_trace_resetfix2_4x4.json`) | task |
| CoM/head CoM DR | `dr.body_ipos` add, non-accumulating | reset event + `curriculum_event_range` live manager update | MATCHED | `asset_dynamics_runtime_probe_4.json`: 4-env runtime CoM tensors are finite and change on reset from restored defaults; CPU range/curriculum tests remain | task |
| Mass/inertia DR | `dr.pseudo_inertia` startup | `velocity_flat_dr.randomize_mass_inertia` | MATCHED | `asset_dynamics_runtime_probe_4.json`: runtime mass/inertia tensors are finite and identical across reset (startup-only sample), with coupled writer path covered by CPU tests | task |
| Armature/friction DR | `dr.joint_armature`, BAM friction scale | actuator/asset hooks | MATCHED | `asset_dynamics_runtime_probe_4.json`: runtime armature tensors are finite and change on reset; runtime joint friction is zero; friction scale effect is covered by `friction_sweep_contract.json` | actuator |
| Push / foot friction DR | interval pushes + foot material range | IsaacLab event terms | MATCHED | `push_runtime_probe.json` proves fixed-seed 4-env subset push and reset non-accumulation; `foot_material_runtime_probe.json` proves production PhysX low/high coefficient write/readback for both ankle bodies and selective shape updates. Solver-dependent tangential response remains covered by the explicit physics `BACKEND_DELTA` boundary | task |
| Encoder bias | actor-only `biased=True`, +/-0.015 | `policy_joint_pos` | MATCHED | actor/critic separation and reset-state tests | task |
| IMU noise/misalignment | actor noise + random mounting rotation | root-state adapter corruption | MATCHED | `velocity_flat_parity_probe_16_sensor_v2.json`: clean/corrupt runtime deltas finite across 16 envs × 4 resets; gyro <=0.03 and gravity <=0.082 including 6deg mounting rotation | task |
| Gyro/gravity delay | lag 0..1, update period 64 | state history adapter | MATCHED | `sensor_delay_runtime_probe_4.json`: concrete 4-env PhysX task with injected yaw-rate ramp observes finite dynamic raw signal and delayed output matching the previous control step; CPU warm-up/period tests cover lag 0/1 and reset semantics | task |
| Raw gyro/gravity source | named MJCF IMU/root-link adapters | root-state gyro/gravity adapters | MATCHED | `.cache/isaaclab-assets/sensor_source_comparison.json`: seeded 12-step trace max errors `8.94e-8` gyro and `3.58e-7` gravity, with delay index `0` documented | task |
| Joint velocity delay/noise | fixed lag 1, +/-0.25 | state history adapter | MATCHED | `velocity_flat_parity_probe_16_sensor_v2.json`: clean/corrupt joint-velocity deltas finite and <=0.25 across seeded resets; control-step history remains CPU-tested | task |
| Raw joint encoder source | MuJoCo joint state in policy order | PhysX articulation joint state explicitly scattered into policy order | MATCHED | `.cache/isaaclab-assets/sensor_source_comparison.json`: 12-step joint position and velocity traces exactly equal at float precision | task |
| Head/body commands | 4D + 6D non-zero commands, 2..5 s resampling | `UniformPoseCommand` manager terms, 13D block | MATCHED | `command_runtime_probe_64.json`: 64-env/1500-step runtime counters observe head/body intervals 2.02..5.00 s, finite held 4/6D commands, and command evolution | task |
| Turn-in-place bucket | 15%, lin=0, yaw 0.4..1.0 at every velocity resample | `MicroduckVelocityCommand` | MATCHED | `command_runtime_probe_64.json`: 51/389 sampled base commands (13.11%) have exactly zero linear velocity and yaw magnitude 0.421..0.987; all finite | task |
| Velocity rewards | XY+Z Gaussian std sqrt(.1) | `track_linear_velocity` | MATCHED | `reward_termination_runtime_fixture_4.json`: live manager weighted value equals direct kernel to <=1e-5; 4-env outputs finite | task |
| Angular rewards | yaw+XY Gaussian std sqrt(.5) | `track_angular_velocity` | MATCHED | `reward_termination_runtime_fixture_4.json`: live manager weighted value equals direct kernel to <=1e-5; 4-env outputs finite | task |
| Pose/head rewards | variable leg pose, head tracking, body tracking, head bias | `pose_tracking`, `head_pose_tracking`, `body_pose_tracking`, `head_pose_bias_penalty` | MATCHED | `reward_termination_runtime_fixture_4.json`: pose/head/body manager values match direct kernels (<=1e-5 where active); CPU stage/kernel tests cover variable posture and bias state | task |
| Regularizers | body ang vel, subtree angular momentum, dof limits, action rate | task reward terms | MATCHED | `reward_termination_runtime_fixture_4.json`: active penalties are finite and non-positive; stateless body-angular-velocity, angular-momentum, and joint-limit values match direct weighted kernels to <=1e-5; stateful terms are separately runtime-finite | task |
| Raw subtree angular momentum source | named MuJoCo `root_angmom` sensor | rigid-body COM/mass/inertia/angular-velocity reconstruction | BACKEND_DELTA | `.cache/isaaclab-assets/sensor_source_comparison.json`: seeded trace max absolute error `7.99e-6`; formula is finite and explicit, but source tensors are not the same named sensor | backend |
| Contact rewards | air-time, terrain-relative clearance, swing, slip, self collision | `velocity_flat_contact` + ContactSensor/raw PhysX view | MATCHED | `logs/contact_numerical_fixture.json`: seeded 4-env ground and airborne-to-contact cases have finite foot forces/air-time/site speed and manager-vs-captured-tensor formula parity <=1e-5; raw self-contact count proof remains in `contact_runtime_forced_direct_4.json`. MuJoCo/PhysX trajectory differences remain the separate solver `BACKEND_DELTA` row | task |
| Raw foot-site velocity source | named MJCF site velocity | PhysX ankle body velocity plus canonical site offset reconstruction | BACKEND_DELTA | `.cache/isaaclab-assets/sensor_source_comparison.json`: seeded trace max absolute error `8.84e-3 m/s`; position source matches to `8.94e-8 m`, so the remaining difference is isolated to rigid-body velocity semantics | backend |
| Terminations | timeout, 70deg orientation, terrain bounds, NaN | timeout, 70deg, bounds, NaN | MATCHED | `reward_termination_runtime_fixture_4.json`: exact four termination sources are present, finite, and produce finite booleans; 70-degree boundary and NaN kernels are CPU-tested | task |
| Privileged critic | base lin vel + foot height/air/contact/contact force | separate `critic` observation group | MATCHED | `reward_termination_runtime_fixture_4.json`: live critic shape is `[4,76]`, finite, with command/IMU/contact slots sourced from the dedicated critic group | task |
| PPO implementation | `rsl-rl-lib 5.0.1` | image `rsl-rl-lib 5.4.1` | BACKEND_DELTA | `.cache/isaaclab-assets/rsl_rl_one_update_{mjlab,isaaclab}.json`: identical seeded synthetic one-update hashes/metrics for the exercised feed-forward path; package PPO/storage source hashes still differ and 5.0.1 is incompatible with the IsaacLab 3.0 entrypoint | training |
| IsaacLab training execution | mjlab runner semantics | official IsaacLab 3.0/RSL-RL entrypoint | MATCHED | Current 64-env/5-iteration smoke writes finite `model_0.pt`/`model_4.pt` with no `nan_state`; current-code T22 `model_999.pt` replay passes all six 300-step cases with zero resets in `.cache/isaaclab-assets/velocity_flat_command_battery_current_strict_model999_300.json` | training |
| Fixed command battery harness | mjlab 300-step continuous cases | shared `velocity_flat_battery_spec` and IsaacLab harness | MATCHED | six fixed cases run deterministically with seed 2026, 16 envs, 300 steps, finite tensors, and identical command/reset instrumentation; fresh smoke checkpoint artifact: `.cache/isaaclab-assets/velocity_flat_command_battery_fresh.json` | eval |
| Trained-policy battery behavior | mjlab trained-policy acceptance | IsaacLab trained-policy acceptance | MATCHED | T22 from-scratch strictification produced `model_500.pt`, `model_750.pt`, and `model_999.pt`, each passing the final strict live-manager six-case IsaacLab battery with zero resets. The accepted `model_999.pt` SHA256 is `2034e7c3f6893f700de2d321746f1267e062aa3d3d29a8ceec4f3005e17e16ce`; its official IsaacLab export is `[1,61] -> [1,14]`, finite, and has ONNX SHA256 `371cb8d92300377363c89b5ada5c7c39683dd1464f93ea620c4c83859b3ce30e`. The same graph passes the 300-step headless MuJoCo six-case battery with zero resets, positive forward/lateral/yaw/turn response, and max tilt below `0.068 rad` (`.cache/isaaclab-assets/mujoco_onnx_battery_t22_strictification_model999.json`). Historical adapted and warm-start checkpoints remain separate provenance; their earlier failures and passes are retained in the handoff as chronological diagnostics. | eval |
| Solver/contact behavior | MuJoCo implicitfast | PhysX GPU solver | BACKEND_DELTA | fixed-root same-state proof remains finite and target-aligned; fresh `fixed_root_dynamics_handoff_4x1.json` (`4/1`) gives step/sine qdot errors `0.4020/0.2396` vs BAM-solver and `0.0856/0.0335` vs motor-only. The `8/2` point is effectively unchanged (`0.4041/0.2405` vs BAM; `0.0824/0.0322` vs motor-only), so production remains `4/1`; no reward/PPO compensation | backend |

T18 checkpoint-window evidence is retained only as an invalid default-pair
diagnostic: its manifest has `track_lin_vel=2.0` and `track_ang_vel=2.0`, so it
does not test T17's `4.0/6.0` hypothesis. The corrected T19 run is valid but
also failed to produce an all-six checkpoint; its dense reports are under
`.cache/isaaclab-assets/velocity_flat_command_battery_t19_exact_pair_model*.json`.

## Software-stack probe

The IsaacLab 3.0.0 image was tested with `rsl-rl-lib==5.0.1`; its RSL-RL
configuration/import contract is incompatible with the IsaacLab 3.0 entrypoint.
The supported image therefore keeps `rsl-rl-lib==5.4.1`.  This changes trainer
implementation details but not task semantics; optimizer/rollout fields remain
matched and the version difference is kept visible in every run manifest.

## Gate order

1. Close P0 action/BAM/asset and reset/DR rows with CPU fixtures. The
   backend-neutral BAM fixture now closes target delay, sag, and motor torque;
   solved external-load friction timing remains the explicit PhysX delta.
2. Run the common fixed command battery harness and deterministic parity tests;
   defer motion-quality gates until a trained strict-parity checkpoint exists.
3. Run `64` environments for `5` iterations and export/shape-check the policy.
   Current checkout evidence: the 64-env finite smoke and official five-
   iteration RSL-RL run completed, writing `model_0.pt` and `model_4.pt` in
   `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-11_09-47-26/`;
   a short reload battery is finite with zero resets.
4. T22 completed the independent strict-trained policy gate with a separate
   adapted-to-strict curriculum. The accepted `model_999.pt` exports a finite
   `[1,61] -> [1,14]` policy and passes the headless MuJoCo six-case rehearsal.
   The prior contaminated long run remains diagnostic-only. Further long runs
   require a new bounded hypothesis; VelStand and later task ports remain
   deferred.
