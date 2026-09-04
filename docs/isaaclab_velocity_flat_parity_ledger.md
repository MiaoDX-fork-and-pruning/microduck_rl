# IsaacLab Velocity-Flat Parity Ledger

Status: active strict semantic closure.  `MATCHED` means the implementation
and a deterministic contract exist; `BACKEND_DELTA` is reserved for solver or
same-step PhysX limitations.  No PPO long run is an acceptance baseline while
any P0 row is `NOT_STARTED` or `BLOCKED`.

| Surface | mjlab source | IsaacLab source | Status | Tolerance / evidence | Owner |
| --- | --- | --- | --- | --- | --- |
| Actor ABI 61D / action 14D | `microduck_velocity_env_cfg.py`, policy ABI | `tasks/velocity_flat.py`, `policy_abi.py` | MATCHED | ABI fixture and shape smoke | task |
| HOME + action scale | mjlab `HOME_FRAME`, JointPositionActionCfg | `policy_abi.HOME_POSITION`, `ActionsCfg` | MATCHED | direct cross-source HOME test, raw target fixture, and `velocity_flat_home_smoke_1.json` (`max_abs_error=0`) | actuator |
| Action clipping | `RslRlVecEnvWrapper` default `clip_actions=None`; no env-side clip | `RslRlVecEnvWrapper(..., clip_actions=None)`, action term `clip=None` | MATCHED | live mjlab cfg/source proof plus opt-in clip helper fixture; prior `model_5999.pt` was trained with an Isaac-only clip and is diagnostic-invalid | actuator |
| BAM target delay | `DelayBuffer(min_lag=3, max_lag=6, update_period=0)`: per-env lag sampled on each actuator command, reset rows cleared and first command backfilled | `BamActuator` + `ControlStepDelay(sample_lag_each_push=True)` | MATCHED | `.cache/isaaclab-assets/velocity_flat_backend_neutral_fixture_4x10.json`: seeded 10-step target/lags match mjlab step-for-step (`max_target_error=0`), including env-1 subset reset at step 5; CPU contract test covers reference `DelayBuffer` | actuator |
| BAM voltage DR | `vin_range=(6.5,8.2)` | `BamActuator` per-env supply | MATCHED | seeded range/floor test | actuator |
| BAM voltage sag | `vin_drop_gain_range=(0,0.2)`, `vin_min=6.0` | `effective_supply_voltage` | MATCHED | pure math fixture | actuator |
| BAM friction scale | `randomize_bam_friction`, friction budget | `randomize_bam_friction` reset event + actuator scale hook | MATCHED | seeded range evidence plus `friction_sweep_contract.json`: scale 0.5/1.0/1.5 writes finite PhysX coefficients and changes velocity response monotonically; external-load torque remains explicit delta | actuator |
| External-load friction timing | same-step solved torque in MuJoCo | PhysX force getters refresh post-step | BACKEND_DELTA | `force_timing_probe.json`; motor-only bridge explicit | backend |
| Asset joint order / limits | walk MJCF | converted `microduck_walk.usd` | MATCHED | MJCF and `microduck_walk.usd.report.json` agree on 14 names/order and limits; runtime HOME is within all limits | asset |
| Asset damping/friction | MJCF damping 0.053; BAM zeroes dof friction | USD import + explicit BAM metadata | MATCHED | `asset_dynamics_runtime_probe_4.json`: concrete runtime joint friction and damping tensors are finite and exactly zero under BAM; solved external-load timing remains the separate `BACKEND_DELTA` row | asset |
| Reset height / HOME | reset z 0.12..0.13, x/y ±0.5, yaw ±3.14, joint offsets (0,0) | `reset_velocity_flat_state`, `EventsCfg` | MATCHED | `velocity_flat_parity_probe_16_contract_v2.json`: 4 seeded resets, relative x/y within ±0.5, z within 0.12..0.13, yaw within ±3.14, and joint default error 0; explicit BAM reset/target seeding prevents cross-episode FIFO leakage (`velocity_flat_directional_trace_resetfix2_4x4.json`) | task |
| CoM/head CoM DR | `dr.body_ipos` add, non-accumulating | reset event + `curriculum_event_range` live manager update | MATCHED | `asset_dynamics_runtime_probe_4.json`: 4-env runtime CoM tensors are finite and change on reset from restored defaults; CPU range/curriculum tests remain | task |
| Mass/inertia DR | `dr.pseudo_inertia` startup | `velocity_flat_dr.randomize_mass_inertia` | MATCHED | `asset_dynamics_runtime_probe_4.json`: runtime mass/inertia tensors are finite and identical across reset (startup-only sample), with coupled writer path covered by CPU tests | task |
| Armature/friction DR | `dr.joint_armature`, BAM friction scale | actuator/asset hooks | MATCHED | `asset_dynamics_runtime_probe_4.json`: runtime armature tensors are finite and change on reset; runtime joint friction is zero; friction scale effect is covered by `friction_sweep_contract.json` | actuator |
| Push / foot friction DR | interval pushes + foot material range | IsaacLab event terms | MATCHED | `push_runtime_probe.json` proves fixed-seed 4-env subset push and reset non-accumulation; `foot_material_runtime_probe.json` proves production PhysX low/high coefficient write/readback for both ankle bodies and selective shape updates. Solver-dependent tangential response remains covered by the explicit physics `BACKEND_DELTA` boundary | task |
| Encoder bias | actor-only `biased=True`, +/-0.015 | `policy_joint_pos` | MATCHED | actor/critic separation and reset-state tests | task |
| IMU noise/misalignment | actor noise + random mounting rotation | root-state adapter corruption | MATCHED | `velocity_flat_parity_probe_16_sensor_v2.json`: clean/corrupt runtime deltas finite across 16 envs × 4 resets; gyro <=0.03 and gravity <=0.082 including 6deg mounting rotation | task |
| Gyro/gravity delay | lag 0..1, update period 64 | state history adapter | MATCHED | `sensor_delay_runtime_probe_4.json`: concrete 4-env PhysX task with injected yaw-rate ramp observes finite dynamic raw signal and delayed output matching the previous control step; CPU warm-up/period tests cover lag 0/1 and reset semantics | task |
| Joint velocity delay/noise | fixed lag 1, +/-0.25 | state history adapter | MATCHED | `velocity_flat_parity_probe_16_sensor_v2.json`: clean/corrupt joint-velocity deltas finite and <=0.25 across seeded resets; control-step history remains CPU-tested | task |
| Head/body commands | 4D + 6D non-zero commands, 2..5 s resampling | `UniformPoseCommand` manager terms, 13D block | MATCHED | `command_runtime_probe_64.json`: 64-env/1500-step runtime counters observe head/body intervals 2.02..5.00 s, finite held 4/6D commands, and command evolution | task |
| Turn-in-place bucket | 15%, lin=0, yaw 0.4..1.0 at every velocity resample | `MicroduckVelocityCommand` | MATCHED | `command_runtime_probe_64.json`: 51/389 sampled base commands (13.11%) have exactly zero linear velocity and yaw magnitude 0.421..0.987; all finite | task |
| Velocity rewards | XY+Z Gaussian std sqrt(.1) | `track_linear_velocity` | MATCHED | `reward_termination_runtime_fixture_4.json`: live manager weighted value equals direct kernel to <=1e-5; 4-env outputs finite | task |
| Angular rewards | yaw+XY Gaussian std sqrt(.5) | `track_angular_velocity` | MATCHED | `reward_termination_runtime_fixture_4.json`: live manager weighted value equals direct kernel to <=1e-5; 4-env outputs finite | task |
| Pose/head rewards | variable leg pose, head tracking, body tracking, head bias | `pose_tracking`, `head_pose_tracking`, `body_pose_tracking`, `head_pose_bias_penalty` | MATCHED | `reward_termination_runtime_fixture_4.json`: pose/head/body manager values match direct kernels (<=1e-5 where active); CPU stage/kernel tests cover variable posture and bias state | task |
| Regularizers | body ang vel, subtree angular momentum, dof limits, action rate | task reward terms | MATCHED | `reward_termination_runtime_fixture_4.json`: active penalties are finite and non-positive; stateless body-angular-velocity, angular-momentum, and joint-limit values match direct weighted kernels to <=1e-5; stateful terms are separately runtime-finite | task |
| Contact rewards | air-time, terrain-relative clearance, swing, slip, self collision | `velocity_flat_contact` + ContactSensor/raw PhysX view | MATCHED | `logs/contact_numerical_fixture.json`: seeded 4-env ground and airborne-to-contact cases have finite foot forces/air-time/site speed and manager-vs-captured-tensor formula parity <=1e-5; raw self-contact count proof remains in `contact_runtime_forced_direct_4.json`. MuJoCo/PhysX trajectory differences remain the separate solver `BACKEND_DELTA` row | task |
| Terminations | timeout, 70deg orientation, terrain bounds, NaN | timeout, 70deg, bounds, NaN | MATCHED | `reward_termination_runtime_fixture_4.json`: exact four termination sources are present, finite, and produce finite booleans; 70-degree boundary and NaN kernels are CPU-tested | task |
| Privileged critic | base lin vel + foot height/air/contact/contact force | separate `critic` observation group | MATCHED | `reward_termination_runtime_fixture_4.json`: live critic shape is `[4,76]`, finite, with command/IMU/contact slots sourced from the dedicated critic group | task |
| PPO implementation | `rsl-rl-lib 5.0.1` | image `rsl-rl-lib 5.4.1` | BACKEND_DELTA | 5.0.1 install attempt incompatible with IsaacLab 3.0; controlled probe/report | training |
| Fixed command battery harness | mjlab 300-step continuous cases | shared `velocity_flat_battery_spec` and IsaacLab harness | MATCHED | six fixed cases run deterministically with seed 2026, 16 envs, 300 steps, finite tensors, and identical command/reset instrumentation; fresh smoke checkpoint artifact: `.cache/isaaclab-assets/velocity_flat_command_battery_fresh.json` | eval |
| Trained-policy battery behavior | mjlab trained-policy acceptance | IsaacLab trained-policy acceptance | BLOCKED | corrected strict run completed `4096` envs / `6000` iterations with `clip_actions=None`: `model_5999.pt` (`db0cbd9c...`). Fixed six-case battery is finite with zero resets and tilt <=0.375 rad, but forward and lateral command response fail (actual XY velocity near zero); zero/yaw/turn-left/turn-right pass. The directional trace confirms exact command tails, unclipped raw actions, canonical-to-simulator joint reorder, finite BAM delay/effort, and no command/action boundary mismatch; the reset-fix trace additionally proves HOME-seeded delayed targets at every case boundary. Artifacts: `.cache/isaaclab-assets/velocity_flat_directional_trace_4x24.json`, `.cache/isaaclab-assets/velocity_flat_directional_trace_resetfix2_4x4.json`, `.cache/isaaclab-assets/velocity_flat_command_battery_resetfix_strict.json`. This is a real trained-policy behavior failure, not a license to tune reward/PPO. | eval |
| Solver/contact behavior | MuJoCo implicitfast | PhysX GPU solver | BACKEND_DELTA | fixed-root same-state BAM dynamics proof (`.cache/isaaclab-assets/fixed_root_dynamics_parity_1x12.json`, finite): target starts align to <=3e-9, while IsaacLab qdot is closer to MuJoCo motor-only than BAM-solver fields (step qdot error 0.0818 vs 0.4020 rad/s; sine 0.0060 vs 0.1503); no reward/PPO compensation | backend |

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
4. The corrected replacement `4096`-environment, `6000`-iteration run has
   completed and exports a valid `[1,61] -> [1,14]` policy. Its trained-policy
   battery remains blocked on forward/lateral response; diagnose the command,
   action, and locomotion dynamics path before any further run. VelStand and
   the prior contaminated long run are excluded.
