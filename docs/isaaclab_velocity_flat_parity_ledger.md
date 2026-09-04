# IsaacLab Velocity-Flat Parity Ledger

Status: active strict semantic closure.  `MATCHED` means the implementation
and a deterministic contract exist; `BACKEND_DELTA` is reserved for solver or
same-step PhysX limitations.  No PPO long run is an acceptance baseline while
any P0 row is `NOT_STARTED` or `BLOCKED`.

| Surface | mjlab source | IsaacLab source | Status | Tolerance / evidence | Owner |
| --- | --- | --- | --- | --- | --- |
| Actor ABI 61D / action 14D | `microduck_velocity_env_cfg.py`, policy ABI | `tasks/velocity_flat.py`, `policy_abi.py` | MATCHED | ABI fixture and shape smoke | task |
| HOME + action scale | mjlab `HOME_FRAME`, JointPositionActionCfg | `policy_abi.HOME_POSITION`, `ActionsCfg` | MATCHED | direct cross-source HOME test, raw target fixture, and `velocity_flat_home_smoke_1.json` (`max_abs_error=0`) | actuator |
| Action clipping | RSL-RL wrapper clip=1.0 | `parity.clip_policy_action`, battery wrapper | MATCHED | exact `[-1,1]` fixture; runtime training wrapper proof still required | actuator |
| BAM target delay | `delay_min_lag=3`, `delay_max_lag=6` | `BamActuator` FIFO | MATCHED | deterministic queue test | actuator |
| BAM voltage DR | `vin_range=(6.5,8.2)` | `BamActuator` per-env supply | MATCHED | seeded range/floor test | actuator |
| BAM voltage sag | `vin_drop_gain_range=(0,0.2)`, `vin_min=6.0` | `effective_supply_voltage` | MATCHED | pure math fixture | actuator |
| BAM friction scale | `randomize_bam_friction`, friction budget | `randomize_bam_friction` reset event + actuator scale hook | MATCHED | seeded range evidence plus `friction_sweep_contract.json`: scale 0.5/1.0/1.5 writes finite PhysX coefficients and changes velocity response monotonically; external-load torque remains explicit delta | actuator |
| External-load friction timing | same-step solved torque in MuJoCo | PhysX force getters refresh post-step | BACKEND_DELTA | `force_timing_probe.json`; motor-only bridge explicit | backend |
| Asset joint order / limits | walk MJCF | converted `microduck_walk.usd` | MATCHED | MJCF and `microduck_walk.usd.report.json` agree on 14 names/order and limits; runtime HOME is within all limits | asset |
| Asset damping/friction | MJCF damping 0.053; BAM zeroes dof friction | USD import + explicit BAM metadata | BLOCKED | runtime USD still carries nominal joint friction; same-step BAM external-load bridge unavailable | asset |
| Reset height / HOME | reset z 0.12..0.13, x/y ±0.5, yaw ±3.14, joint offsets (0,0) | `reset_velocity_flat_state`, `EventsCfg` | MATCHED | `velocity_flat_parity_probe_16_contract_v2.json`: 4 seeded resets, relative x/y within ±0.5, z within 0.12..0.13, yaw within ±3.14, and joint default error 0 | task |
| CoM/head CoM DR | `dr.body_ipos` add, non-accumulating | reset event + `curriculum_event_range` live manager update | MATCHED | `asset_dynamics_runtime_probe_4.json`: 4-env runtime CoM tensors are finite and change on reset from restored defaults; CPU range/curriculum tests remain | task |
| Mass/inertia DR | `dr.pseudo_inertia` startup | `velocity_flat_dr.randomize_mass_inertia` | MATCHED | `asset_dynamics_runtime_probe_4.json`: runtime mass/inertia tensors are finite and identical across reset (startup-only sample), with coupled writer path covered by CPU tests | task |
| Armature/friction DR | `dr.joint_armature`, BAM friction scale | actuator/asset hooks | MATCHED | `asset_dynamics_runtime_probe_4.json`: runtime armature tensors are finite and change on reset; runtime joint friction is zero; friction scale effect is covered by `friction_sweep_contract.json` | actuator |
| Push / foot friction DR | interval pushes + foot material range | IsaacLab event terms | BLOCKED | push/material terms wired; OVPhysX material effect and seeded runtime proof pending | task |
| Encoder bias | actor-only `biased=True`, +/-0.015 | `policy_joint_pos` | MATCHED | actor/critic separation and reset-state tests | task |
| IMU noise/misalignment | actor noise + random mounting rotation | root-state adapter corruption | MATCHED | `velocity_flat_parity_probe_16_sensor_v2.json`: clean/corrupt runtime deltas finite across 16 envs × 4 resets; gyro <=0.03 and gravity <=0.082 including 6deg mounting rotation | task |
| Gyro/gravity delay | lag 0..1, update period 64 | state history adapter | MATCHED | `sensor_delay_runtime_probe_4.json`: concrete 4-env PhysX task with injected yaw-rate ramp observes finite dynamic raw signal and delayed output matching the previous control step; CPU warm-up/period tests cover lag 0/1 and reset semantics | task |
| Joint velocity delay/noise | fixed lag 1, +/-0.25 | state history adapter | MATCHED | `velocity_flat_parity_probe_16_sensor_v2.json`: clean/corrupt joint-velocity deltas finite and <=0.25 across seeded resets; control-step history remains CPU-tested | task |
| Head/body commands | 4D + 6D non-zero commands, 2..5 s resampling | `UniformPoseCommand` manager terms, 13D block | BLOCKED | `velocity_flat_parity_probe_16_contract.json` proves finite sampled 3/4/6D command shapes and non-zero ranges; resampling interval/evolution proof remains pending | task |
| Turn-in-place bucket | 15%, lin=0, yaw 0.4..1.0 at every velocity resample | `MicroduckVelocityCommand` | BLOCKED | `velocity_flat_parity_probe_16_contract.json` observes seeded zero-linear turn buckets (6.25%..25% per reset); multi-resample statistical proof remains pending | task |
| Velocity rewards | XY+Z Gaussian std sqrt(.1) | `track_linear_velocity` | BLOCKED | kernel is CPU-tested and the production reward manager is runtime-finite; seeded command trace under the unified battery remains | task |
| Angular rewards | yaw+XY Gaussian std sqrt(.5) | `track_angular_velocity` | BLOCKED | kernel is CPU-tested and the production reward manager is runtime-finite; seeded command trace under the unified battery remains | task |
| Pose/head rewards | variable leg pose, head tracking, body tracking, head bias | `pose_tracking`, `head_pose_tracking`, `body_pose_tracking`, `head_pose_bias_penalty` | BLOCKED | variable-posture, body 6D, and head-bias kernels are implemented and CPU-tested; IsaacLab runtime command/data-source proof and curriculum closure remain | task |
| Regularizers | body ang vel, subtree angular momentum, dof limits, action rate | task reward terms | BLOCKED | subtree angular momentum is reconstructed from body COM/mass/inertia/velocity tensors; all terms are CPU-tested and runtime-finite, but the cross-backend numerical fixture remains | task |
| Contact rewards | air-time, terrain-relative clearance, swing, slip, self collision | `velocity_flat_contact` + ContactSensor/raw PhysX view | BLOCKED | foot-site terms, direct mjlab count kernel, concrete 4-env raw views, and forced runtime proof (`contact_runtime_forced_direct_4.json`, baseline 0 -> 16 contacts, finite) are covered; full force/slip cross-backend fixture remains | task |
| Terminations | timeout, 70deg orientation, terrain bounds, NaN | timeout, 70deg, bounds, NaN | BLOCKED | extra root-z cutoff was removed and the 70-degree boundary is CPU-tested; runtime sensor/source parity still needs proof | task |
| Privileged critic | base lin vel + foot height/air/contact/contact force | separate `critic` observation group | BLOCKED | 76D wiring and runtime shape smoke pass; command/IMU source and body-level contact source are not yet strict runtime parity | task |
| PPO implementation | `rsl-rl-lib 5.0.1` | image `rsl-rl-lib 5.4.1` | BACKEND_DELTA | 5.0.1 install attempt incompatible with IsaacLab 3.0; controlled probe/report | training |
| Fixed command battery | mjlab 300-step continuous cases | shared `velocity_flat_battery_spec` and IsaacLab harness | BLOCKED | six fixed cases have run deterministically, but the current checkpoint fails behavior gates (saturated actions/high tilt/poor yaw); rerun is required after semantic closure | eval |
| Solver/contact behavior | MuJoCo implicitfast | PhysX GPU solver | BACKEND_DELTA | deterministic physics battery; no reward/PPO compensation | backend |

## Software-stack probe

The IsaacLab 3.0.0 image was tested with `rsl-rl-lib==5.0.1`; its RSL-RL
configuration/import contract is incompatible with the IsaacLab 3.0 entrypoint.
The supported image therefore keeps `rsl-rl-lib==5.4.1`.  This changes trainer
implementation details but not task semantics; optimizer/rollout fields remain
matched and the version difference is kept visible in every run manifest.

## Gate order

1. Close P0 action/BAM/asset and reset/DR rows with CPU fixtures.
2. Run the common fixed command battery and deterministic parity tests.
3. Run `64` environments for `5` iterations and export/shape-check the policy.
4. Only after those gates pass may a new `4096`-environment, `6000`-iteration
   run start.  VelStand and the prior long run are excluded from this gate.
