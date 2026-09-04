# IsaacLab Velocity-Flat Parity Ledger

Status: active strict semantic closure.  `MATCHED` means the implementation
and a deterministic contract exist; `BACKEND_DELTA` is reserved for solver or
same-step PhysX limitations.  No PPO long run is an acceptance baseline while
any P0 row is `NOT_STARTED` or `BLOCKED`.

| Surface | mjlab source | IsaacLab source | Status | Tolerance / evidence | Owner |
| --- | --- | --- | --- | --- | --- |
| Actor ABI 61D / action 14D | `microduck_velocity_env_cfg.py`, policy ABI | `tasks/velocity_flat.py`, `policy_abi.py` | MATCHED | ABI fixture and shape smoke | task |
| HOME + action scale | mjlab JointPositionActionCfg | `parity.policy_action_to_target`, `ActionsCfg` | MATCHED | raw action clip/target unit tests | actuator |
| Action clipping | RSL-RL wrapper clip=1.0 | `parity.clip_policy_action`, battery wrapper | MATCHED | exact `[-1,1]` fixture | actuator |
| BAM target delay | `delay_min_lag=3`, `delay_max_lag=6` | `BamActuator` FIFO | MATCHED | deterministic queue test | actuator |
| BAM voltage DR | `vin_range=(6.5,8.2)` | `BamActuator` per-env supply | MATCHED | seeded range/floor test | actuator |
| BAM voltage sag | `vin_drop_gain_range=(0,0.2)`, `vin_min=6.0` | `effective_supply_voltage` | MATCHED | pure math fixture | actuator |
| BAM friction scale | `randomize_bam_friction`, friction budget | actuator scale hook / asset contract | MATCHED | BAM numerical bench; PhysX write pending | actuator |
| External-load friction timing | same-step solved torque in MuJoCo | PhysX force getters refresh post-step | BACKEND_DELTA | `force_timing_probe.json`; motor-only bridge explicit | backend |
| Asset joint order / limits | walk MJCF | converted `microduck_walk.usd` | MATCHED | asset report + joint mapping tests | asset |
| Asset damping/friction | MJCF damping 0.053; BAM zeroes dof friction | USD import + explicit BAM metadata | BLOCKED | runtime USD still carries nominal joint friction; same-step BAM external-load bridge unavailable | asset |
| Reset height / HOME | reset z 0.12..0.13, reset joint offsets | `EventsCfg`, task initial state | MATCHED | seeded reset contract; right hip limit clamp documented | task |
| CoM/head CoM DR | `dr.body_ipos` add, non-accumulating | `velocity_flat_dr.randomize_com_offsets` | BLOCKED | reset hook and restore-then-apply code exist; seeded runtime distribution proof pending | task |
| Mass/inertia DR | `dr.pseudo_inertia` startup | `velocity_flat_dr.randomize_mass_inertia` | BLOCKED | startup hook and coupled scaling exist; asset tensor mutation benchmark pending | task |
| Armature/friction DR | `dr.joint_armature`, BAM friction scale | actuator/asset hooks | BLOCKED | armature reset hook and friction scale exist; PhysX dynamics write proof pending | actuator |
| Push / foot friction DR | interval pushes + foot material range | IsaacLab event terms | BLOCKED | push/material terms wired; OVPhysX material effect and seeded runtime proof pending | task |
| Encoder bias | actor-only `biased=True`, +/-0.015 | `policy_joint_pos` | MATCHED | actor/critic separation test | task |
| IMU noise/misalignment | actor noise + random mounting rotation | root-state adapter corruption | BLOCKED | adapter and reset hook exist; runtime distribution proof pending | task |
| Gyro/gravity delay | lag 0..1, update period 64 | state history adapter | MATCHED | deterministic history test | task |
| Joint velocity delay/noise | fixed lag 1, +/-0.25 | state history adapter | MATCHED | deterministic history test | task |
| Head/body commands | 4D + 6D non-zero commands | `_pose_commands`, 13D block | MATCHED | command shape/non-zero fixture | task |
| Turn-in-place bucket | 15%, lin=0, yaw 0.4..1.0 | `policy_command_block` | MATCHED | seeded command battery | task |
| Velocity rewards | XY+Z Gaussian std sqrt(.1) | `track_linear_velocity` | MATCHED | pure reward test | task |
| Angular rewards | yaw+XY Gaussian std sqrt(.5) | `track_angular_velocity` | MATCHED | pure reward test | task |
| Pose/head rewards | leg pose + head pose tracking | `pose_tracking`, `head_pose_tracking` | MATCHED | sign/config tests | task |
| Regularizers | body ang vel, momentum, action rate, joint vel | task reward terms | MATCHED | penalty sign tests | task |
| Contact rewards | air-time, clearance, swing, slip, self collision | `velocity_flat_contact` + ContactSensor | BACKEND_DELTA | real body-level sensor: 15 named bodies including `ankle_left`/`ankle_right`, finite forces in `.cache/isaaclab-assets/velocity_flat_smoke_contact_final2.json`; geom-level pair filtering remains unavailable | task |
| Terminations | timeout, 70deg fell-over, bounds, NaN | timeout, 70deg + z, bounds, NaN | MATCHED | manager runtime loaded; 70-degree formula and finite sensor checks covered by focused tests/smoke | task |
| Privileged critic | base lin vel + foot height/air/contact | separate `critic` observation group | MATCHED | runtime group shape 76D with finite contact forces and named ankle bodies; body-level contact limitation recorded above | task |
| PPO implementation | `rsl-rl-lib 5.0.1` | image `rsl-rl-lib 5.4.1` | BACKEND_DELTA | 5.0.1 install attempt incompatible with IsaacLab 3.0; controlled probe/report | training |
| Fixed command battery | mjlab 300-step continuous cases | shared `velocity_flat_battery_spec` and IsaacLab harness | NOT_STARTED | spec is shared (seed 2026, 300 steps, six cases); revised checkpoint run still required | eval |
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
