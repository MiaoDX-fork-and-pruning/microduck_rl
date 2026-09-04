# IsaacLab Velocity-Flat Parity Ledger

Status: active strict semantic closure.  `MATCHED` means the implementation
and a deterministic contract exist; `BACKEND_DELTA` is reserved for solver or
same-step PhysX limitations.  No PPO long run is an acceptance baseline while
any P0 row is `NOT_STARTED` or `BLOCKED`.

| Surface | mjlab source | IsaacLab source | Status | Tolerance / evidence | Owner |
| --- | --- | --- | --- | --- | --- |
| Actor ABI 61D / action 14D | `microduck_velocity_env_cfg.py`, policy ABI | `tasks/velocity_flat.py`, `policy_abi.py` | MATCHED | ABI fixture and shape smoke | task |
| HOME + action scale | mjlab JointPositionActionCfg | `parity.policy_action_to_target`, `ActionsCfg` | MATCHED | raw action clip/target unit tests | actuator |
| Action clipping | RSL-RL wrapper clip=1.0 | `parity.clip_policy_action`, battery wrapper | MATCHED | exact `[-1,1]` fixture; runtime training wrapper proof still required | actuator |
| BAM target delay | `delay_min_lag=3`, `delay_max_lag=6` | `BamActuator` FIFO | MATCHED | deterministic queue test | actuator |
| BAM voltage DR | `vin_range=(6.5,8.2)` | `BamActuator` per-env supply | MATCHED | seeded range/floor test | actuator |
| BAM voltage sag | `vin_drop_gain_range=(0,0.2)`, `vin_min=6.0` | `effective_supply_voltage` | MATCHED | pure math fixture | actuator |
| BAM friction scale | `randomize_bam_friction`, friction budget | `randomize_bam_friction` reset event + actuator scale hook | BLOCKED | reset-time per-env sampling and absolute replacement are wired and CPU-tested; PhysX dynamics write and runtime effect proof remain missing | actuator |
| External-load friction timing | same-step solved torque in MuJoCo | PhysX force getters refresh post-step | BACKEND_DELTA | `force_timing_probe.json`; motor-only bridge explicit | backend |
| Asset joint order / limits | walk MJCF | converted `microduck_walk.usd` | BLOCKED | names/order are mapped, but USD right-hip-yaw limit `0.436` conflicts with canonical HOME `0.4579`; strict runtime handling remains unresolved | asset |
| Asset damping/friction | MJCF damping 0.053; BAM zeroes dof friction | USD import + explicit BAM metadata | BLOCKED | runtime USD still carries nominal joint friction; same-step BAM external-load bridge unavailable | asset |
| Reset height / HOME | reset z 0.12..0.13, x/y ±0.5, yaw ±3.14, joint offsets (0,0) | `reset_velocity_flat_state`, `EventsCfg` | BLOCKED | reset distribution and exact-default joint writes are implemented and CPU-tested; IsaacLab runtime distribution proof pending | task |
| CoM/head CoM DR | `dr.body_ipos` add, non-accumulating | reset event + `curriculum_event_range` live manager update | BLOCKED | restore-then-apply reset hook and stage schedule are wired and CPU-tested; seeded runtime distribution proof pending | task |
| Mass/inertia DR | `dr.pseudo_inertia` startup | `velocity_flat_dr.randomize_mass_inertia` | BLOCKED | startup hook and coupled scaling exist; asset tensor mutation benchmark pending | task |
| Armature/friction DR | `dr.joint_armature`, BAM friction scale | actuator/asset hooks | BLOCKED | armature and reset-time friction-scale hooks are wired; PhysX dynamics write proof pending | actuator |
| Push / foot friction DR | interval pushes + foot material range | IsaacLab event terms | BLOCKED | push/material terms wired; OVPhysX material effect and seeded runtime proof pending | task |
| Encoder bias | actor-only `biased=True`, +/-0.015 | `policy_joint_pos` | MATCHED | actor/critic separation and reset-state tests | task |
| IMU noise/misalignment | actor noise + random mounting rotation | root-state adapter corruption | BLOCKED | control path and episode-stable state are implemented; runtime distribution and sensor-source proof pending | task |
| Gyro/gravity delay | lag 0..1, update period 64 | state history adapter | BLOCKED | control-step helper and deterministic CPU tests exist; runtime manager/substep proof pending | task |
| Joint velocity delay/noise | fixed lag 1, +/-0.25 | state history adapter | BLOCKED | helper is wired and CPU-tested; runtime actor observation proof pending | task |
| Head/body commands | 4D + 6D non-zero commands, 2..5 s resampling | `UniformPoseCommand` manager terms, 13D block | BLOCKED | terms, ranges, and staged range curriculum are wired and CPU-tested; actual IsaacLab CommandManager initialization/runtime shape proof pending | task |
| Turn-in-place bucket | 15%, lin=0, yaw 0.4..1.0 at every velocity resample | `MicroduckVelocityCommand` | BLOCKED | command-term implementation, standing schedule, and unit/config tests exist; runtime resampling/battery proof pending | task |
| Velocity rewards | XY+Z Gaussian std sqrt(.1) | `track_linear_velocity` | BLOCKED | kernel is covered in isolation, but full reward uses an incomplete task reward set and runtime command manager | task |
| Angular rewards | yaw+XY Gaussian std sqrt(.5) | `track_angular_velocity` | BLOCKED | kernel is covered in isolation, but full reward uses an incomplete task reward set and runtime command manager | task |
| Pose/head rewards | variable leg pose, head tracking, body tracking, head bias | `pose_tracking`, `head_pose_tracking`, `body_pose_tracking`, `head_pose_bias_penalty` | BLOCKED | variable-posture, body 6D, and head-bias kernels are implemented and CPU-tested; IsaacLab runtime command/data-source proof and curriculum closure remain | task |
| Regularizers | body ang vel, angular momentum sensor, dof limits, action rate | task reward terms | BLOCKED | body angular velocity and dof-limit/action-rate terms are aligned in code; IsaacLab 3.0 still lacks the required subtree angular-momentum sensor, so that term cannot be wired without an explicit approximation | task |
| Contact rewards | air-time, clearance, swing, slip, self collision | `velocity_flat_contact` + ContactSensor | BLOCKED | current formulas/data sources are body-level approximations; only geom-level filtering is an explicit backend delta | task |
| Terminations | timeout, 70deg orientation, terrain bounds, NaN | timeout, 70deg, bounds, NaN | BLOCKED | extra root-z cutoff was removed and the 70-degree boundary is CPU-tested; runtime sensor/source parity still needs proof | task |
| Privileged critic | base lin vel + foot height/air/contact/contact force | separate `critic` observation group | BLOCKED | 76D wiring exists, but command terms, IMU source, and body-level contact source are not yet strict runtime parity | task |
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
