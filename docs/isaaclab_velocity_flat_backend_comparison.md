# Velocity-Flat Backend Comparison

Status: **strict semantic closure in progress; runtime smoke validated, battery gate pending**

This compares the current IsaacLab smoke checkpoint with the accepted mjlab
`velocity_flat` policy. The only directly comparable case is a forward command
of `0.20 m/s`; the IsaacLab battery also covers zero, lateral, and yaw, while
the existing mjlab battery uses forward-speed sweeps. The horizons differ
(IsaacLab: 250 control steps / 5 s; MuJoCo: 300 control steps / 6 s), so the
comparison is directional evidence, not a parity claim.

The current comparison experiment is `microduck_isaaclab_velocity_flat_mjlab_match`.
Its 64-environment, 5-iteration IsaacLab smoke completed successfully on
Isaac Sim 6.0.1 / IsaacLab 3.0.0. The default runner budget is now 6000
iterations, matching the accepted mjlab training budget; no long run is
considered valid until it uses the artifact and battery identity recorded here.

## Recipe alignment

Strictly aligned for the next cross-backend run:

- physics timestep `0.005 s`, control timestep `0.020 s`, decimation `4`, and
  20-second episodes;
- 61D policy observation and 14D action ABI;
- command ranges `lin_vel_x=(-0.4, 0.4)`, `lin_vel_y=(-0.3, 0.3)`,
  `ang_vel_z=(-1.0, 1.0)`, resampling `(3.0, 8.0)`, and 2% standing commands;
- XY+Z linear tracking, yaw+XY angular tracking, and flat-ground upright
  Gaussian formulas (with mjlab-equivalent standard deviations);
- actor MLP `[512, 256, 128]`, observation normalization, 24 steps per env;
- critic MLP `[512, 256, 128]` and explicit `actor`/`critic` observation-group
  mapping (the previous long run used an accidental `[128, 128, 128]` critic);
- PPO entropy `0.01`, 5 learning epochs, 4 mini-batches, learning rate
  `1e-3`, `gamma=0.99`, `lambda=0.95`, adaptive KL schedule, and 6000
  iterations.

Still excluded from a backend-only claim because runtime parity evidence or
equivalent PhysX data sources are not closed:

- contact sensors and geom-level foot clearance, air-time, slip, and
  self-collision semantics;
- runtime proof for pose/head/body tracking and head-bias curricula;
- mjlab domain randomization, observation noise/delay, curricula, and its
  explicit turn-in-place command bucket;
- privileged critic source closure and the USD/PhysX drive configuration.

## Full parity inventory

The following differences remain after the core recipe alignment. They are
ordered roughly by expected effect on the learned behavior.

| Area | mjlab Velocity-Flat | IsaacLab status | Impact |
| --- | --- | --- | --- |
| Actuator delay | BAM position loop delay `3..6` control steps | per-environment FIFO delay `3..6` | aligned; runtime trace pending |
| BAM voltage | per-environment `vin` plus previous-load voltage drop, floor `6.0 V` | startup `vin` range plus previous-effort sag and floor | aligned; actuator bench evidence |
| BAM friction | friction budget written into solver friction/damping using solved external load | friction budget is only a bench helper; PhysX callback cannot see solved load | high / backend limit |
| BAM gain DR | reset-time `kp`/`kd` and friction-scale hooks | no `kp`/`kd` DR; friction scale is not applied to PhysX | medium-high |
| Rewards | pose, body angular velocity, angular momentum, joint limits, action-rate, air-time, foot clearance/swing/slip, self-collision, head pose and bias | terms wired; contact terms use real body-level sensors, geom-level filtering is a backend delta | high / contact delta |
| Head/body commands | sampled non-zero `head_pose` (4D) and `body_pose` (6D), with curricula | non-zero sampled 4D+6D command block and pose terms wired; curriculum proof pending | medium-high |
| Turn-in-place | explicit 15% command bucket | explicit held 15% turn bucket | aligned; battery proof pending |
| Actor observations | IMU misalignment, noise, gyro/gravity delay, joint encoder bias, joint-velocity delay/noise | stateful adapter wired; runtime seeded distribution proof pending | high |
| Critic observations | privileged base velocity plus foot height/air-time/contact/contact-force sensors | separate 76D critic group with real contact sensor tensors | aligned; body-level contact delta |
| Events / DR | pushes, foot friction, reset action history, CoM/head-CoM, mass/inertia, armature, encoder bias, NaN guard | event terms wired with restore-then-apply adapters; seeded runtime distribution proof pending | high |
| Termination | timeout, 70-degree orientation, terrain bounds, NaN state | timeout, 70-degree orientation, terrain bounds, NaN state | aligned; runtime smoke |
| Asset sensors | MJCF IMU/ang-momentum/foot sites are available to mjlab | USD has the articulated geometry but does not import MJCF sensor objects | medium; root IMU terms are approximations |
| Asset dynamics | MJCF default joint damping `0.053` and BAM solver-side friction path | USD damping is zero and PhysX joint friction is nominal-only; explicit BAM does not close the loop | high |
| Physics solver | MuJoCo `implicitfast`, 10 iterations / LS 20 | PhysX GPU solver, articulation iterations 4/1 | intentional backend difference |
| RL implementation | `rsl-rl-lib 5.0.1` | IsaacLab image bundles `rsl-rl-lib 5.4.1` | medium; PPO code path is not byte-identical |
| Evaluation | mjlab accepted battery is 300 steps, one continuous env, forward sweeps | IsaacLab battery is 250 steps across 16 envs, fixed zero/forward/lateral/yaw | medium; metrics need a common harness |

Additional concrete mismatches found in the runtime/config audit:

- **Reset distribution:** mjlab samples reset base height in `0.12..0.13 m`,
  randomizes joint offsets through `reset_robot_joints`, and starts PPO with
  randomized episode phases. IsaacLab currently resets to one exact HOME pose
  at `z=0.12 m`; only RSL-RL's episode-length randomization is shared.
- **Termination boundary:** mjlab's default `fell_over` limit is 70 degrees,
  plus terrain-bounds and NaN-state terms. IsaacLab's `gravity_xy > 0.75`
  criterion trips at roughly 49 degrees and has a separate `z < 0.055 m`
  cutoff. This changes the learning problem before rewards are considered.
- **Sensor frame:** mjlab actor gyro/gravity use the named IMU sensor and apply
  the same per-environment mounting misalignment, noise, and delay. IsaacLab
  reads root-state tensors directly, so even identically named observations are
  not the same measurement.
- **Friction duplication:** the converted USD report still shows
  `joint_friction=0.0048` on all 14 joints. mjlab's BAM `edit_spec` zeroes
  `dof_frictionloss` and injects its own solver-side budget. Until the PhysX
  friction bridge exists, leaving the USD value active is an additional hidden
  dynamics difference.
- **Action path:** raw policy actions are clipped at the RSL-RL VecEnv boundary
  (`clip_actions=1.0`) before the absolute HOME+scale target transform.
  IsaacLab's action term intentionally has `clip=None`, because its clip field
  runs after scale+offset and would change the mjlab semantics.
- **HOME/limit inconsistency:** the canonical policy HOME uses
  `right_hip_yaw=0.4579 rad`, while both the MJCF and imported USD cap that joint
  at `0.4363 rad`. IsaacLab clamps only the spawn value, so its initial policy
  observation carries a persistent `-0.0219 rad` offset on that joint.
- **Software implementation:** the accepted mjlab run uses `rsl-rl-lib 5.0.1`;
  the IsaacLab 3.0.0 image uses `rsl-rl-lib 5.4.1`. Even with matching scalar
  PPO fields, optimizer/distribution implementation details are not byte-level
  identical.

Asset geometry and canonical joint limits are currently close: the compiled
MJCF and USD both contain 14 actuated revolute joints and 75 collision
geometries. The remaining asset concern is dynamics and sensor authoring, not
the joint-name mapping itself.

The first long run answers whether the matched core recipe learns comparable
velocity behavior. A second run that ports the remaining reward, DR, and
curriculum terms is required before attributing any residual gap to simulator
backend behavior.

## Artifact identity

| Backend | Task | Policy artifact | Artifact hash | Scene/runtime |
| --- | --- | --- | --- | --- |
| IsaacLab (mjlab-match final) | `IsaacLab-Velocity-Flat-MicroDuck` | `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-04_03-03-23/model_5999.pt` | `a60ff567c4849725798870bf383b9443c56b378d8106df9f92d21190dd40ce37` | IsaacLab 3.0.0 / Isaac Sim 6.0.1 |
| mjlab | `Mjlab-Velocity-Flat-MicroDuck` | `artifacts/specialists/velocity_flat/policy.onnx` | `a092ee993b691fab1fdc96206a7b3eab832d88c14248e205b3cd43f3c9e73317` | MuJoCo `scene.xml` |

The mjlab artifact is the accepted policy from source commit `facd4f4`; its
checkpoint hash is `a71c6c26ff369cb3c2a093649467d3a10bb2dd818dd75f21fb29918ae2cbae3c`.

The earlier IsaacLab-only smoke checkpoint remains useful for debugging, but
must not be compared as a matched baseline because it used a different command
profile, reward kernel, network, and PPO budget.

## Matched final battery

The final IsaacLab checkpoint was evaluated with the same fixed command battery
shape used for the runtime gate (250 control steps per case, 16 environments):

| Command | Mean actual velocity | Mean tracking error | Resets | Max tilt | Mean absolute raw action |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | `(-0.010, 0.003) m/s` | `0.087 m/s` | `0` | `0.665 rad` | `75.4` |
| forward `0.20` | `(0.131, 0.002) m/s` | `0.121 m/s` | `0` | `1.526 rad` | `74.1` |
| lateral `0.20` | `(-0.011, 0.129) m/s` | `0.112 m/s` | `0` | `1.877 rad` | `72.2` |
| yaw `0.50` | `(-0.008, 0.006) m/s`, `0.004 rad/s` yaw | `0.088 m/s`, `1.201 rad/s` yaw | `0` | `0.888 rad` | `75.3` |

Raw policy actions are large (p95 absolute action is about 125); the harness
clips actions before applying them, so this is evidence of a saturated policy,
not a hardware-ready command magnitude. The run is finite and survives the
battery, but it does not match the accepted mjlab policy's forward behavior
(`0.079 m/s` equivalent over its continuous slice) plus low tilt (`0.082 rad`),
and it fails the commanded-yaw behavior. The correct conclusion is therefore
“training is reproducible, behavior is not yet equivalent.”

Battery artifact:
`.cache/isaaclab-assets/velocity_flat_command_battery_mjlab_match_5999.json`.

## Common forward slice

| Metric | IsaacLab smoke checkpoint | mjlab accepted policy |
| --- | ---: | ---: |
| Requested command | `0.20 m/s` | `0.20 m/s` |
| Mean actual forward velocity | `0.0916 m/s` | `0.0786 m/s` equivalent from `0.4718 m / 6 s` |
| Lateral velocity | `-0.0234 m/s` | `0.0320 m/s` equivalent from `0.1922 m / 6 s` |
| Mean XY tracking error | `0.1421 m/s` | not reported by this MuJoCo battery |
| Episode/reset evidence | `85` resets / `4000` env-steps (`0.0213`) | no reset; finite 300-step rollout |
| Maximum tilt | `0.8514 rad` | `0.0818 rad` |
| 61D/14D finite | yes | yes |

The forward displacement values are not identical observables: IsaacLab resets
individual environments during the slice, while MuJoCo records one continuous
episode. The useful conclusion is qualitative: the accepted mjlab policy holds
an upright continuous rollout, whereas the five-iteration IsaacLab checkpoint
does not yet show that behavior. This is consistent with the IsaacLab PPO
smoke metrics and is not evidence that either simulator is physically wrong.

## PhysX force timing boundary

The IsaacLab runtime probe confirms that the PhysX articulation view exposes
both `get_dof_projected_joint_forces()` (`(envs, 14)`) and
`get_link_incoming_joint_force()` (`(envs, 15, 6)`). They return finite buffers,
but refresh after `sim.step()`: reads after `scene.write_data_to_sim()` still
describe the previous solved state. The BAM actuator callback runs before that
step, so a same-step external-load torque is not available to
`BamActuator.compute()`. A one-step delayed feedback bridge would be a different
controller and is not accepted as BAM parity. The current motor-only friction
bridge therefore remains the honest boundary.

Evidence: `.cache/isaaclab-assets/force_timing_probe.json`, generated by
`scripts/isaaclab/force_timing_probe.py` in the pinned IsaacLab 3.0.0 / Isaac
Sim 6.0.1 runtime.

## Decision boundary

Task H has a reproducible runtime battery, checkpoint manifest, and an
explicitly matched core recipe. The 6000-iteration run completed normally, but
it should not be marked accepted as a walking or backend-parity result: the
final battery shows saturated actions, high tilt, and failed yaw tracking.
The next engineering step is to close the remaining semantic gaps (especially
action scaling/actuator semantics, contact rewards, DR, noise/delay, and
curricula) before attributing the difference to simulator backend behavior;
the remaining model limitation is also explicit: USD `drive_configured=false`
and BAM external-load friction parity is unavailable.

Raw inputs:

- `.cache/isaaclab-assets/velocity_flat_command_battery.json`
- `.cache/mjlab_velocity_flat_command_battery/velocity_flat/report.json`
