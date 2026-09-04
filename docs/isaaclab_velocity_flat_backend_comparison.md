# Velocity-Flat Backend Comparison

Status: **strict semantic closure in progress; runtime smoke validated, battery gate pending**

This compares the current IsaacLab smoke checkpoint with the accepted mjlab
`velocity_flat` policy. The only directly comparable case is a forward command
of `0.20 m/s`; the IsaacLab battery also covers zero, lateral, and yaw, while
the existing mjlab battery uses forward-speed sweeps. The horizons differ
(IsaacLab: 250 control steps / 5 s; MuJoCo: 300 control steps / 6 s), so the
comparison is directional evidence, not a parity claim.

The recorded 6000-iteration IsaacLab run predates the strict-parity plan lock
(`5f4b206`).  The earlier plan explicitly allowed a useful backend result
without complete feature parity.  That run is retained as diagnostic history;
it is not evidence that the current strict Task H was executed or accepted.

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
| BAM gain DR | `kp`/`kd` randomization disabled; friction-scale hook enabled | no gain event; friction-scale state is wired but its PhysX dynamics effect is unproven | friction remains high impact |
| Rewards | pose, body angular velocity, angular momentum, joint limits, action-rate, air-time, foot clearance/swing/slip, self-collision, head pose and bias | all 16 terms are wired; foot-site terms are runtime-finite, but the filtered self-contact view currently initializes only for one environment | self-collision is blocked for multi-env training |
| Head/body commands | sampled non-zero `head_pose` (4D) and `body_pose` (6D), with curricula | non-zero sampled 4D+6D command block and pose terms wired; curriculum proof pending | medium-high |
| Turn-in-place | explicit 15% command bucket | explicit held 15% turn bucket | aligned; battery proof pending |
| Actor observations | IMU misalignment, noise, gyro/gravity delay, joint encoder bias, joint-velocity delay/noise | stateful adapter wired; runtime seeded distribution proof pending | high |
| Critic observations | privileged base velocity plus foot height/air-time/contact/contact-force sensors | separate runtime-finite 76D critic group with real contact tensors and MJCF foot-site heights | seeded source proof pending |
| Events / DR | pushes, foot friction, reset action history, CoM/head-CoM, mass/inertia, armature, encoder bias, NaN guard | event terms wired with restore-then-apply adapters; seeded runtime distribution proof pending | high |
| Termination | timeout, 70-degree orientation, terrain bounds, NaN state | timeout, 70-degree orientation, terrain bounds, NaN state | aligned; runtime smoke |
| Asset sensors | MJCF IMU/ang-momentum/foot sites are available to mjlab | canonical foot sites are reconstructed from MJCF constants and subtree momentum from rigid-body tensors; actor IMU remains a root-state adapter | IMU seeded proof pending |
| Asset dynamics | MJCF default joint damping `0.053` and BAM solver-side friction path | USD damping is zero and PhysX joint friction is nominal-only; explicit BAM does not close the loop | high |
| Physics solver | MuJoCo `implicitfast`, 10 iterations / LS 20 | PhysX GPU solver, articulation iterations 4/1 | intentional backend difference |
| RL implementation | `rsl-rl-lib 5.0.1` | IsaacLab image bundles `rsl-rl-lib 5.4.1` | medium; PPO code path is not byte-identical |
| Evaluation | historical accepted battery uses 300-step forward sweeps | shared IsaacLab spec now uses 300 steps, 16 envs, and six cases; the mjlab harness still needs the same spec/metrics | unified battery remains blocked |

Concrete mismatches found by the runtime/config audit, with current disposition:

- **Reset distribution (implemented, proof pending):** IsaacLab now samples
  base height `0.12..0.13 m`, x/y and yaw from the mjlab ranges, restores HOME,
  and resets actor history. A seeded production-runtime distribution report is
  still required before the ledger row can close.
- **Termination boundary (corrected):** the early `gravity_xy > 0.75` / root-z
  rule was removed. The production manager now carries timeout, 70-degree
  orientation, terrain-bounds, and NaN terms; seeded source proof remains.
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
- **Corrected HOME transcription:** an earlier IsaacLab-only ABI table put the
  right-hip-pitch HOME value (`0.4579 rad`) in the right-hip-yaw slot. The mjlab
  `HOME_FRAME` sets both hip-yaw joints to `0.0`; the IsaacLab table, spawn pose,
  action offset, golden target, and direct cross-source regression test now use
  that value. The authored right-hip-yaw limit `[-0.5236, 0.4363] rad` is therefore
  not a HOME/limit conflict. Checkpoints trained before this correction are not
  valid strict-parity baselines.
- **Contact source (partially corrected, still blocked):** foot height and
  velocity now use the canonical MJCF site offsets, and the direct
  self-collision count kernel matches mjlab. The two filtered PhysX count views
  pass in a one-environment probe, but fail to initialize with four environments
  (`expected 8, found 4`). This is not accepted as production evidence until a
  multi-environment-safe source preserves the same three-body, ground-excluding
  semantics.
- **Software implementation:** the accepted mjlab run uses `rsl-rl-lib 5.0.1`;
  the IsaacLab 3.0.0 image uses `rsl-rl-lib 5.4.1`. Even with matching scalar
  PPO fields, optimizer/distribution implementation details are not byte-level
  identical.

Asset geometry and canonical joint limits are currently close: the compiled
MJCF and USD both contain 14 actuated revolute joints and 75 collision
geometries. The remaining asset concern is dynamics and sensor authoring, not
the joint-name mapping itself.

No replacement long run is valid until the ledger, unified battery, and fresh
64-environment/5-iteration smoke all pass. There is no planned "core recipe"
run followed by a later semantic port; the next run must already contain the
strict semantics.

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
The next engineering step is to close the remaining semantic evidence gaps
(especially PhysX friction effects, seeded DR/noise/delay/command traces, and
the unified command battery) before attributing any difference to
simulator backend behavior;
the remaining model limitation is also explicit: USD `drive_configured=false`
and BAM external-load friction parity is unavailable.

Raw inputs:

- `.cache/isaaclab-assets/velocity_flat_command_battery.json`
- `.cache/mjlab_velocity_flat_command_battery/velocity_flat/report.json`
