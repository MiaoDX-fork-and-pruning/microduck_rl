# IsaacLab Velocity-Flat Long-Run Report

## Run

- Runtime: Isaac Sim `6.0.1`, IsaacLab `3.0.0`
- Task: `IsaacLab-Velocity-Flat-MicroDuck`
- Environment count: `4096`
- PPO iterations: `4000`
- Total environment steps: `393,216,000`
- Training time: about `2393 s` (39.9 minutes)
- Checkpoint: `logs/rsl_rl/microduck_isaaclab_velocity_flat_normalized/2026-09-04_01-54-17/model_3999.pt`
- Checkpoint SHA256: `d6d235a3b528079a4a714efbcad1131cb09ec4c7b02e411d3fb3f51f61fed13d`

## Training Signals

At iteration 3999 the run remained finite. Mean action standard deviation was
about `0.17`, mean XY velocity error was about `0.42 m/s`, mean yaw error was
about `0.55 rad/s`, timeout fraction was about `0.65`, and fallen fraction was
about `0.35`. These signals show substantial survival and posture learning, but
they do not establish locomotion.

## Fixed Command Battery

The checkpoint was evaluated with 16 environments for 250 steps per command
using `.cache/isaaclab-assets/velocity_flat_command_battery_long_3999.json`.

| Command | Mean actual XY velocity (m/s) | Mean yaw velocity (rad/s) | Resets | Max tilt (rad) |
|---|---:|---:|---:|---:|
| zero `(0, 0, 0)` | `(0.004, 0.002)` | `-0.029` | `0` | `0.438` |
| forward `(0.2, 0, 0)` | `(0.008, -0.006)` | `0.106` | `0` | `0.615` |
| lateral `(0, 0.2, 0)` | `(-0.006, 0.008)` | `-0.053` | `0` | `0.349` |
| yaw `(0, 0, 0.5)` | `(0.003, 0.002)` | `0.150` | `5` | `1.299` |

The policy remains a stable or crouched posture policy rather than a walking
policy: forward and lateral commands produce almost no translation, and yaw
tracking is incomplete and unstable.

## Decision

Task H is **not accepted** for reliable walking or simulator parity. The long
run proves that the IsaacLab runtime can sustain a 4096-environment PPO job and
produce a finite checkpoint. It does not justify moving to VelStand/Task I.

The next experiment should address the standing/crouched local optimum and
early balance curriculum, or validate IsaacLab action/effort semantics against
a controlled stepping benchmark. External-load friction parity remains
unavailable because PhysX force getters refresh only after the actuator callback.
