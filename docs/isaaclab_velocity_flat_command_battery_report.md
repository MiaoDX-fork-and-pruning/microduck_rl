# IsaacLab Velocity-Flat Command Battery

Status: **runtime baseline complete; walking acceptance pending**

This is a fixed-seed replay of the IsaacLab 3.0.0 task on Isaac Sim 6.0.1.
It validates the runtime and policy replay path, not a reliable gait or
simulator parity.

## Run manifest

- Task: `IsaacLab-Velocity-Flat-MicroDuck`
- IsaacLab: `3.0.0`, source revision `c7fd163736878a4a348a63880ff6001ea8b3143e`
- Isaac Sim: `6.0.1`
- Checkpoint: `logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/2026-09-03_13-02-22/model_4.pt`
- Checkpoint SHA256: `fc915bfbffbc53ddc628c6fb628ca2e876b72336ab53a311dbb9fda5870508df`
- Seed: `2026`; `16` environments; `250` steps per command
- Actuator: explicit `BamActuator` effort path
- Friction bridge: `motor_only_external_effort_unavailable`

## Results

| Command | Mean XY error (m/s) | P95 XY error (m/s) | Mean yaw error (rad/s) | P95 yaw error (rad/s) | Resets | Reset fraction | Max tilt (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 0.0911 | 0.3361 | 0.1256 | 0.3474 | 82 | 0.0205 | 0.8494 |
| forward | 0.1421 | 0.2000 | 0.1321 | 0.3474 | 85 | 0.0213 | 0.8514 |
| lateral | 0.2493 | 0.4270 | 0.1248 | 0.2949 | 83 | 0.0208 | 0.8514 |
| yaw | 0.0938 | 0.3401 | 0.4413 | 0.6663 | 83 | 0.0208 | 0.8495 |

All observations, actions, and measured state tensors were finite. Mean root
height was approximately `0.1095 m`. The high tilt and reset rates are kept
visible because this is a five-iteration smoke checkpoint, not a trained gait.

## Interpretation and remaining gates

The harness covers fixed zero, forward, lateral, and yaw commands, loads the
checkpoint through the official RSL-RL compatibility path, and records a
reproducible checkpoint hash. The lateral case is the weakest tracker.

Task H still needs a same-format current-mjlab baseline comparison and an
explicit decision about whether this behavior justifies another IsaacLab
training run. PhysX external-load friction parity is unresolved; the USD
continues to use `drive_configured=false`, so BAM is explicit effort control
rather than implicit PhysX PD.

Raw output: `.cache/isaaclab-assets/velocity_flat_command_battery.json`.
