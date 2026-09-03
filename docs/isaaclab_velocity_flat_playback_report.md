# IsaacLab Velocity-Flat Playback Report

## Run

- Backend: IsaacLab `3.0.0` at `c7fd163736878a4a348a63880ff6001ea8b3143e`
- Simulator: Isaac Sim `6.0.1`, Kit `110.1.2`
- Task: `IsaacLab-Velocity-Flat-MicroDuck`
- Checkpoint: `logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/2026-09-03_13-02-22/model_4.pt`
- Seed: `42`
- Visualizer: Kit, headless/no-window

## Observed result

The official IsaacLab 3.0.0 RSL-RL playback entrypoint loaded the checkpoint,
exported JIT and ONNX policies, completed the environment rollout, and exited
with code `0`. The recorder wrote 32 frames to:

`logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/2026-09-03_13-02-22/videos/play/clip_0000.mp4`

The imported articulation warning
`/World/envs/env_0/Robot/Geometry/trunk_base/trunk_base` remains in the Kit log,
but it does not prevent scene reset, 14-joint resolution, policy execution, or
video capture.

## Interpretation

This is a runtime/playback gate, not evidence of a reliable walking gait or
MuJoCo/PhysX parity. The 5-iteration PPO smoke and the earlier 4096-env run
remain finite but have not established competitive tracking or qualitative
walking. The fixed-seed command battery, mjlab comparison, and external-load
friction parity are still pending.
