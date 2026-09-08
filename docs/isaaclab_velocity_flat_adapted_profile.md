# IsaacLab Adapted Velocity-Flat Profile

`IsaacLab-Velocity-Flat-MicroDuck` remains the strict MJLab-parity task. The
separate `IsaacLab-Velocity-Flat-MicroDuck-Adapted` task reproduces the earlier
IsaacLab-specific training recipe that produced the strongest locomotion result
in this repository.

The adapted profile shares the robot asset, 61D actor observation ABI, 14D
action ABI, BAM actuator, domain randomization, privileged critic, and PPO
architecture with strict mode. It restores the historical orientation-only
termination boundary; strict mode additionally has a separate root-height
guard. The adapted profile changes only the values recorded in the historical
run manifest:

| Setting | Strict | Adapted |
| --- | ---: | ---: |
| `track_lin_vel` | 2.0 | 4.0 |
| `track_ang_vel` | 2.0 | 6.0 |
| `pose` | 1.0 | 0.5 |
| `air_time` | 3.0 | 1.0 |
| action-rate curriculum | `-0.1` to `-1.0` | fixed `-0.1` |
| `rel_forward_envs` | 0.2 | 0.0 |
| `rel_lateral_envs` | 0.0 | 0.25 |

The reference checkpoint is `logs/rsl_rl/microduck_isaaclab_velocity_flat_mjlab_match/2026-09-05_15-16-03/model_1000.pt`.
Its fixed six-case battery passed with mean forward velocity `0.139 m/s` and
mean lateral velocity `0.112 m/s`. This is a reproduction target, not a claim
that every new seed will match that checkpoint.

## Reproduction run

The profile was retrained from scratch with 4096 environments for 6000
iterations on IsaacLab 3.0.0 / Isaac Sim 6.0.1:

`logs/rsl_rl/microduck_isaaclab_velocity_flat_adapted/2026-09-08_09-08-31`

The fixed six-case battery used 16 environments, seed `2026`, and 300 control
steps per case. The checkpoints below all passed zero, forward, lateral, yaw,
and both turn cases with zero resets:

| Checkpoint | Forward (m/s) | Lateral (m/s) | Yaw (rad/s) | Battery |
| --- | ---: | ---: | ---: | --- |
| `model_500.pt` | 0.131 | 0.062 | 0.364 | pass |
| `model_750.pt` | 0.156 | 0.097 | 0.452 | pass |
| `model_1000.pt` | 0.170 | 0.076 | 0.423 | pass |
| `model_1250.pt` | 0.151 | 0.043 | 0.437 | pass |
| `model_1500.pt` | 0.126 | 0.049 | 0.415 | pass |

`model_750.pt` is the balanced reproduction candidate because its forward and
lateral response is closest to the historical reference as a pair. The exact
battery artifacts are stored under `.cache/isaaclab-assets/` with the
`velocity_flat_command_battery_adapted_20260908_170815_<iteration>.json`
prefix.

The final checkpoint is not the deployment candidate for this profile. At
`model_2000.pt`, forward/lateral response had fallen to `0.016/0.017 m/s`, and
at `model_5999.pt` it was `0.022/0.004 m/s`; the latter failed only the
translation response cases while posture and turn cases remained stable. The
training reward therefore continues to improve after the useful walking
window, while the fixed behavior battery detects translation skill loss.

The final run itself completed normally (`nan_state=0`, episode length about
`991/1000`, fallen rate `2.4%`, success rate `97.9%`). Checkpoint SHA256 values
for the recommended `model_750.pt` and the final `model_5999.pt` are
`6d90910602873fa536b27810a27f59550de3decf937ebf7a4c2e2ca9de8cf7cc` and
`f065b791e89d5650f5aeddc932439e652a9bf1d02544e873f6f245f560404956`.

The recommended checkpoint was exported through IsaacLab's official play path:

`logs/rsl_rl/microduck_isaaclab_velocity_flat_adapted/2026-09-08_09-08-31/exported/policy.onnx`

The exported graph passes ONNX checker and CPU ONNX Runtime validation with
input shape `[1, 61]` and output shape `[1, 14]`. A one-step video play attempt
also exported the graph, but frame capture could not start because the pinned
container image does not include `omni.replicator`; this does not affect the
checkpoint battery results.

Run a smoke test with:

```bash
ISAACLAB_DOCKER_WRITE=1 scripts/isaaclab/docker-run.sh -lc \
  'PYTHONPATH=/workspace/microduck_rl/src:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_physx:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/IsaacLab/source/isaaclab_newton:/workspace/IsaacLab/source/isaaclab_ov:/workspace/IsaacLab/source/isaaclab_ovphysx:/workspace/IsaacLab/source/isaaclab_visualizers \
   /isaac-sim/python.sh scripts/isaaclab/velocity_flat_smoke.py \
   --task IsaacLab-Velocity-Flat-MicroDuck-Adapted --num-envs 64 --steps 5'
```

Start the full adapted run with the same runner used by strict mode:

```bash
ISAACLAB_DOCKER_WRITE=1 scripts/isaaclab/docker-run.sh -lc \
  'PYTHONPATH=/workspace/microduck_rl/src:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_physx:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/IsaacLab/source/isaaclab_newton:/workspace/IsaacLab/source/isaaclab_ov:/workspace/IsaacLab/source/isaaclab_ovphysx:/workspace/IsaacLab/source/isaaclab_visualizers \
   /isaac-sim/python.sh scripts/isaaclab/rl_launcher.py train \
   --rl_library rsl_rl \
   --task IsaacLab-Velocity-Flat-MicroDuck-Adapted \
   --num_envs 4096 --max_iterations 6000 --headless'
```

Evaluate a checkpoint under the matching profile:

```bash
ISAACLAB_DOCKER_WRITE=1 scripts/isaaclab/docker-run.sh -lc \
  'PYTHONPATH=/workspace/microduck_rl/src:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_physx:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/IsaacLab/source/isaaclab_newton:/workspace/IsaacLab/source/isaaclab_ov:/workspace/IsaacLab/source/isaaclab_ovphysx:/workspace/IsaacLab/source/isaaclab_visualizers \
   /isaac-sim/python.sh scripts/isaaclab/velocity_flat_command_battery.py \
   --task IsaacLab-Velocity-Flat-MicroDuck-Adapted \
   --checkpoint logs/rsl_rl/microduck_isaaclab_velocity_flat_adapted/<run>/model_1000.pt \
   --output .cache/isaaclab-assets/velocity_flat_command_battery_adapted_1000.json'
```
