# IsaacLab Adapted Velocity-Flat Profile

`IsaacLab-Velocity-Flat-MicroDuck` remains the strict MJLab-parity task. The
separate `IsaacLab-Velocity-Flat-MicroDuck-Adapted` task reproduces the earlier
IsaacLab-specific training recipe that produced the strongest locomotion result
in this repository.

The adapted profile shares the robot asset, 61D actor observation ABI, 14D
action ABI, BAM actuator, domain randomization, root-height termination,
privileged critic, and PPO architecture with strict mode. It changes only the
values recorded in the historical run manifest:

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
