# MJLab Velocity-Flat Comment Audit

Date: 2026-09-11

This is a configuration and evidence audit of the comments in
`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`. The audit records
which comments describe a live contract, which describe a historical result,
and which need an experiment before changing the task. It does not change the
walking recipe by itself.

## Findings

### 1. Curriculum overlap is the main training risk

Several difficulty changes happen in the same 500-2000 iteration window:

| Iteration | Change |
| ---: | --- |
| 500 | action-rate `-0.1 -> -0.2`; standing envs `2% -> 5%`; trunk/head CoM begin widening; head-pose range widens |
| 600 | head-pose-bias reward turns on |
| 750 | action-rate `-0.4`; standing envs `10%` |
| 1000 | action-rate `-0.6`; standing envs `15%`; trunk CoM reaches `10 mm`; head CoM reaches `10 mm`; head-pose range widens again |
| 1500 | action-rate `-1.0`; standing envs `20%`; trunk CoM reaches `15 mm`; head-pose-bias reaches `3.0`; head-pose range widens |
| 2000 | standing envs reach `25%`; head-pose range reaches its final cap |

The adapted IsaacLab run gives a concrete warning about this schedule. Its
fixed battery passed at checkpoints 500 through 1500, then translation fell to
`0.016/0.017 m/s` at 2000 and `0.022/0.004 m/s` at 5999 (forward/lateral).
The total training reward continued to look healthy. This means reward and
episode length are not sufficient checkpoint criteria for this task.

The same failure shape is present in the MJLab configuration history: its
command comment records a post-iteration-1000 reward and episode-length decline
when command widening outpaced the robot. A late decline is therefore possible
in MJLab too, but it is not a desirable or guaranteed property of training.
The correct operational rule is to run the behavior battery at every saved
checkpoint and retain the best behavior checkpoint.

The next causal experiment should change one schedule only. The most useful
order is:

1. Hold `rel_standing_envs` at `0.02` through the run.
2. If the decline remains, delay the standing ramp while leaving all other
   stages unchanged.
3. Test the action-rate, CoM, and head-pose stages separately.

Do not combine these changes in one run, because the current evidence cannot
identify which stage moves the policy out of the walking basin.

### 2. The CoM comments were stale

The trunk curriculum is `0.003 -> 0.005 -> 0.010 -> 0.015` m, so the final
range is `+/-15 mm`, not `+/-8 mm`. The head curriculum ends at `+/-10 mm`.
The module comments now state the initial and final values explicitly. All
curriculum step expressions use `NUM_STEPS_PER_ENV`, so changing the rollout
length cannot silently leave one stage in old units.

### 3. The final neck-pitch cap is not mechanically reachable on the walk XML

The final command is a delta from HOME. For `neck_pitch`:

```text
HOME                 = 0.3491 rad
XML hard range       = [-1.5708, 1.0472] rad
requested delta      = [-1.10, 1.10] rad
requested absolute   = [-0.7509, 1.4491] rad
```

The positive request is therefore about `0.402 rad` above the XML upper limit.
The other three head joints fit their requested caps. The task keeps the
historical range for checkpoint compatibility, but the old comment claiming a
uniform mechanical margin was inaccurate. A future reachability A/B should
either clamp this command or use an asymmetric `neck_pitch` range; neither is
silently applied here.

### 4. `bearing_roll` is a known selector error

The name appears in `HEAD_BODY_NAMES`, but in `robot_walk.xml` it is the
right-hip-yaw link under `trunk_base`, not a head link. It has been retained to
preserve historical domain-randomization behavior. Removing it is a behavior
change and should be an isolated A/B, measuring walking, turning, and left/right
balance before and after.

### 5. Heading flags differ in representation, not active sampling

The MJLab base factory leaves `heading_command=True` and sets
`rel_heading_envs=0.0`; the effective heading bucket is therefore empty. The
IsaacLab adapter uses `heading_command=False` and omits the heading range. With
the current zero heading fraction these paths have the same active command
distribution, but the difference should remain visible in parity ledgers and
should be revisited if heading commands are enabled.

### 6. Root-height termination is a profile boundary

The strict MJLab/IsaacLab profile has a separate `root_height < 0.055 m`
termination. The historical adapted IsaacLab profile removes that term to
match the earlier checkpoint's orientation-only boundary. This is a task
definition difference, not a backend implementation detail. It must stay
explicit when comparing runs.

## Comments worth keeping

The following comments encode verified engineering constraints and should not be
removed as noise:

- BAM friction randomization must scale `friction_scale`; MuJoCo
  `dof_frictionloss` is zero under BAM.
- BAM damping randomization is ineffective when `dof_damping` is zero; armature
  randomization still affects BAM.
- `dr.*` add/scale operations are non-accumulating in the pinned mjlab version;
  custom randomizers must still restore defaults before applying a sample.
- IMU rotation randomization is zero-centered tolerance training and does not
  replace a real mounting-bias calibration.
- The one-control-step joint-velocity delay, actor-only encoder bias, and
  passive-joint exclusion are part of the 61D/14D deployment contract.
- The rough-terrain contact and geometry-count notes prevent known NaN and OOM
  failures.
- The head-pose history explains why an instantaneous narrow tracking standard
  stopped walking; it is a measured experiment result, not a generic tuning
  preference.

## Reproduction status

The earlier IsaacLab-specific recipe is available as the separate task
`IsaacLab-Velocity-Flat-MicroDuck-Adapted`. The completed reproduction run is
documented in
[`isaaclab_velocity_flat_adapted_profile.md`](isaaclab_velocity_flat_adapted_profile.md).
`model_750.pt` is the current walking candidate; the final checkpoint is not
selected automatically because its translation response degraded.

The strict task remains the MJLab-parity baseline. No new long training should
be started until a single curriculum hypothesis is selected and the first
checkpoint battery is part of the run procedure.
