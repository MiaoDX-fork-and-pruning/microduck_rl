# Velocity-Flat Backend Comparison

Status: **comparison slice complete; Task H acceptance pending**

This compares the current IsaacLab smoke checkpoint with the accepted mjlab
`velocity_flat` policy. The only directly comparable case is a forward command
of `0.20 m/s`; the IsaacLab battery also covers zero, lateral, and yaw, while
the existing mjlab battery uses forward-speed sweeps. The horizons differ
(IsaacLab: 250 control steps / 5 s; MuJoCo: 300 control steps / 6 s), so the
comparison is directional evidence, not a parity claim.

## Artifact identity

| Backend | Task | Policy artifact | Artifact hash | Scene/runtime |
| --- | --- | --- | --- | --- |
| IsaacLab | `IsaacLab-Velocity-Flat-MicroDuck` | `logs/rsl_rl/microduck_isaaclab_velocity_flat_smoke/2026-09-03_13-02-22/model_4.pt` | `fc915bfbffbc53ddc628c6fb628ca2e876b72336ab53a311dbb9fda5870508df` | IsaacLab 3.0.0 / Isaac Sim 6.0.1 |
| mjlab | `Mjlab-Velocity-Flat-MicroDuck` | `artifacts/specialists/velocity_flat/policy.onnx` | `a092ee993b691fab1fdc96206a7b3eab832d88c14248e205b3cd43f3c9e73317` | MuJoCo `scene.xml` |

The mjlab artifact is the accepted policy from source commit `facd4f4`; its
checkpoint hash is `a71c6c26ff369cb3c2a093649467d3a10bb2dd818dd75f21fb29918ae2cbae3c`.

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

## Decision boundary

Task H has a reproducible runtime battery, checkpoint manifest, and current
mjlab comparison. It should not be marked accepted as a walking result because
the IsaacLab checkpoint has high tilt/reset rates and was trained for only five
iterations. Starting a longer run is a new Task H resource/experiment decision;
the remaining model limitation is also explicit: USD `drive_configured=false`
and BAM external-load friction parity is unavailable.

Raw inputs:

- `.cache/isaaclab-assets/velocity_flat_command_battery.json`
- `.cache/mjlab_velocity_flat_command_battery/velocity_flat/report.json`
