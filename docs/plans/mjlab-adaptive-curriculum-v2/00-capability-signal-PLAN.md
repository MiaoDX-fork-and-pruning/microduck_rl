---
phase: "MJLab Adaptive Curriculum v2 / 00"
plan: "capability-signal"
type: implementation
wave: 1
depends_on: []
requirements:
  - ACV2-SIGNAL-01
  - ACV2-SIGNAL-02
  - ACV2-AXIS-01
  - ACV2-AXIS-02
must_haves:
  - "The frozen battery emits a versioned continuous report for all six command buckets."
  - "An upright but non-tracking or wrong-turn trace scores worse than a valid trace."
  - "Each adaptive task mode can mutate only its declared axes."
  - "No autonomous runner transition is enabled until this phase passes."
autonomous: true
files_modified:
  - src/mjlab_microduck/evaluation/capability.py
  - src/mjlab_microduck/evaluation/__init__.py
  - scripts/run_adaptive_capability_battery.py
  - scripts/run_specialist_action_battery.py
  - src/mjlab_microduck/tasks/adaptive_curriculum.py
  - src/mjlab_microduck/tasks/adaptive_runner.py
  - src/mjlab_microduck/tasks/microduck_adaptive_velocity_env_cfg.py
  - scripts/advance_adaptive_curriculum.py
  - scripts/run_adaptive_windows.sh
  - tests/test_capability_metrics.py
  - tests/test_adaptive_axis_isolation.py
---

<objective>
Replace the saturated binary adaptive battery with a reproducible, continuous
capability report and make adaptive axis selection explicit end to end.

This phase supplies the only input contract that the runner-owned controller may
consume. It does not add an automatic evaluator callback or launch long runs.
</objective>

<scope_fence>
Keep `Mjlab-Velocity-Flat-MicroDuck` and its fixed curriculum unchanged. Keep
the 61D actor ABI, 14D actions, BAM M6, PPO parameters, reward semantics, and
the final `scripts/export.py` deployment export unchanged. Do not add a Codex
advisor, reward adaptation, ALP-GMM, PLR, DORAEMON, or long CloudML training.
</scope_fence>

<contract>

### Capability report v2

`CapabilityReport` is a versioned JSON payload with these required fields:

```text
schema_version: 2
metadata: {task_id, source_sha, evaluator_config_sha256, checkpoint,
           checkpoint_sha256, policy_format, seed_set_id, generated_at}
axis_mode: all_static | com | head_com | composed
enabled_axes: ["com_range", ...]
buckets: {
  zero|forward|lateral|yaw|turn-left|turn-right: {
    raw: {tracking_error?, angular_tracking_error?, survival_fraction,
          episode_length_mean, reset_reasons, root_height_mean, tilt_p95,
          zero_drift?, action_magnitude_mean, action_rate_mean,
          joint_limit_proximity?, actuator_saturation?},
    components: {tracking, survival, upright, ...},
    score: float in [0, 1],
    passed: bool,
    valid: bool
  }
}
aggregate: {lower_tail_score, critical_buckets, passed, valid}
```

All component normalization functions must be explicit, deterministic, clamped
to `[0, 1]`, and covered by synthetic tests. Non-finite, missing critical data,
or an empty trace marks the affected bucket invalid with score `0.0`; it never
creates NaN and never silently substitutes a pass. The gate receives the six
normalized bucket scores, validates `aggregate.valid`, and keeps its current
lower-tail semantics. Means are report-only and cannot hide a failed bucket.

`zero` uses drift and upright components. `yaw` and the turn buckets use signed
angular tracking. Forward/lateral use their matching commanded velocity error.
Displacement alone is not a tracking score.

### Axis allowlist

One shared resolver defines:

```text
all_static -> ()
com        -> (com_range,)
head_com   -> (head_com_range,)
composed   -> (com_range, head_com_range)
```

The task factory, runner, advance CLI, report metadata, serialized gate state,
and stage audit artifact all use this resolver. Loading state with a different
mode/allowlist fails closed. Inactive axes are absent, not reset to stage zero.
</contract>

<tasks>

<task type="implementation">
  <name>Define and test the continuous capability-report contract</name>
  <files>src/mjlab_microduck/evaluation/__init__.py, src/mjlab_microduck/evaluation/capability.py, tests/test_capability_metrics.py</files>
  <action>
    Create stdlib-first report dataclasses or typed mappings and pure metric
    functions. Define the exact six bucket names, report v2 validation, report
    canonical JSON hashing, required provenance, continuous component formulas,
    lower-tail aggregation, and a `CapabilityReport.to_gate_metrics()` adapter.
    Put the fixed nominal and held-out anchor case IDs/seeds in the evaluator
    config/metadata, rather than deriving them from an output directory. Keep
    metric code importable without MuJoCo, Torch, or ONNX Runtime.

    Add synthetic traces for: valid tracking; upright but zero velocity under a
    forward command; signed yaw/turn in the wrong direction; excess zero-command
    drift; missing/non-finite observations. Assert expected relative scores,
    `valid` behavior, and deterministic report hashes.
  </action>
  <verify>
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_capability_metrics.py
    uv run ruff check src/mjlab_microduck/evaluation tests/test_capability_metrics.py
  </verify>
  <done>
    The good and deliberately degraded synthetic traces produce distinct,
    finite v2 reports; every critical bucket is present; invalid data fails
    deterministically instead of producing a pass.
  </done>
</task>

<task type="implementation">
  <name>Make the frozen battery produce capability report v2</name>
  <files>scripts/run_specialist_action_battery.py, scripts/run_adaptive_capability_battery.py, tests/test_specialist_action_battery.py, tests/test_capability_metrics.py</files>
  <depends_on>Define and test the continuous capability-report contract</depends_on>
  <action>
    Extend the rollout trace and case report only where the simulator exposes
    the needed values. Record commanded and achieved linear/angular velocities,
    episode termination/reset information, time alive, tilt/height, action
    statistics, and available joint-limit/actuator fields. Feed traces into the
    pure report builder. Preserve the existing standalone CLI but write v2 JSON
    and NPZ evidence; accept an explicit task ID, source SHA/config hash, axis
    mode, and fixed seed-set identifier. Keep ONNX as the current external
    evaluator format, with `policy_format="onnx"`; do not make ONNX export part
    of the upcoming runner state contract.

    Bound ONNX Runtime session threads if the existing runner supports it, only
    to suppress host affinity noise. Do not change policy outputs or actions.
  </action>
  <verify>
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_capability_metrics.py tests/test_specialist_action_battery.py
    uv run ruff check scripts/run_specialist_action_battery.py scripts/run_adaptive_capability_battery.py src/mjlab_microduck/evaluation
    uv run scripts/run_adaptive_capability_battery.py --help
  </verify>
  <done>
    A battery report contains provenance, all six continuous bucket reports,
    anchor metadata, finite lower-tail aggregation, and no longer defines
    capability solely as `1.0 if passed else 0.0`.
  </done>
</task>

<task type="implementation">
  <name>Enforce adaptive axis isolation across task, state, and CLI</name>
  <files>src/mjlab_microduck/tasks/adaptive_curriculum.py, src/mjlab_microduck/tasks/adaptive_runner.py, src/mjlab_microduck/tasks/microduck_adaptive_velocity_env_cfg.py, scripts/advance_adaptive_curriculum.py, scripts/run_adaptive_windows.sh, tests/test_adaptive_curriculum.py, tests/test_mjlab_adaptive_velocity_config.py, tests/test_adaptive_axis_isolation.py</files>
  <action>
    Centralize the axis-mode resolver beside `ADAPTIVE_AXIS_CONFIGS` and use it
    everywhere. Update the advance CLI to require `--axis-mode` or an
    equivalent serialized allowlist, instantiate only those configurations, and
    persist mode/axes in state. Reject unknown modes, duplicate names, and
    resume-state mismatches. Update the shell harness only as a backwards-
    compatible explicit caller; it remains an audit harness, not a controller.

    Test all-static has no gate; CoM-only never contains or mutates head-CoM;
    head-CoM-only has the symmetric behavior; composed transitions one allowed
    axis at a time in stable order; save/load retains the declared allowlist and
    live EventManager ranges. Keep standing/action-rate diagnostic schedules out
    of the CoM/head-CoM adaptive gate.
  </action>
  <verify>
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_adaptive_curriculum.py tests/test_mjlab_adaptive_velocity_config.py tests/test_adaptive_axis_isolation.py
    uv run ruff check src/mjlab_microduck/tasks/adaptive_curriculum.py src/mjlab_microduck/tasks/adaptive_runner.py src/mjlab_microduck/tasks/microduck_adaptive_velocity_env_cfg.py scripts/advance_adaptive_curriculum.py
    uv run list-envs | rg 'Mjlab-Velocity-Flat-Adaptive'
    WANDB_MODE=disabled PYTHONPATH=$PWD/src uv run train Mjlab-Velocity-Flat-Adaptive-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard --agent.experiment-name /tmp/microduck-adaptive-signal-smoke
  </verify>
  <done>
    A CoM-only trace cannot contain head-CoM; a head-CoM-only trace cannot
    contain CoM; canonical registration remains unchanged; the adaptive smoke
    has finite rewards, 61D actor observations, and 14D actions.
  </done>
</task>

</tasks>

<verification>

1. Run all focused metric, battery, gate, and configuration tests together.
2. Run `uv run ruff check` on every changed Python file.
3. Run `git diff --check` and confirm the canonical Velocity config has no diff.
4. Smoke the adaptive task at 64 environments and five iterations.
5. Run a known deployment-quality ONNX battery and inspect v2 JSON manually:
   the six bucket scores must be finite and include component evidence. This is
   evidence validation, not a policy-quality claim.

</verification>

<success_criteria>

- `CapabilityReport` v2 rejects malformed or non-finite critical evidence.
- Synthetic degradation changes the relevant continuous bucket score while
  keeping an upright-only signal insufficient to pass.
- The same input report produces the same gate metrics and hash.
- Axis mode is serialized and enforced end to end.
- Phase 1 remains disabled until this plan's tests and smoke pass.

</success_criteria>
