---
phase: "MJLab Adaptive Curriculum v2 / 02A"
plan: "diagnostic-repair-and-native-usability"
type: implementation-and-experiment
wave: 4
status: ready
depends_on: ["00-capability-signal", "01-runner-control"]
blocks: ["02-matched-budget-evaluation"]
requirements: [ACV2-DIAG-01, ACV2-DIAG-02, ACV2-DIAG-03, ACV2-DIAG-04]
---

# Objective

Determine whether the existing Adaptive recipe can produce a usable policy,
and identify whether the r4 failure came from native learning, export/
observation parity, or the CPU actuator transfer rehearsal. Keep the canonical
velocity task unchanged.

# Findings that this phase must resolve

- `make_microduck_adaptive_velocity_env_cfg` currently deletes the whole
  canonical curriculum. Adaptive modes must preserve every non-owned canonical
  term and remove only the selected DR-axis schedules.
- The battery accepts a seed argument, but its deterministic reset and no-delay
  path do not consume that seed. Gate and held-out seed sets therefore need an
  explicit consumed reset/DR/perturbation manifest.
- Training uses BAM M6 while the CPU rehearsal uses XML position actuators.
  Native MJLab/BAM evaluation is required for a training-quality conclusion.

# Tasks

## A. Repair adaptive config ownership

1. Update the adaptive factory to filter curriculum terms by an explicit
   allowlist. For `all_static`, `com`, `head_com`, and `composed`, preserve
   standing, action-rate, command, pose, terrain, and other canonical terms.
2. Keep only the selected `com_range` and/or `head_com_range` under runner
   ownership. Add config tests for the exact term sets and enabled axes.
3. Run the existing adaptive smoke before any training command.

## B. Build a native evaluator contract

1. Reuse the training task construction and BAM M6 actuator path to evaluate a
   checkpoint in six canonical buckets: `zero`, `forward`, `lateral`, `yaw`,
   `turn-left`, and `turn-right`.
2. Record the 61D actor observation, 14D action, command, reset/DR seed,
   actuator configuration, raw metrics, and report/config hashes.
3. Compare native policy output with the exported ONNX output on the same
   observation stream. The export must include the normalizer.
4. Keep `scripts/run_adaptive_capability_battery.py` as a separately named CPU
   MuJoCo/ONNX transfer rehearsal. Do not mix its verdict with the native
   product gate.

## C. Prove seed consumption

1. Define a versioned seed manifest that names each consumed reset, DR, or
   perturbation source.
2. Test that same seed plus same checkpoint reproduces every recorded trace
   field, while a different seed changes at least one consumed field.
3. Use disjoint gate and held-out seed sets and include both manifests in the
   report provenance. A different `seed_set_id` without a different trace is a
   validation failure.

## D. Diagnose the current learning curve

For fixed and all-static, evaluate checkpoints at iterations 500, 1000, 2000,
and 4000 with both evaluators. Record:

- native six-bucket raw metrics and lower-tail score;
- CPU rehearsal six-bucket raw metrics and lower-tail score;
- reward terms, episode length, command tracking, and fall rate;
- observation/action parity and checkpoint/export hashes.

Classify the result as one of:

| result | next action |
| --- | --- |
| native fails and CPU fails | diagnose recipe, command sampling, or reward before Adaptive campaign |
| native passes and CPU fails | repair transfer rehearsal/actuator parity |
| native passes and CPU passes | proceed to Phase 2 matched-budget matrix |
| native/export parity fails | fix export or observation contract before training |

## E. Calibrate gate parameters

Use native canonical checkpoint distributions to document the units and choose
EMA, dwell, preservation tolerance, and curriculum thresholds. Preserve the
final product gate unless the calibration report proves a unit or semantic
error. Threshold changes must be versioned and hashed with the evaluator.

# Verification commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest \
  tests/test_mjlab_adaptive_velocity_config.py \
  tests/test_adaptive_checkpoint_battery.py \
  tests/test_adaptive_production_runner.py \
  tests/test_adaptive_checkpoint_resume.py \
  tests/test_adaptive_rollback.py
uv run ruff check src/mjlab_microduck/evaluation \
  src/mjlab_microduck/tasks/adaptive_curriculum.py \
  src/mjlab_microduck/tasks/adaptive_runner.py \
  scripts/run_adaptive_capability_battery.py \
  scripts/run_adaptive_checkpoint_battery.py tests/test_adaptive_*.py
git diff --check
```

# Exit gate

Do not submit the Phase 2 five-branch, three-seed matrix until all of the
following are attached to the active capsule:

- adaptive config ownership tests pass;
- native MJLab/BAM evaluator and ONNX parity are validated;
- same-seed replay is exact and different seed sets change consumed traces;
- the fixed/all-static checkpoint ladder identifies the dominant failure mode;
- threshold/EMA calibration and evaluator config hashes are recorded.

The phase may conclude that the current recipe needs a bounded training change,
but it must not claim Adaptive is impossible from the r4 CPU battery alone.
