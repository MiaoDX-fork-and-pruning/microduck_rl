---
phase: "MJLab Adaptive Curriculum v2 / 02"
plan: "matched-budget-evaluation"
type: experiment
wave: 3
status: blocked_pending_phase_2a_diagnostic_repair
depends_on: ["00-capability-signal", "01-runner-control"]
requirements: [ACV2-EXP-01, ACV2-EXP-02, ACV2-EXP-03]
---

# Objective

Determine whether runner-owned adaptive curriculum training produces a policy
that matches or improves the canonical fixed curriculum under a matched budget.

This plan is the follow-up campaign, not the immediate next command. The r4
campaign completed its jobs but is retained as inconclusive because its
held-out seed labels did not change the consumed battery trace, its adaptive
factory removed non-adaptive canonical curricula, and its CPU rehearsal used a
different actuator model from training. Complete Phase 2A in the parent plan
before submitting this matrix again.

## Required Phase 2A exit criteria

The following evidence must be attached to the new campaign manifest:

1. Adaptive config tests show that each mode removes only its owned DR-axis
   schedule and preserves standing, action-rate, command, pose, and other
   canonical curricula.
2. A native MJLab/BAM evaluator produces the six buckets with the same 61D
   observations, 14D actions, normalizer, reset path, and command semantics as
   training. CPU MuJoCo/ONNX remains a separate transfer rehearsal.
3. Two disjoint seed sets are consumed by reset/DR/perturbation code. Running
   the same seed twice is identical; changing the seed changes at least one
   recorded trace field.
4. A fixed/all-static checkpoint ladder identifies whether failure occurs in
   native learning, export/observation parity, or CPU actuator transfer.
5. Threshold and EMA values are calibrated from native canonical reports and
   recorded in the evaluator config hash. Threshold changes require an
   evidence-backed calibration note.

# Matrix

Use identical source, PPO configuration, total environment steps, environment
count, evaluator schema, and at least three training seeds for:

| branch | task |
| --- | --- |
| fixed | `Mjlab-Velocity-Flat-MicroDuck` |
| all static | `Mjlab-Velocity-Flat-Adaptive-Static-MicroDuck` |
| CoM | `Mjlab-Velocity-Flat-Adaptive-CoM-MicroDuck` |
| head CoM | `Mjlab-Velocity-Flat-Adaptive-HeadCoM-MicroDuck` |
| composed | `Mjlab-Velocity-Flat-Adaptive-MicroDuck` |

Default budget: 4000 iterations at 4096 environments. If resources require a
different budget, keep it identical across branches and record the change.
Use one held-out battery seed set that is never used by the gate.

# Procedure

1. Pass the Phase 2A exit criteria and inspect one checkpoint for
   co-located PPO, gate, stage, RNG, provenance, and rollback metadata.
2. Upload one immutable source snapshot and create one manifest per branch/seed.
3. Submit through Executor/CloudML. Launch scripts may monitor and collect, but
   runner owns stage decisions and resume uses an explicit checkpoint.
4. Evaluate checkpoints in native MJLab/BAM first. Export each final
   checkpoint with `scripts/export.py`, then run the CPU MuJoCo/ONNX rehearsal
   as a separate transfer artifact.
5. Verify hashes, finite 61D/14D traces, seed consumption, and both product and
   research verdicts before updating the active capsule.

# Decision gate

Accept adaptive as a **usable policy** when its native six-bucket product gate
passes and its exported CPU rehearsal is within the documented transfer
contract. Accept adaptive as a **research improvement** only if it also reaches
the canonical final distribution, preserves zero-command and nominal
capability, and matches or exceeds fixed control on held-out native lower-tail
capability across seeds without unrecovered rollback or provenance failures.
Otherwise retain fixed as the comparison control and record negative or
inconclusive evidence. A mismatched budget, single seed, label-only seed split,
CPU-only gate, or training-only metric is insufficient.

# Pre-compute verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest \
  tests/test_capability_metrics.py tests/test_adaptive_curriculum.py \
  tests/test_adaptive_axis_isolation.py tests/test_adaptive_runner_windows.py \
  tests/test_adaptive_checkpoint_resume.py tests/test_adaptive_rollback.py \
  tests/test_mjlab_adaptive_velocity_config.py
uv run ruff check src/mjlab_microduck/evaluation \
  src/mjlab_microduck/tasks/adaptive_curriculum.py \
  src/mjlab_microduck/tasks/adaptive_runner.py tests/test_adaptive_*.py
git diff --check
```

Do not claim policy improvement until all five branches have matched-budget,
multi-seed, held-out native artifacts. Do not claim that Adaptive is unusable
until the native-vs-transfer diagnosis has been completed.
