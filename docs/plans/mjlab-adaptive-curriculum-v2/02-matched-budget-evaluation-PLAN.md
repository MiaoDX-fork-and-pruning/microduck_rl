---
phase: "MJLab Adaptive Curriculum v2 / 02"
plan: "matched-budget-evaluation"
type: experiment
wave: 3
status: ready_after_runner_gate
depends_on: ["00-capability-signal", "01-runner-control"]
requirements: [ACV2-EXP-01, ACV2-EXP-02, ACV2-EXP-03]
---

# Objective

Determine whether runner-owned adaptive curriculum training produces a policy
that matches or improves the canonical fixed curriculum under a matched budget.

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

1. Pass the Phase 1 fake-runner replay gate and inspect one checkpoint for
   co-located PPO, gate, stage, RNG, provenance, and rollback metadata.
2. Upload one immutable source snapshot and create one manifest per branch/seed.
3. Submit through Executor/CloudML. Launch scripts may monitor and collect, but
   runner owns stage decisions and resume uses an explicit checkpoint.
4. Export each final checkpoint with `scripts/export.py` and run the same frozen
   six-bucket battery with the held-out seed set.
5. Verify hashes, finite 61D/14D traces, and update the active capsule.

# Decision gate

Accept adaptive only if it reaches the canonical final distribution, preserves
zero-command and nominal capability, and matches or exceeds fixed control on
held-out lower-tail capability across seeds without unrecovered rollback or
provenance failures. Otherwise retain fixed curriculum and record negative or
inconclusive evidence. A mismatched budget, single seed, or training-only
metric is insufficient.

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
multi-seed, held-out artifacts.
