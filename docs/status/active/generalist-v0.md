# Generalist v0

Status: ACTIVE

Source plan: `docs/plans/generalist-v0-execution-plan.md`

Latest user intent: execute P0 completely; stop before P1.

Current slice: P0 battery implementation and evidence run complete. The
no-wheel walk/all-collisions specialists use the canonical `scene.xml`; roller
specialists use `scene_rollers.xml`. Each policy has independent reset and
per-case evidence. No controller or scheduler behavior was added.

Last proven evidence: the frozen manifest contains 13 accepted 61D -> 14D
specialists with local checkpoint, ONNX, evaluation, and parity artifacts.

Last proof: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q
tests/test_specialist_action_battery.py tests/test_evaluate_specialist_policy.py
tests/test_specialist_scenario.py tests/test_compare_specialist_onnx.py` -> 42
passed. Smoke: 13/13 policies passed finite 61D/14D artifact/stepping checks.
Full battery: `artifacts/generalist-v0/p0-action-battery-semantic-final/summary.json`
contains 13 policies and 36 cases, with direct-step and command-EMA coverage
for both velocity policies.

P0 gate result: NOT PASSED for the full requested velocity grid. Ten policies
pass all primary cases; `standup_flat` now passes its primary sit-to-stand case.
`velocity_flat` fails the 0.03/0.05 m/s buckets in both input modes for
insufficient forward displacement while remaining upright. The separate
`standup_flat/prone_recovery_probe` remains failed, but is evidence for the
future recovery chain and does not count against standalone acceptance.

Stop condition: P0 is complete only if every intended teacher has an explicit,
reproducible passing per-case report. Decide whether the low-speed buckets are
required locomotion behavior or an explicit standing deadband before P1; do not start P1
from this evidence.

No-touch scope: official controller repository/API, runtime defaults, a second
policy scheduler, generalist training, and hardware rollout.

Parked work: P1-P4.
