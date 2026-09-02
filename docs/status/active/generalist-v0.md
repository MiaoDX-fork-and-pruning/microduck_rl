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

P0 gate result: PASSED. All 13 specialists pass their primary gates after
0.03/0.05 m/s were explicitly accepted as a standing deadband. The separate
`standup_flat/prone_recovery_probe` remains failed, but is evidence for the
future recovery chain and does not count against standalone acceptance.

P1 current evidence: official `pollen-robotics/microduck` is pinned at
`590b986bd8c0d50ae02cb3ea2f59c463b6828168`. Its fall predictor tests pass 8/8;
the production limp-pose ramp and default-enabled tests pass; the MuJoCo
sit-to-stand handoff passes. Report:
`artifacts/generalist-v0/p1-official-fall-recovery-prebridge.json`.

P1 gate result: NOT PASSED. The pinned `robotd --fake` backend cannot accept
external IMU/joint frames, so MuJoCo cannot yet drive the official persistent
state machine end to end. Adding a test-only `RobotIo` NDJSON replay adapter in
the official repository is the next bridge slice; do not duplicate the state
machine in this repository.

Stop condition: P0 is complete only if every intended teacher has an explicit,
reproducible passing per-case report. P1 requires a deterministic persistent
official-controller replay before composition parity may be claimed.

No-touch scope: official controller repository/API, runtime defaults, a second
policy scheduler, generalist training, and hardware rollout.

Parked work: P1-P4.
