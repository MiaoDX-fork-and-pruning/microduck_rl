# Generalist v0

Status: PAUSED_AT_P2_DIRECTION_REVIEW

Source plan: `docs/plans/generalist-v0-execution-plan.md`

Current slice: P0 and P1 are complete. P2 produced the G0 merged-policy
implementation and a substantial set of BC, DAgger, initialization, actor,
direct PPO, and hybrid PPO diagnostics, but no merged candidate passed its
closed-loop acceptance gates. P2 is paused pending the direction review in
`docs/status/active/generalist-g0-direction-review.md`.

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

P1 gate result: PASSED. The persistent official-controller replay is pinned at
fork commit `66d4fa8facd4c564f8346dc54f361ceaa5e28d59`; its 300-active-tick walk and
roller reports pass at the accepted `0.20 m/s` command. The official controller
continues to own scheduling and state.

P2 gate result: NOT PASSED. Specialist artifacts and action reconstruction are
validated, while the single conditioned actor repeatedly shows stand versus
locomotion interference. The latest 32-case VELSTAND recovery probe passes
action parity but uses hand-authored recovery poses, so reset and termination
equivalence to the accepted specialist evaluator remains unproven.

Next decision: either repair the exact specialist reset contract once, retain
specialists as the product architecture, or approve a new architecture
hypothesis under a new plan. Do not continue merged PPO, add `VELOCITY`, change
rewards, or expand the actor under the current plan.

No-touch scope: specialist 61D ABI, production runtime defaults, official
scheduler, roller/hardware behavior, and specialist fallback artifacts.

Parked work: P2 continuation, P3, and P4.
