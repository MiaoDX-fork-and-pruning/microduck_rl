# Generalist G0 Merged Policy

Status: ACTIVE

Source plan: `docs/plans/generalist-g0-merged-policy-plan.md`

Control plane: root Codex session

Latest user intent: implement the full plan through `intuitive-flow`.

Current slice: P0 evidence and contract freeze complete for `velstand_flat`,
`velocity_flat`, and `sitstand_flat` on the all-collisions/no-wheel model.

Last proven evidence: `artifacts/specialist_artifact_manifest.json` contains
checkpoint and ONNX hashes for all three teachers. The no-reset Track A report
at `artifacts/generalist-v0/specialist-switch-track-a-final.json` passes and
contains the two accepted transition pairs through `velstand_flat`.

Completed slice summary: the existing schema v2 adapter freezes the 71D input
and 14D action contract. `docs/plans/generalist-g0-teacher-manifest.json`
freezes provenance and hashes and verifies successfully. The machine-readable
graph in `docs/generalist_g0_transition_graph.json` exposes exactly four legal
directed edges and rejects unproven direct pairs. Existing BC/DAgger candidates
are diagnostics only; none has passed the G0 rollout gate.

Next slice: reproduce each frozen teacher and legal Track A edge with fixed
seeds, then begin balanced P1 collection only if that evidence passes.

Last proof: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q
tests/test_generalist_transition_graph.py tests/test_generalist_schema.py
tests/test_generalist_model.py tests/test_collect_generalist_dagger.py
tests/test_specialist_artifacts.py` -> 23 passed. `uv run python
scripts/freeze_generalist_teachers.py --output
docs/plans/generalist-g0-teacher-manifest.json --verify` passed all hashes.

P0 reproduction evidence: fixed-seed smoke batteries for each frozen teacher
(`velstand_flat`, `velocity_flat`, `sitstand_flat`) passed finite 61D/14D,
tilt, and action-range gates at seed 42. The retained no-reset Track A final
report proves the accepted `VELSTAND -> VELOCITY -> VELSTAND` and
`VELSTAND -> SITSTAND -> VELSTAND` sequence with zero resets.

P0 gate result: PASSED. P1 balanced collection may begin; excluded skills and
unproven direct edges remain out of the dataset.

Next proof: fixed-seed teacher and no-reset Track A reproduction; no P1 data
collection before that gate. (The reproduction gate is now satisfied; the next
proof is P1 dataset shape, balance, replay-state, and determinism validation.)

Stop condition: stop before P1 collection if a teacher or legal graph edge
cannot be reproduced. Final completion requires a candidate passing every P4
gate or conclusive evidence that the bounded merge is impossible under the
frozen contracts.

No-touch scope: specialist 61D ABI, production runtime defaults, official
scheduler/state machine, roller configuration, hardware rollout, and excluded
skills (`ground_pick`, `ball_kick`, `roulade`).

Parked work: P1 balanced collection, P2 comparable BC/direct/hybrid baselines,
P3 rollout and ONNX batteries, and P4 decision.
