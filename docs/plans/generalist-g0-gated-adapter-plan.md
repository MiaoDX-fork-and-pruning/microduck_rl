# Generalist G0 Gated Adapter Experiment

Status: active-fail-closed

Parent evidence: `docs/plans/generalist-g0-merged-policy-plan.md`

Current checkpoint (2026-09-06): the model, trainer, loader, and ONNX export
path are implemented. A 1-epoch CLI smoke on 4,200 frozen samples emits a
reconstructable `gated_adapter` manifest; a 100-epoch diagnostic reached
validation MSE `0.001619`. ONNX parity passes with max absolute error
`3.43e-7`, while the 120-tick standalone rollout fails all three stability
gates. This is an implementation-complete but research-failed checkpoint; no
PPO escalation is authorized by this plan.

## Goal

Test whether explicit condition gating and low-rank hidden residual adapters
reduce the stand/locomotion/sitstand interference observed in the shared dense
and multi-head candidates.

## Scope

- Use the existing schema-v2 `71D -> 14D` contract and frozen three-teacher
  dataset.
- Use one shared trunk, one shared 14D action head, a condition gate, and
  low-rank per-behavior hidden adapters.
- Reuse the existing BC trainer, artifact validator, ONNX exporter, and G0
  evaluator.
- Keep specialist artifacts and the official scheduler unchanged as fallback.

## Non-goals

- No copied specialist policy branches in the candidate.
- No change to the production 61D specialist ABI or runtime scheduler.
- No new reward terms, behaviors, rollers, recurrent state, or PPO sweep until
  the candidate shows standalone BC improvement.

## Acceptance

- Deterministic: model metadata reconstructs the exact state dict; focused
  model/trainer/export tests pass.
- Technical: finite outputs, exact 71D/14D shapes, ONNX parity, and one shared
  action head.
- Research gate: compare against the frozen 1x/2x/4x dense diagnostics using
  the same dataset, seed, and G0 evaluator. Do not claim success from offline
  MSE alone.
- Product gate: standalone behavior and four legal no-reset transitions pass
  the existing G0 thresholds. Until then this experiment is diagnostic.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen --with pytest pytest tests/test_generalist_model.py tests/test_train_generalist_bc_filter.py tests/test_export_generalist_g0.py tests/test_prepare_generalist_hybrid.py`
- `uv run python scripts/train_generalist_bc.py --gated-adapter ...`
- `uv run python scripts/export_generalist_g0.py <run> ...`
- `uv run python scripts/evaluate_generalist_g0.py --run <run> ...`

## Stop conditions

Stop and diagnose if the adapter remains finite but fails the standalone gate;
do not add more PPO or widen behavior scope under this plan. Any materially
different routing, copied specialist branch, recurrent state, or reward change
requires a separate approved plan.
