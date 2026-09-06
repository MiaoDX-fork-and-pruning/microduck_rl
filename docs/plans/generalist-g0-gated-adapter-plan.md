# Generalist G0 Gated Adapter Experiment

Status: active-fail-closed; bounded probes exhausted; follow-up evidence recorded

Parent evidence: `docs/plans/generalist-g0-merged-policy-plan.md`

Current checkpoint (2026-09-06): the model, trainer, loader, and ONNX export
path are implemented. The corrected 100-epoch run on 4,200 frozen samples
emits a reconstructable `gated_adapter` manifest and reaches validation MSE
`0.001619`. Canonical 50 Hz evaluation is finite, but all three standalone
behavior gates and all four exercised legal transition gates fail. This is an
implementation-complete but research-failed checkpoint; no further training
or PPO escalation is authorized by this plan.

Bounded diagnostic probes (2026-09-06) are complete. Relabeling 5,600
student-visited states with the frozen teachers and retraining the same actor
did not improve canonical behavior: all three standalone and all four legal
transition gates still failed. Removing the training-side output `tanh` reduced
offline validation MSE to `0.001458`, but canonical evaluation still failed
all seven gates. The current evidence does not support isolated state-coverage
or output-bound fixes, so this plan authorizes no further blind architecture
or optimizer sweep.

An explicit 1,000-epoch convergence probe reached offline validation MSE
`0.000313`, yet the corrected canonical evaluator still failed all three
standalone and all four legal transition gates. This rules out the current
full-batch optimizer budget as the primary explanation for closed-loop loss.

The canonical evaluator also required contract repairs before interpreting the
control arm: velocity now uses the accepted `0.20 m/s` command, tilt uses trunk
world-up alignment rather than total quaternion angle, and raw teacher action
overflow is reportable separately from the shared actor's action-range gate.
The G0 `1.0 m` locomotion product gate remains unchanged. The routed control
reproduced the expected stand and locomotion physics, while the shared gated
adapter still failed its standalone and transition gates.

Follow-up evidence (2026-09-06): compatible generalist initialization and a
teacher-mixed DAgger collector were added. With `beta=0.9`, the student-state
shard's 65-degree tilt breach fraction fell to `0.208` from approximately
`0.96`, but phase-corrected gated-adapter fine-tuning still failed all seven
canonical gates. Standalone phase was aligned with the training timer contract
(zero outside transitions), and the routed teacher was freshly exported with
exact specialist action parity.

A separate FiLM probe retained one shared trunk and one shared 14D action head
while applying condition-dependent feature modulation. Both bounded and
unbounded 1000-epoch BC runs passed ONNX parity and failed all seven canonical
gates. The unbounded run also failed with raw actions above one, so output
range is not the primary explanation.

Follow-up canonical-teacher probe (2026-09-06) collected the exact 50 Hz
standalone and four legal-transition trajectories used by the evaluator:
5,600 samples across seven trajectories, with segment-local transition phase
and the correct frozen teacher selected at each segment. Training gated-adapter
and FiLM actors on this data reached offline validation MSE below `2.4e-5`,
and both exported with ONNX parity, but both still failed all seven canonical
gates. A critical-window oversampling arm and a 2,000-epoch unbounded FiLM arm
also failed. This rules out missing evaluator-horizon data, simple startup
sample weighting, and output clipping as sufficient fixes.

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
do not add more BC epochs, PPO, or widen behavior scope under this plan. Any
materially different routing, copied specialist branch, recurrent state, or
reward change requires a separate approved plan.
