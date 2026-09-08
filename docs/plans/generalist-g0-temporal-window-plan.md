# Generalist G0 Temporal Window Plan

Status: IN_PROGRESS

## Goal

Test one materially different shared-policy hypothesis: the current 71D
single-frame input is insufficient to disambiguate behavior phase and recent
momentum, so pointwise action regression averages incompatible teacher actions.
Use a fixed observation history window and sequence-aware training while
preserving one shared actor.

## Evidence Boundary

The existing dense, DAgger, gated, FiLM, and fixed one-hot action-adapter
actors all achieved finite execution and often low offline MSE, but failed
`0/7` canonical gates. Those probes rule out more optimizer budget, ordinary
coverage, learned gate mixing, and output clipping under the single-frame
pointwise objective. They do not test temporal context.

## Scope

- Keep the three G0 behaviors and four legal transitions.
- Keep the specialist teachers, canonical evaluator, 14D raw action semantics,
  production 61D specialist ABI, and runtime scheduler unchanged.
- Implement H4 as four consecutive 48D proprioception frames plus the current
  23D condition block: `4 * 48 + 23 = 215D`.
- Use one shared trunk and one shared 14D action head. Condition fields remain
  explicit; no specialist policy branches or hard-coded per-policy heads.
- Train with sequence-preserving windows and the existing teacher labels.
- Sample one frame per 50 Hz policy tick; H4 therefore covers 80 ms.
- Stack only the latest four 48D proprioception frames. Keep the 23D condition
  block from the newest frame only; do not duplicate commands or phase fields.
- At segment start, repeat the first frame for padding. Never borrow frames from
  a preceding segment.

## Non-goals

- No recurrent hidden state, Transformer, full MoE, reward redesign, new
  behaviors, rollers, direct PPO, or hardware rollout in this plan.
- No change to production scheduler or 61D specialist ONNX contract.
- No claim of success from offline MSE, parity, or finite simulation alone.

## Phases

1. **Contract and windowing**: define versioned H4 schema, frame ordering,
   padding/reset behavior, trajectory boundaries, and metadata. Add tests for
   dimensions, no cross-segment windows, reset fill, and deterministic replay.
2. **Canonical data**: collect the same seven evaluator trajectories as H1,
   then generate H4 windows. Validate teacher hashes, finite values, behavior
   balance, and per-joint/per-behavior losses.
3. **H4 BC probe**: train one shared H4 actor with the same evaluator and
   unchanged gates. Stop immediately if standalone gates fail.
4. **One DAgger round**: only if H4 BC passes all three standalone gates;
   collect
   student-visited H4 windows with teacher relabeling and retrain once.
5. **Export and acceptance**: export ONNX with H4 metadata, verify PyTorch/ONNX
   parity and 50 Hz latency, then run all three standalone and four legal-edge
   gates with no hidden resets.

## Acceptance

- Technical: exact `215D -> 14D` contract, finite outputs, deterministic reset
  and history behavior, one shared trunk/head, ONNX parity.
- Research: H4 BC must pass all standalone gates before DAgger is permitted.
- Product: final candidate must pass all `7/7` canonical gates, preserve action
  range, and meet the existing transition thresholds.
- Production boundary: specialists remain the validated fallback; no runtime
  activation is implied.
- H4 is a generalist-candidate ABI only. Production specialists keep their 61D
  ABI and existing scheduler; runtime integration requires a later plan.

## Stop Gates

- H4 BC fails any standalone gate: stop; do not run DAgger or PPO.
- H4 BC passes standalone but fails transitions: run exactly one DAgger round.
- H4 + one DAgger fails `7/7`: stop and reassess temporal/state hypothesis;
  recurrent state or a new objective requires a new plan.
- Any schema, scheduler, threshold, or hardware change requires re-approval.

## Verification

- Focused schema/window/model/export tests with plugin autoload disabled.
- Deterministic canonical data validator and teacher action parity.
- ONNX golden parity and metadata inspection.
- Existing `evaluate_generalist_g0.py` canonical battery, unchanged gates.
- Report standalone, edge, per-behavior, per-joint, and latency metrics.

## Risks and Defaults

- H4 increases input size and dataset memory; use float32 and streaming window
  generation if needed.
- At segment starts, repeat the first frame for padding; never borrow a prior
  segment's frame.
- Keep the current condition block on the newest frame only; do not duplicate
  commands across history unless data shows that is necessary.
- Use the existing optimizer and seed first; changing optimizer is a separate
  hypothesis.

## Planning Decision

This is a new plan after the single-frame plan exhausted its stop conditions.
It preserves the requested single shared policy while changing the state
representation and sequence data contract. It is ready for user review before
implementation.

## Resolved Planning Decisions

- H4 means four proprioception frames at 50 Hz (80 ms), with current condition
  fields only.
- Segment boundaries are hard history boundaries with first-frame repeat
  padding.
- DAgger is gated on all standalone behavior passes and is limited to one round.
- No production ABI or scheduler change is included.

## Implementation Evidence

- H4 schema and deterministic history implementation: `src/mjlab_microduck/generalist_temporal.py`.
- Canonical H4 training entry point: `scripts/train_generalist_canonical.py --temporal-window`.
- H4-aware evaluator, export, benchmark, canonical collection, and one-round
  DAgger guard are wired while preserving the legacy 71D path.
- Focused contract proof: 37 tests passed (schema, history, model, evaluator,
  export, benchmark, and DAgger helpers).
- Remaining gates: collect/hash the seven canonical trajectories, train the H4
  BC probe, run all three standalone gates, and only then consider one DAgger
  round. No candidate is promoted to production.
