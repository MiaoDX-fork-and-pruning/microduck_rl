# G0 Shared Merged Policy Preflight

Preflight status: DRAFT
Task source: approved direction review and `docs/plans/generalist-g0-merged-policy-plan.md`
Canonical source: `docs/plans/generalist-g0-merged-policy-plan.md`
Route: durable `$intuitive-flow` after approval
Goal: train and verify one shared, behavior-conditioned no-wheel actor for the G0 transition graph without depending on copied specialist networks.

## Scope

- Implement one shared actor with a shared action head using the existing 71D schema and 14D raw action contract.
- Compare approximately 2x and 4x the single-specialist parameter budget.
- Use behavior conditioning through an embedding plus lightweight adapter or FiLM modulation; no complete specialist heads in the product candidate.
- Train only `VELSTAND`, `VELOCITY`, and `SITSTAND` on the frozen legal graph.
- Collect balanced teacher, student-state DAgger, recovery, and legal-transition data.
- Run BC/DAgger, direct PPO, and hybrid PPO as comparable experiments.
- Preserve the specialist scheduler and all specialist artifacts as fallback/control arms.

## Non-goals

- No rollers or cross-profile model.
- No ground-pick, kick, roulade, new reward family, recurrent policy, full MoE, motion-reference model, or arbitrary direct transition.
- No mutation of the production 61D specialist ABI or official runtime scheduler.
- No hardware rollout or production runtime activation in this phase.

## Entity Budget

- Reuse: `generalist_schema.py`, existing teacher manifest, transition graph, BC/DAgger tools, G0 env/evaluator, ONNX exporter, and specialist acceptance artifacts.
- Remove/merge: treat the six-specialist routed ONNX only as a control upper bound; do not extend it as the product candidate.
- New: one shared-conditioned actor implementation/config, capacity metadata, dataset/run manifests, and focused tests only where existing surfaces cannot express the contract.
- Re-approval trigger: changing the public 71D/14D contract, adding behaviors, adding rollers, replacing the scheduler, introducing recurrent state/full MoE, changing acceptance thresholds, or requiring paid/provider infrastructure.

## Context

Must-read: `AGENTS.md`; `docs/plans/generalist-g0-merged-policy-plan.md`; `docs/generalist_policy_v0_design.md`; `docs/generalist_g0_transition_graph.json`; `docs/plans/generalist-g0-teacher-manifest.json`; accepted specialist manifests and G0 diagnostic reports.

Useful: `docs/research/multi-skill-robot-policy-landscape.md`; `artifacts/generalist-v0/specialist-switch-track-a-final.json`; existing BC/DAgger reports.

Avoid-unless-needed: stale historical plans, raw `/tmp` checkpoints, and full specialist videos until a candidate reaches the relevant gate.

## Architecture Contract

- Input: exact schema-v2 71D tensor; output: exact 14D raw action.
- Shared trunk processes proprioception and condition. Behavior identity is encoded explicitly, never inferred from an overloaded zero command.
- Candidate variants:
  - `shared-2x`: approximately twice the parameter count of one specialist.
  - `shared-4x`: approximately four times the parameter count of one specialist.
- Both variants use the same data, seed list, optimizer family, condition layout, and evaluator. Capacity is the primary variable.
- Adapter/FiLM parameters may depend on behavior, but no candidate may contain a complete copied specialist policy branch.
- The routed specialist ONNX is a control upper bound only and is excluded from product acceptance.

## Data Contract

- Freeze teacher checkpoint and ONNX hashes before collection.
- Collect `VELSTAND`, `VELOCITY`, and `SITSTAND` nominal trajectories plus recovery buckets under `VELSTAND`.
- Collect only the four legal edges in the transition graph; route unsupported pairs through `VELSTAND`.
- Balance behavior, command bucket, recovery bucket, and transition phase. Record counts before and after balancing.
- Preserve replay state required for deterministic labels: qpos/qvel, command, phase, posture, previous action, delay/history buffers, episode time, and task latches.
- Validate finite values, exact shapes, teacher hashes, nonzero condition coverage, and deterministic replay before training.

## Execution Order

1. Freeze/re-verify teacher and graph evidence.
2. Run the required 64-env, 5-iteration smoke test for the G0 environment.
3. Collect and validate the balanced dataset and transition windows.
4. Train the 2x and 4x BC/DAgger candidates from identical data and seeds.
5. Run direct PPO from scratch and hybrid PPO from the best distilled candidate; keep teacher anchor schedules explicit and logged.
6. Evaluate standalone behavior batteries, recovery buckets, and both legal bidirectional transition pairs at 50 Hz with zero hidden resets.
7. Export the best shared candidate to ONNX and run PyTorch/ONNX parity, latency, and fallback checks.

## Acceptance

- SUCCESS: one shared actor (not copied specialist routing) passes every G0 behavior gate and both legal transition pairs under the frozen evaluator.
- Per behavior: at least 90% of the corresponding specialist success rate and within 10% of baseline task metrics.
- Recovery: no more than 25% relative fall-rate increase from the validated baseline, with reset semantics explicitly matched or marked diagnostic.
- Transitions: `VELSTAND -> VELOCITY -> VELSTAND` and `VELSTAND -> SITSTAND -> VELSTAND` succeed without reset at least 90% of the time.
- Technical: finite states/actions, action range valid, no positive weighted penalty, exact 71D/14D ABI, ONNX parity, and 50 Hz budget with margin.
- Product boundary: specialists remain installable and selectable as fallback; no unsupported direct edge is silently exercised.
- BLOCKED_NEEDS_DECISION: none currently.
- BLOCKED_NEEDS_LOCAL_VALIDATION: GPU/simulator product runs, long training, and any hardware/runtime claim are required before final acceptance; they are not proven by this document.
- INTERMEDIATE_ONLY: smoke, dataset validation, and failed candidate reports may be committed as diagnostics but are not acceptance.
- No regressions: existing specialist artifacts, 61D ABI, official scheduler, and roller path remain unchanged.

## Verification

- Deterministic: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_generalist_schema.py tests/test_generalist_model.py tests/test_collect_generalist_dagger.py tests/test_generalist_transition_graph.py tests/test_generalist_g0_evaluation.py`
- Integration: `uv run list-envs`; teacher manifest/hash validation; dataset manifest/replay validator; ONNX metadata and golden parity report.
- Product-run: `uv run train Mjlab-GeneralistG0-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`, followed by the frozen 50 Hz G0 evaluator and both no-reset transition batteries.
- Local/live/manual: required GPU training and MuJoCo evaluation; hardware/runtime activation is outside this phase and must remain unclaimed. If GPU or exact reset traces are unavailable, report `BLOCKED_NEEDS_LOCAL_VALIDATION`.
- Optional: routed specialist upper-bound comparison, representative videos, latency profiling beyond the 50 Hz gate.

## Execution Ownership

- Main: root session owns route, experiment ledger, stop gates, and final verification.
- Worker: none required for the initial implementation; later isolated data collection or review may be delegated only with explicit file ownership.
- Worker goal: none.

To execute: `/goal execute docs/plans/generalist-g0-merged-policy-plan.md with intuitive-flow`

Approval: `LGTM` / `approve` / `go ahead` approves this contract; edits request revision.
