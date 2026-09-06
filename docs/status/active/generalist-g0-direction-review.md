# G0 Direction Review

Date: 2026-09-04; resolved 2026-09-06
Status: resolved; gated-adapter experiment tracked separately
Related plan: `docs/plans/generalist-g0-merged-policy-plan.md`

## Executive Summary

We have done enough implementation and training exploration to stop treating
the problem as a missing PPO trick. The specialist policies are validated. The
single merged conditioned actor is not validated, despite substantial work on
BC, DAgger, initialization, multi-head routing, direct PPO, hybrid PPO, reset
routing, curriculum stages, and teacher anchors.

The evidence points to a mismatch between the proposed abstraction and the
behavior/state distribution, but it does not yet distinguish representation,
coverage, or reset semantics. The latest recovery probe cannot make that
distinction because its recovery poses were constructed by the probe rather
than recovered from the specialist acceptance evaluator.

## Evidence

| Layer | Current truth | Interpretation |
|---|---|---|
| Specialist policies | Accepted per-task 61D/14D artifacts and ONNX parity | Stable reference and fallback |
| Schema/teacher adapter | 71D adapter finite; ONNX reconstruction parity passes | Not the primary blocker |
| Offline imitation | MSE can be low; closed-loop stability fails | Pointwise imitation is insufficient |
| Shared actor | Stand/locomotion tradeoffs persist across model variants | Strong behavior-interference signal |
| PPO | Direct and hybrid candidates fail G0 gates | More PPO budget is not currently justified |
| Recovery probe | Action parity passes; reset equivalence is unproven | Phase A remains fail-closed |

Latest diagnostic native snapshot (hand-authored recovery states, therefore not
an acceptance baseline):

| Reset bucket | Cases | Success rate |
|---|---:|---:|
| upright | 4 | 1.00 |
| seated | 4 | 0.75 |
| recovery_face_up | 4 | 0.00 |
| recovery_face_down | 4 | 0.00 |
| recovery_left_side | 4 | 0.00 |
| recovery_right_side | 3 | 0.00 |
| recovery_crouched | 3 | 1.00 |
| recovery_natural_fall | 3 | 0.00 |
| post_recovery_upright | 3 | 0.67 |
| **aggregate** | **32** | **0.375** |

The same report gives a native main-task metric of `18.7642`. Teacher action
parity against the reconstructed 71D path is `max_abs=1.31e-6`,
`mean_abs=2.98e-8` over 32,000 control ticks. These numbers establish finite
execution and adapter parity only; they do not establish specialist recovery
quality until reset qpos/qvel, commands, horizon, and termination semantics are
replayed exactly.

Control-arm status on the frozen Phase-B battery:

| Arm | Status | Interpretation |
|---|---|---|
| native teacher | diagnostic only | Ran on hand-authored recovery states; not yet the frozen specialist baseline |
| random/no-op | not run | Required only after native reset equivalence passes |
| teacher-initialized student | not run | Mapping is implemented/tested, but no comparable battery result |
| trajectory-split BC | not run on valid frozen battery | Earlier canonical BC was finite but failed closed loop |
| cumulative DAgger | not run on valid frozen battery | Earlier two-round DAgger was finite but failed closed loop |

The earlier BC/DAgger results are retained as directional evidence, not as
Phase-B control comparisons, because they used incomplete reset coverage.

## Direction Options

### Option A: One bounded evidence repair

Recover the exact specialist recovery reset states, commands, and termination
classes from the accepted evaluator and rerun native parity once. Only if the
native teacher passes that battery should the five Phase-B control arms run.

This is the recommended next action because it is cheap relative to training,
reversible, and can turn the current ambiguity into a real diagnosis.

### Option B: Keep specialists as the product architecture

Treat the merged policy as an optional research branch. The runtime already has
validated specialist artifacts and fallback behavior. This is the most
conservative product decision if the goal is reliable robot behavior rather
than a single neural checkpoint.

### Option C: New architecture hypothesis

If a merged policy remains strategically required, write a new plan around the
observed interference. The new plan must name the mechanism, define a fresh
acceptance contract, and avoid silently weakening the current G0 gates. It may
not be introduced as another anchor/tax/seed sweep under this plan.

## Historical No-Go Before Decision

- no merged PPO;
- no `VELOCITY` or additional behavior;
- no reward redesign;
- no architecture expansion under the current plan;
- no claim that the specialist recovery policy is broken based on the
  hand-authored recovery poses.

This decision gate is resolved. The bounded gated-adapter experiment is tracked
in `docs/plans/generalist-g0-gated-adapter-plan.md`; it is a new diagnostic
architecture hypothesis, not an extension of the old PPO/capacity sweep.

The gated-adapter implementation is present and exportable. Its 100-epoch
frozen-data diagnostic reaches validation MSE `0.001619` and ONNX parity
`3.43e-7`, but its 120-tick standalone rollout fails stand, locomotion, and
sitstand stability gates. It remains diagnostic; specialist fallback stays the
validated product path.

## Current Recommendation

Choose Option A once. If the exact reset contract is unavailable or the native
teacher still fails after exact replay, stop the current merge plan as
inconclusive and keep the validated specialist fallback. Reopen merged-policy
work only under a new approved hypothesis.

Evidence hashes:

- Phase-A/native diagnostic report: `45eba29e12454f8bad2de724fd183c22b259eef945d859acd05be851ad831c31`
- Earlier closed-loop diagnostic report: `f27f431a5ab826c8b97f176abbc43d0a8c2267347e9c5b8ee455b0dbdeef451c`
- Diagnosis report: `3724e8783b184d6d9800bb75a2b3db79b3fc260d6239cfd827685ce367e9f54e`
- Best diagnostic checkpoint: `283766b9749372a281b3fad39b70cc2af6681b3b0c2ef3c0c700c3a2bd658765`
