# Generalist G0 Merged Policy

Status: PAUSED_PENDING_DIRECTION_REVIEW

Source plan: `docs/plans/generalist-g0-merged-policy-plan.md`
Causal probe: `docs/plans/generalist-g0-velstand-causal-probe.md`
Probe manifest: `docs/plans/generalist-g0-velstand-causal-probe-manifest.json`

## Current conclusion

The merged-policy implementation surface exists and is smoke-tested: the 71D
schema, frozen teachers, reset/transition routing, hybrid anchor, staged taxes,
measured unlocks, corrected 50 Hz evaluator, parity, latency, fallback, and P4
gate are all wired. None of the direct or hybrid PPO candidates passed P4.

Do not launch another merged PPO run from the current recipe. Independent
changes to reset routing, termination horizon, discovery taxes, transition
spawns, measured stage unlocks, and anchor strengths `0.1`/`1.0` all left
VELSTAND below its gate. The accumulated evidence now warrants a direction
review before more implementation or hyperparameter work.

The important distinction is between two claims:

- The specialist policies and their 61D ONNX/Torch action semantics are
  validated on their own frozen acceptance batteries.
- The merged-policy problem has not been shown learnable. Shared BC,
  DAgger, multi-head BC, teacher initialization, direct PPO, and hybrid PPO
  experiments all produced finite but closed-loop failures or behavior
  tradeoffs.

The latest 32-case recovery probe also exposed a contract problem: its
hand-authored recovery poses are useful stress probes, but they are not yet
the manifest-frozen specialist reset states. Therefore its native recovery
success rates must not be used to conclude that `velstand_flat` itself fails
recovery. Action parity passed, while reset/termination equivalence remains
unproven.

## VELSTAND causal probe

The earlier probe produced useful but incomplete diagnostic evidence:

- Frozen teacher reconstruction now matches the exported ONNX normalizer
  (`_std + 0.01`). On the 300-tick canonical trace, action parity passes with
  `max_abs=3.84e-7` and `mean_abs=4.78e-8`.
- VELSTAND-only nominal BC used 300 samples for 100 epochs and reached
  validation MSE `1.38e-5`, but closed-loop canonical-rollout success was 0.
- Two stand-only DAgger rounds added 120 samples each. Success remained 0, so
  the fixed two-no-improvement stop condition fired. No third round ran.
- All observed student actions and states were finite. No student was promoted,
  no PPO arm was started, and no G0 acceptance claim was made.

The provisional diagnosis remains
`reset_semantics_or_state_distribution_coverage`, not a proven model-capacity
or reward conclusion.

## Evidence boundary

The original probe did not fully execute its acceptance contract:

- the original compatibility comparison covered one canonical trace, not the
  complete 32-episode hold/recovery battery;
- the original student battery repeated the same canonical reset, so it did not
  exercise the declared recovery reset buckets;
- the CPU student harness did not expose the specialist main-task reward metric;
- BC used nominal data only, not balanced nominal/recovery traces;
- the original DAgger did not implement the declared 8-tick frontier windows
  or a cumulative multi-round aggregate;
- teacher-initialized, scripted-teacher upper-bound, and random lower-bound
  comparisons were not completed;
- no best-student ONNX export/parity or representative student video review was
  performed because no candidate passed closed loop.

Treat the existing `/tmp` reports and checkpoints as ephemeral diagnostics.
Their paths and hashes are recorded in the probe manifest, but they are not the
immutable external evidence package required for acceptance.

## Direction-review handoff

The next context must choose a direction-review option before any new
execution. The default recommendation is one bounded evidence repair:

1. recover the exact specialist reset qpos/qvel, command, horizon, and
   termination contract;
2. rerun native per-bucket parity once, with the manifest-frozen case list;
3. only if that native baseline passes, run the already-frozen Phase-B control
   arms on the same battery.

If the exact reset contract cannot be recovered, or the native teacher still
fails after exact replay, close this merge attempt as inconclusive and retain
the validated specialist fallback. Any new architecture, reward, PPO, or
additional behavior requires a separate approved plan.

## Current execution update (2026-09-03)

The evaluator and data-contract repairs are now implemented in:

- `scripts/run_velstand_causal_probe.py` (fixed 32-case/9-bucket/20 s/50 Hz
  native battery, current-state FrozenG0 parity, per-bucket metrics);
- `src/mjlab_microduck/generalist_model.py` (exact teacher-to-71D
  initialization by folding the exported normalizer into the first layer);
- `scripts/train_generalist_bc.py` (trajectory split before balancing);
- `scripts/collect_generalist_dagger.py` (cumulative shards and strict 8+1+8
  recovery frontier window helpers).

Focused contract tests pass (17 tests). The required 64-env/5-iteration
VELSTAND smoke test passes offline through iteration 4; online WandB mode is
unavailable on this machine because no API key is configured.

Phase-A diagnostic report: `/tmp/g0-velstand-phase-a.json`, SHA-256
`45eba29e12454f8bad2de724fd183c22b259eef945d859acd05be851ad831c31`.
Teacher action parity passes (`max_abs=1.31e-6`, `mean_abs=2.98e-8`, 32,000
control ticks). The hand-authored recovery rollout reached aggregate success
0.375, but this is not an acceptance baseline because the reset states have not
been proven identical to the specialist report's states. Therefore Phase B is
intentionally not interpreted or promoted: the probe is fail-closed on reset
semantics and no merged PPO, VELOCITY addition, reward change, or architecture
expansion is authorized.

## Direction review

The current question is no longer “which PPO recipe should we try next?” It is
“is a single conditioned actor the right abstraction for this behavior set?”
The strongest evidence against the current direction is repeated
behavior-specific interference: stand can improve while locomotion regresses,
or the reverse, even when offline action MSE is low and all outputs are finite.
Teacher anchors and curriculum changes reduced symptoms but did not establish
closed-loop learnability. More sweeps over anchor weights, taxes, or reset
probabilities are therefore paused.

The next decision should choose one of:

1. **Repair evidence only:** recover the exact specialist recovery reset and
   termination definitions, rerun native parity, then run the frozen Phase-B
   controls once. This is the smallest reversible action.
2. **Reshape the product hypothesis:** keep specialist policies as the runtime
   solution and treat a merged policy as optional research, rather than a
   required replacement. This avoids spending more training budget on an
   unproven shared actor.
3. **Approve a new architecture experiment:** only with an explicit hypothesis
   for the observed interference (for example a gated/multi-policy runtime
   composition), a new plan, and a fresh acceptance contract. This is outside
   the current G0 plan.

Until that decision is made, the current plan is paused, not accepted and not
abandoned. The specialist fallback path remains the only validated product
path.

Any materially different reward, initialization, actor objective, capacity, or
PPO experiment requires a new approved plan.

## Last proof

- Option 1 reset-contract repair advanced 2026-09-06: added
  `scripts/capture_specialist_reset_contract.py` and captured deterministic
  initial reset state, command, horizon, and termination-manager contracts for
  all three accepted G0 specialists. Versioned contracts are stored in
  `docs/plans/generalist-g0-reset-contract-{velstand,velocity,sitstand}.json`.
- The captures show that the accepted specialist evaluator's reset is an
  upright initial distribution; it does not persist the later recovery qpos/qvel
  states. `velstand` uses a 1000-step horizon and `fallen_too_long`, `velocity`
  uses `fell_over`, and `sitstand` uses a 600-step horizon. Therefore the
  hand-authored recovery buckets in the causal probe still cannot be promoted
  to specialist recovery baselines without a trace-capture run from actual
  fallen episodes.
- Reset-contract capture plus schema/model/graph/evaluator/BC tests pass
  (23/23).
- Added `scripts/capture_specialist_recovery_traces.py` and captured real
  `velstand_flat` fallen states from deterministic x/y angular impulses. The
  traces contain qpos/qvel, 61D observations, 14D actions, commands, tilt, and
  height; both falls were finite and crossed the 40-degree fallen gate, but
  neither recovered within 500 control ticks. These are diagnostic rollout
  traces, not specialist acceptance resets or product claims.
- Recovery capture and regression tests pass (15/15).

- The explicit shared-actor capacity ablation completed 2026-09-06. The 4x
  dense actor (`[71, 1024, 512, 256, 14]`) trained on the identical frozen
  traces and seed, remained finite, and failed all standalone behavior and
  legal-transition gates in `/tmp/g0-bc-4x-eval.json`.
- The post-ablation trainer/evaluator tests pass (6/6). Across the 1x, 2x,
  and 4x shared dense arms, no candidate has passed closed-loop G0 acceptance.

- Execution resumed 2026-09-06: focused G0 contract tests passed (32/32).
- Teacher manifest regenerated and hash-verified successfully with
  `scripts/freeze_generalist_teachers.py`.
- The required 64-env/5-iteration G0 smoke test completed in offline WandB
  mode through iteration 4/5 with finite states/actions and non-positive
  weighted penalties. Online WandB remains unavailable without an API key.
- The smoke checkpoint is diagnostic only; no merged candidate has passed the
  canonical evaluator or P4 acceptance gate.

- Probe-focused tests: 17 passed.
- Canonical compatibility report SHA-256:
  `f78d833fd17ca3e0921593cbd0ed1d1c25512e76cdc5f530d6c28f73e189d947`.
- Best diagnostic student checkpoint SHA-256:
  `283766b9749372a281b3fad39b70cc2af6681b3b0c2ef3c0c700c3a2bd658765`.
- Final diagnostic report SHA-256:
  `3724e8783b184d6d9800bb75a2b3db79b3fc260d6239cfd827685ce367e9f54e`.

No-touch scope: specialist 61D ABI, production runtime, roller/hardware
behavior, and the unrelated `uv.lock` worktree change.
