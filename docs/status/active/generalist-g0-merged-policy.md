# Generalist G0 Merged Policy

Status: BLOCKED_PENDING_DIAGNOSTIC_PLAN

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
VELSTAND below its gate.

## VELSTAND causal probe

The probe produced useful but incomplete diagnostic evidence:

- Frozen teacher reconstruction now matches the exported ONNX normalizer
  (`_std + 0.01`). On the 300-tick canonical trace, action parity passes with
  `max_abs=3.84e-7` and `mean_abs=4.78e-8`.
- VELSTAND-only nominal BC used 300 samples for 100 epochs and reached
  validation MSE `1.38e-5`, but closed-loop canonical-rollout success was 0.
- Two stand-only DAgger rounds added 120 samples each. Success remained 0, so
  the fixed two-no-improvement stop condition fired. No third round ran.
- All observed student actions and states were finite. No student was promoted,
  no PPO arm was started, and no G0 acceptance claim was made.

The provisional diagnosis is
`state_distribution_coverage_or_model_capacity`, not a proven final root cause.

## Evidence boundary

The probe did not fully execute its acceptance contract:

- the compatibility comparison covered one canonical trace, not the complete
  32-episode hold/recovery battery with termination/outcome parity;
- the reported 32 episodes repeated the same canonical reset, so they did not
  exercise the declared recovery reset buckets;
- the CPU student harness did not expose the specialist main-task reward metric;
- BC used nominal data only, not balanced nominal/recovery traces;
- DAgger did not implement the declared 8-tick frontier windows or a cumulative
  multi-round aggregate;
- teacher-initialized, scripted-teacher upper-bound, and random lower-bound
  comparisons were not completed;
- no best-student ONNX export/parity or representative student video review was
  performed because no candidate passed closed loop.

Treat the existing `/tmp` reports and checkpoints as ephemeral diagnostics.
Their paths and hashes are recorded in the probe manifest, but they are not the
immutable external evidence package required for acceptance.

## Next discussion

The next context should decide and approve a bounded debugging plan before any
training. The recommended first step is to complete evaluator fidelity and
state coverage, then rerun the causal probe:

1. reproduce the frozen teacher's exact reset buckets, episode horizon,
   termination classes, reward terms, and seed semantics for both teacher and
   student;
2. collect balanced nominal/recovery traces and cumulative frontier DAgger
   windows with explicit bucket labels;
3. compare teacher-initialized, BC, DAgger, scripted-teacher, and random arms on
   that same battery;
4. only after a student passes, export ONNX, run parity/video review, and propose
   incremental `VELOCITY` merging with a VELSTAND non-regression gate.

Any materially different reward, initialization, actor objective, capacity, or
PPO experiment requires a new approved plan.

## Last proof

- Probe-focused tests: 15 passed.
- Canonical compatibility report SHA-256:
  `f78d833fd17ca3e0921593cbd0ed1d1c25512e76cdc5f530d6c28f73e189d947`.
- Best diagnostic student checkpoint SHA-256:
  `283766b9749372a281b3fad39b70cc2af6681b3b0c2ef3c0c700c3a2bd658765`.
- Final diagnostic report SHA-256:
  `3724e8783b184d6d9800bb75a2b3db79b3fc260d6239cfd827685ce367e9f54e`.

No-touch scope: specialist 61D ABI, production runtime, roller/hardware
behavior, and the unrelated `uv.lock` worktree change.
