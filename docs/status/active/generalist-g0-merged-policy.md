# Generalist G0 Merged Policy

Status: ACTIVE

Source plan: `docs/plans/generalist-g0-merged-policy-plan.md`

Control plane: root Codex session

Latest user intent: implement the full plan through `intuitive-flow`.

Current slice: P2 baseline/evaluation scaffold executes the required
five-iteration smoke;
behavior state, legal interval transitions, and compatible masked reward terms
are wired in the training-only task.

P0 evidence and contract freeze complete for `velstand_flat`,
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

P1 data slice: the BC collector now includes all three behaviors with
deterministic balancing and strict replay validation. Boundary extraction uses
the authoritative Track A report, emits phase/bucket metadata, covers exactly
the four legal edges, and rejects direct velocity/sit transitions.

P2 scaffold: the base, direct-PPO, and hybrid-PPO tasks are registered with
distinct experiment identities and matched budgets. The runner has a Gaussian
actor distribution, reset/interval behavior state, legal-edge sampling, and
masked compatible reward terms. Class-based manager terms remain unwrapped to
preserve mjlab invocation semantics. The canonical evaluator covers all three
behaviors and four legal edges, records unsupported edges, and supports
PyTorch or 71D ONNX inputs.

Next slice: run comparable BC/direct/hybrid training artifacts and execute the
canonical three-behavior plus legal-transition rollout battery, including ONNX
parity and P4 metrics.

P2 BC artifact: `/tmp/g0-bc-multihead-bounded` contains a deterministic
three-behavior balanced dataset (4,200 samples) and bounded actor. Offline
evaluation is finite and in range, but the 120-tick MuJoCo battery fails stand,
locomotion, and sit/stand tilt gates (max tilt 1.88, 1.66, and 1.65 rad).
This remains a failed diagnostic; no G0 candidate is accepted.

Next slice: run student-state DAgger and hybrid-PPO initialization from this
artifact, then repeat the same battery without relaxing acceptance thresholds.

DAgger diagnostic: the collector now supports all three G0 teachers and
produced 120 student-state samples at beta 0.5. Retraining the bounded
multi-head actor produced `/tmp/g0-bc-dagger`; its 120-tick battery still fails
stand, locomotion, and sit/stand (max tilt 1.85, 1.65, and 1.50 rad). The
student-state pass did not cure the shared covariate-shift failure, so this is
not an accepted candidate and hybrid PPO remains the next experiment.

Hybrid bridge proof: `scripts/prepare_generalist_hybrid.py` validates nested BC
manifests and strict actor state dictionaries, then emits matched direct and
hybrid commands. It successfully prepared `/tmp/g0-hybrid-bridge.json` for the
bounded DAgger artifact. The bridge deliberately does not pretend a standalone
BC checkpoint is an `rsl_rl` resume checkpoint; actual actor injection remains
an integration task before hybrid training can be claimed.

Hybrid integration proof: `GeneralistG0HybridRunner` now injects only a dense
71->512->256->128->14 BC actor from `MICRODUCK_G0_BC_ACTOR`; critic and PPO
optimizer state remain fresh. A dense BC artifact was prepared and the hybrid
task completed the required 64-env, five-iteration smoke with finite rewards
and no NaN termination. This is an integration smoke, not a behavior pass.

Matched baseline smoke evidence: `/tmp/g0-p3-comparison.json` records seed 42,
64 environments, five iterations, and SHA256 hashes for direct checkpoint
`fe42af6e84bd3c2cb9f48e508f6e59cb7a85a4937877cf738c74a12a2352bec4` and hybrid
checkpoint `a3e4036649e1ef467fbf2dbf31ff8601afdefde44ab5b40496fe94aebf1d5573`.
Both smokes passed finite-step execution; neither is an acceptance candidate.
The frozen specialist manifest remains available as fallback (`preserved=true`).

ONNX proof: `scripts/export_generalist_g0.py` exported the bounded DAgger actor
to `/tmp/g0-dagger.onnx` with a 71D input and 14D output. The 32-sample golden
vector report passed with maximum absolute PyTorch/ONNX error
`1.79e-7`; metadata records the model kind, architecture, seed, and artifact
hash. Export tests pass for dense and multi-head actors.

Canonical evaluator proof: `/tmp/g0-canonical-eval.json` is a finite PyTorch
report covering all three behaviors, all four legal edges, and both unsupported
direct edges marked unexercised. Its short 40-tick diagnostic still shows
large tilt excursions (stand 0.83, locomotion 0.27, sit/stand 1.46 rad), so it
does not supersede the failed 120-tick acceptance battery.

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

P1/P2 proof: focused collection/schema/model/graph/G0-config tests pass (24
tests across the current focused files). `WANDB_MODE=disabled uv run train
Mjlab-GeneralistG0-Flat-MicroDuck --env.scene.num-envs 64
--agent.max_iterations 5` completed all five iterations with finite rewards,
observations, actions, and no NaN termination.

The updated router/config/evaluator focused tests pass 17 tests, and the smoke
completed after the dwell-aware routing and Gaussian distribution fixes.

ONNX proof: `scripts/export_generalist_g0.py` exported the bounded DAgger actor
to `/tmp/g0-dagger.onnx` with a 71D input and 14D output. The 32-sample golden
vector report passed with maximum absolute PyTorch/ONNX error `1.79e-7`;
export tests pass for dense and multi-head actors. Full-seed behavior,
transition, inference-budget, and specialist-regression gates are still
missing, so no candidate is accepted.

P4 supporting gates: `scripts/benchmark_generalist_g0.py` measured the exported
ONNX actor at p95 `0.020 ms` (15 ms budget after the 5 ms margin), passing the
50 Hz inference-budget gate. `scripts/validate_generalist_fallback.py` passed
against `docs/plans/generalist-g0-teacher-manifest.json`, confirming the three
61D/14D specialists, required artifacts, source commit, and hashes remain
available as fallback. The older 13-policy manifest is intentionally rejected
by this narrower G0 validator.

P4 acceptance validator: `scripts/validate_generalist_g0_gate.py` now fails
closed when behavior or transition success metrics are absent. Running it on
the current finite canonical diagnostic, passing ONNX parity/budget/fallback
reports, returns `DIAGNOSTIC_FAIL` with seven missing-success reasons. This
prevents a numerically valid but behaviorally unmeasured model from being
accepted.

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
