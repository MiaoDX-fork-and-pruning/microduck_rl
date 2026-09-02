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
