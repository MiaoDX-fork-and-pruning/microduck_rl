# Generalist G0 Merged Policy

Status: ACTIVE

Source plan: `docs/plans/generalist-g0-merged-policy-plan.md`

Control plane: root Codex session

Latest user intent: implement the full plan through `intuitive-flow`.

Current slice: corrected G0 actor observation ABI and reran the required
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

The updated router/config/evaluator focused tests pass 42 tests, and the smoke
completed after the dwell-aware routing and Gaussian distribution fixes. The
full focused G0 regression suite is green (`42 passed`, 2 exporter warnings).

ONNX proof: `scripts/export_generalist_g0.py` exported the bounded DAgger actor
to `/tmp/g0-dagger.onnx` with a 71D input and 14D output. The 32-sample golden
vector report passed with maximum absolute PyTorch/ONNX error `1.79e-7`;
export tests pass for dense and multi-head actors. Full-seed behavior,
transition, inference-budget, and specialist-regression gates are still
missing, so no candidate is accepted.

P3 checkpoint adapter: the canonical evaluator accepts `rsl_rl` checkpoints
and loads actor-only state through the registered task runner. A direct PPO
checkpoint was constructed successfully, but rollout against the standalone
MuJoCo helper exposed an observation-shape mismatch between the runner's
grouped environment observation and the raw 71D harness. This is a concrete
integration blocker for checkpoint-level P3 evaluation; BC/ONNX evaluation
continues to pass, and no acceptance claim is made for PPO checkpoints.

The adapter also reconstructs dense raw actors directly from `actor_state_dict`,
but the standalone helper still exposes the same grouped/raw observation shape
mismatch for these checkpoints. The fail-closed P4 gate therefore blocks any
checkpoint acceptance until this boundary is corrected.

Normalization diagnosis: PPO checkpoint evaluation now applies the saved
`obs_normalizer._mean/_std` before raw 71D inference. This reduced direct
stand/locomotion maxima to 1.32/1.17 rad but still fails the 1.134 rad gate;
sit/stand reaches 1.91 rad. Hybrid remains worse (2.10/2.06/1.52 rad).
Therefore the previous failures were partly a normalization mismatch, but
correct normalization does not make either five-iteration checkpoint viable.

P3 checkpoint battery is now executable after flattening singleton observation
dimensions in the canonical deployment helper. Fixed-seed 120-tick reports for
both direct and hybrid PPO checkpoints are finite, but every behavior and legal
edge fails the 65-degree stability/success gate. Direct maxima are 1.48, 1.50,
and 1.49 rad (stand, locomotion, sit/stand); hybrid maxima are 2.29, 1.64,
and 1.57 rad. Both remain diagnostic failures, not candidates.

Authoritative P4 decision: running the fail-closed gate on the direct report
returns `DIAGNOSTIC_FAIL` for all three behavior success gates and all four
legal transition gates. Parity, latency, fallback, finite, and ABI evidence
pass, but the plan's acceptance criteria are not met.

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

Evaluator action-contract fix: raw PPO checkpoint inference now applies the
same configured `[-1, 1]` action clipping as the rsl_rl environment wrapper.
The previous 100-iteration direct checkpoint battery was invalid at the MuJoCo
boundary (raw outputs reached about 28,000). After clipping, actions are finite
and in range, but the checkpoint still fails the 65-degree stability gate
(maximum tilt about 1.83 rad) for the tested behaviors and legal edges. The
evaluator tests pass; this remains diagnostic evidence rather than acceptance.

Extended direct-PPO diagnostic: a local 100-iteration, 64-environment run
completed with `WANDB_MODE=disabled` and produced `model_99.pt`. Training was
numerically finite, but behavior-specific tracking rewards remained effectively
zero while fall termination stayed frequent. The clipped canonical battery
still failed all included behavior and legal-edge success cases. This rules out
the five-iteration checkpoint being merely too early and points to reward/router
activation and task conditioning as the next implementation investigation.

After distributing reset nodes and activating `sitstand_pose`, a fresh
20-iteration direct-PPO smoke/diagnostic completed successfully with
`WANDB_MODE=disabled`. The canonical 120-tick battery remains diagnostic:
stand, locomotion, sit/stand, and all four legal edges report `success=false`
(maximum tilts approximately 1.50, 1.81, and 3.07 rad by behavior). The change
removed the proven reward starvation condition but is insufficient for P4
acceptance; longer training or further reward/state-coverage diagnosis is still
required.

Fresh post-fix direct PPO run: 100 iterations at 64 environments completed
without numerical errors, and the new SITSTAND terms carried non-zero weighted
mass (terminal `sitstand_posture_pose_legs` about 0.09 and composite/height
terms positive). However, the canonical `model_99.pt` battery still failed all
three behaviors and four legal edges; maximum tilts were about 2.11 rad
(stand), 1.63 rad (locomotion), and 2.08 rad (sit/stand), with actions clipped
to range. The reward is now active, but the merged policy has not learned the
required stable behaviors.

Reward-contract diagnosis found a second omission: the G0 task had only the
generic velocity-template pose term, so SITSTAND lacked the validated commanded
height/pose/composite stack used by its specialist. G0 now imports those four
posture terms and masks them exclusively to SITSTAND. The configuration still
requires a fresh PPO battery to establish whether this restores stability.

Evaluator gate integration: behavior and edge reports now emit explicit
`success`/`passed` fields from finite, action-range, tilt, and reset criteria.
The current canonical report returns `DIAGNOSTIC_FAIL`: sit/stand and all four
legal transition cases fail their success gates, while stand/locomotion remain
finite. This is the authoritative P4 decision for the current DAgger actor.

Next proof: improve the merged-policy behavior and transition gates. The current
implementation is test-clean and operationally measurable, but it is not an
accepted candidate because the canonical P4 behavior/edge success criteria
remain unmet.

ABI diagnosis/fix: PPO's inherited velocity observation terms previously placed
the six behavior-conditioning values after the 61D specialist block, while the
frozen schema and BC/export actors require behavior at offset 48. The G0 cfg now
reorders actor/critic terms to place `g0_behavior` immediately after the 48D
actor proprioception, followed by command terms and phase/posture/side. A new
order regression covers this boundary. Focused config tests pass (8 tests), and
the corrected direct-PPO 64-env/5-iteration smoke completes finite with a 71D
actor and no NaN termination. This is a contract fix, not behavior acceptance;
the full P3/P4 battery must be rerun from a fresh checkpoint.

Post-ABI diagnostic: a fresh corrected direct-PPO run (`seed=42`, 64 envs,
100 iterations) completed numerically finite. Its canonical 120-tick battery
remains below the behavior gates: maximum tilt is 1.190 rad for stand, 2.103
rad for locomotion, and 1.941 rad for sit/stand. The two unsupported direct
edges remain unexercised. The ABI mismatch was real and is fixed, but this run
does not qualify as a G0 candidate; hybrid comparison and final P4 evidence
remain pending.

Current baseline decision: neither direct PPO nor hybrid PPO is eligible to
continue as a next-skill seed. Both remain diagnostic failures, while the
distilled BC/DAgger actor remains the strongest available artifact for schema,
ONNX, latency, and fallback checks. Canonical evidence is retained in
`/tmp/g0-direct-100-posture-eval.json`, `/tmp/g0-direct-eval-norm.json`,
`/tmp/g0-hybrid-eval-norm.json`, `/tmp/g0-parity.json`, `/tmp/g0-budget.json`,
and `/tmp/g0-fallback.json`; these generated files stay out of Git. No video
or hardware evidence exists, so those final-report fields remain unproven.

Verification checkpoint: the complete focused G0 suite now passes `43 tests`
(including explicit rejection of actions outside the `[-1, 1]` contract).
Configuration, graph legality, data/replay, evaluator, exporter/parity, latency,
fallback, and fail-closed gate tests are covered. This verifies the software
contract but does not substitute for the still-failing physical behavior and
transition battery.

Repository verification: full `tests/` under the current worktree reports
`284 passed, 1 skipped, 1 failed`. The sole failure is the pre-existing
`test_x86_64_resolution_stays_on_pypi`; the user-owned `uv.lock` resolves the
registry to the local Tsinghua mirror. No G0 test fails, and `uv.lock` was left
untouched as required.

Corrected hybrid smoke: the available diagnostic BC artifacts were multi-head
and intentionally rejected by the dense hybrid runner. A new bounded dense
71-512-256-128-14 BC artifact was trained from the frozen 4,200-sample traces
(`validation_mse=0.0324`, SHA-256
`489dd2b8fc044756fd3dc1e4e4cf898cc88d63cb8557965643ededae886a43c0`).
Using it through `MICRODUCK_G0_BC_ACTOR`, the corrected hybrid task completed
the required 64-environment, 5-iteration smoke without error. The artifact is
diagnostic and remains outside Git; longer hybrid behavior evidence is still
required.

Comparable hybrid diagnostic: with the dense BC initializer, a 20-iteration
hybrid PPO run at 64 environments completed without numerical errors. Its
canonical battery still failed all three behaviors and all four legal edges
(maximum tilts about 1.64 rad stand, 2.09 rad locomotion, and 1.62 rad
sit/stand). Direct PPO and hybrid PPO therefore agree on the current decision:
neither meets the G0 stability contract, despite valid initialization, active
rewards, finite actions, and successful smoke tests.

Stop condition: stop before P1 collection if a teacher or legal graph edge
cannot be reproduced. Final completion requires a candidate passing every P4
gate or conclusive evidence that the bounded merge is impossible under the
frozen contracts.

No-touch scope: specialist 61D ABI, production runtime defaults, official
scheduler/state machine, roller configuration, hardware rollout, and excluded
skills (`ground_pick`, `ball_kick`, `roulade`).

Parked work: P1 balanced collection, P2 comparable BC/direct/hybrid baselines,
P3 rollout and ONNX batteries, and P4 decision. These artifacts are retained as
diagnostic evidence and must not be treated as final acceptance.
