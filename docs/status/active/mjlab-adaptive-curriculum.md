# MJLab Adaptive Curriculum v2

Status: **ACTIVE** — no usable policy established. Updated: 2026-09-23.
Task control plane: `01a0c2ae-6895-7700-accd-89a0e7a46e1b` (`/root`, `holy-ape`).
Scope: [canonical plan](../../plans/mjlab-adaptive-curriculum-v2-plan.md).
Latest intent: continue authorized changes, training and checks via intuitive-flow.
Execution is local in the main session; other worktree changes are not owned.

## Current evidence and decision

The current diagnostic isolates a curriculum/evaluator mismatch. The final
6,000-update checkpoint evaluated on the same three native gate seeds scores
`.267` on `final` CoM and `.701514` on `initial` CoM (±3 mm). Initial bucket
scores are zero `.953`, forward `.702`, lateral `.773`, yaw `.733`, left `.818`,
right `.792`; the artifact is
`/tmp/microduck-adaptive-sensor-reset-s17-6000-r2/diagnostic-initial-com/capability.json`.
Neither distribution passes `.80`. The campaign currently omits `--distribution`
and therefore uses the evaluator's `final` default even while training remains
at stage 0. This is a confirmed feedback-distribution mismatch, not proof that
changing the gate alone will produce a usable policy.

Current bounded repair contract (medium context: evaluator and resume semantics
must agree): evaluate acquisition against checkpointed CoM stages while retaining
the final reference step and the independent final-distribution held-out/product
gate. Persist distribution and stage provenance; reject mismatches; require an
explicit migration to discard incomparable old gate evidence. Rebaseline mastery,
EMA and rollback evidence when a stage changes. Success is validated stage
consumption, fail-closed report/resume behavior and a real 64-env/5-update smoke,
followed by a bounded training segment. Product success still requires all
original gates below. Canonical Velocity, rewards, ABI and unrelated worktree
changes are outside this repair.

The sensor-reset experiment is complete at
`/tmp/microduck-adaptive-sensor-reset-s17-6000-r2/` (source `07d31a5`,
5,000→6,000 updates, seed 17, 4,096 environments, three-member native gate
cohort). Resampling the IMU/encoder realization on every episode reset is
implemented and isolated from evaluators, but it did not establish a retained
six-capability policy. The four in-run gates were valid and all held at CoM
stage 0; the last three gate lower-tail scores were `.382`, `.000`, `.000`,
and the final gate was `.267` (final buckets: zero `.937`, forward `.683`,
lateral `.789`, yaw `.758`, turn-left `.538`, turn-right `.267`). The adaptive
state ended with `last_known_good_buckets=["zero"]`, so the controller did not
claim a usable policy or advance difficulty.

The final held-out native report improved to lower-tail `.653` (zero `.918`,
forward `.871`, lateral `.790`, yaw `.653`, turn-left `.747`, turn-right
`.798`), but still failed the product gate. CPU/XML transfer for that same
checkpoint remained lower-tail `0.000` (zero `.956`, forward `.000`, lateral
`.000`, yaw `.833`, turn-left `.000`, turn-right `.491`). This separates two
remaining questions: native reset/DR robustness is incomplete, and the CPU
rehearsal mismatch still has not been causally isolated. The sensor-reset
mechanism is therefore evidence against repeating the same augmentation alone,
not evidence that adaptive training is ready for formal multi-seed acceptance.

A bounded deployment-side actuator probe is recorded in
`/tmp/microduck-adaptive-sensor-reset-s17-6000-r2/cpu-actuator-ab.jsonl`.
On the same XML scene and ONNX, adding a 1.75 A current limit produced the
same six traces as the unlimited position-actuator run. Adding a 1–2 step
action delay changed turn tracking (for example, turn-left `.345`→`.105`),
but forward/lateral still scored below the tracking gate and the aggregate
remained `0.0`. This rules out the current-limit switch as a standalone
explanation and makes delay a sensitivity factor, while leaving the full
BAM-vs-XML dynamics/observation split unresolved.

The resume contract is now closed for this augmentation: checkpoint metadata
stores `sensor_reset_fraction`, full loads reject a mismatched live event, the
campaign launcher propagates the saved value before environment construction,
and direct `train` resumes read it from `MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT`.
The adaptive/config suite passes (`222`), focused lint and diff checks pass, and
a real 64-env/5-update smoke followed by a one-update resume with the sensor
variable unset restored `sensor_reset_fraction=1.0`. This fixes reproducibility;
it does not improve the failed policy capability scores above.

Runner evaluation, bounded command exposure, checkpoint/resume and retention
rollback operate. Both CoM axes remain at ±3 mm. Formal training seeds 17/23/47
remain gated on a retained six-capability policy; no video/hardware usability or
adaptive superiority is established.

`e86a1f4` isolates training transition overrides from evaluators. The validated
zero-bootstrap campaign is `/tmp/microduck-adaptive-zero-transition-s17-5000-e86/`.
Its same-policy variance battery has yaw scores
`[.809, .000, .776, .000, .000, .595, .555, .842, .000, .000]` for base seeds
20260815–20260824: five zero scores and only two passes at 0.80. Earlier status
prose claiming six failures was imprecise. These are diagnostic reset/DR seeds,
not independent training seeds. Scalar friction correlation was about 0.13 and
does not isolate a causal DR field.

The matched 5000→5500 reward comparison is in
`/tmp/microduck-adaptive-yaw-planar-s17-5500/clean-control/` and
`clean-treatment-v2/`. Control native held-out scores were
zero .931 / forward .842 / lateral .795 / yaw .442 / left .754 / right .754.
The treatment did not produce a retained usable policy and its held-out yaw
score was zero. Its retained-policy score must not be confused with a direct
measurement of the pre-rollback candidate. `ac21204` reverted the `1f17e81`
yaw-planar reward treatment; canonical reward behavior is restored.

`b942d44` replaces the training transition's bootstrap hold/jump with a linear
bootstrap→target command ramp. It attributes the ramp to the bootstrap bucket until
completion and preserves reset `dt=0` and partial-reset isolation. Evaluators still
use direct frozen commands. The bounded experiment is
`/tmp/microduck-adaptive-slew-s17-5500/experiment-contract.json`.

| Slew experiment (5000→5500, seed 17, 4096 envs) | Minimum score | Outcome |
| --- | ---: | --- |
| Pre-rollback candidate at 5250, native gate | .71733 | forward retention failure |
| Pre-rollback candidate at 5500, native gate | .72894 | forward/left retention failure |
| Retained policy, native held-out | .00000 | original 5000-update actor restored |
| Retained policy, CPU/XML transfer | .00000 | transfer fails |

Actor state including normalizer is tensor-identical to the starting checkpoint.
No retained learning gain is established. The campaign source is read-only
`/tmp/microduck-adaptive-slew-source-b942d44/`; 676 tracked file hashes are recorded.
`campaign-result.json` means evidence completed, not policy accepted.

## Current slice and next decision

Blocker fingerprint: `stage_gate_uses_final_com_distribution`.
Root cause: the evaluator command omitted distribution, so final difficulty
controlled acquisition at stage 0. The stage-aware contract is implemented;
250 adaptive/config/capability tests, focused Ruff (`E4,E7,E9,F`) and diff checks
pass. No new runtime training evidence is available yet.

Next: record a read-only source snapshot; run a real 64-env/5-update smoke with
one three-member stage gate; evaluate the 6,000-update policy at checkpointed
stages with the same final reference step; then run a bounded 6,000→6,500 segment
using `--gate-distribution stage --rebaseline-gate`. The smoke must pass before
long training. A stage pass only controls acquisition: final native held-out,
CPU/XML rehearsal, videos, independent seeds 17/23/47 and matched fixed baseline
remain required. Do not repeat sensor-reset-only or slew-only budgets.

Runtime proof already established for prior slices: 61D actor, 14D action,
normalizer-baked export, sensor-reset restore, valid native cohort reports and
retention rollback. Latest policy results above are failures. The local venv
works; fresh-sync portability remains unproven. Repo `logs/rsl_rl` is not writable
and W&B has no local key, so local runs use TensorBoard under `/tmp`.

## Boundaries and verification inventory

Preserve canonical Velocity, 61D/14D ABI, BAM M6, unfiltered actions, reward signs,
0.5 s signed-EMA metrics, product thresholds and exact-zero/nominal anchors.
Leave IsaacLab, uv.lock, generated files and other processes untouched.
Advisor, stronger teachers, generalist, IsaacLab migration and hardware deployment
remain parked. The rejected 20% final-CoM rehearsal remains opt-in, not a default.

Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest pytest -q
 tests/test_adaptive_*.py tests/test_mjlab_adaptive_velocity_config.py`.
Campaign: `scripts/run_adaptive_campaign_job.py --branch lateral-drive --seed 17
--output <fresh-dir> --iterations <cumulative-updates> --resume <exact-checkpoint>
--num-envs 4096 --gate-interval 250 --gate-cohort-size 3`. Fresh runs use stage
gates; resumes inherit their saved distribution. To migrate a legacy final
gate, add `--gate-distribution stage --rebaseline-gate`. Use a read-only source
snapshot and source manifest. Rehearsal/transition settings inherit the
checkpoint unless explicitly overridden; frozen evaluators disable acquisition
aids.
