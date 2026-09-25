---
phase: "MJLab Adaptive Curriculum v2 / 01"
plan: "runner-control"
type: implementation
wave: 2
status: validation_pending
depends_on:
  - "MJLab Adaptive Curriculum v2 / 00"
requirements:
  - ACV2-RUNNER-01
  - ACV2-RUNNER-02
  - ACV2-RUNNER-03
  - ACV2-RESUME-01
must_haves:
  - "The runner, not a shell script, owns evaluation cadence and the transition decision."
  - "One checkpoint is the source of truth for PPO, gate, stage, evaluator provenance, and rollback state."
  - "Invalid reports and infrastructure failures leave live curriculum state unchanged."
  - "The disabled adaptive runner preserves normal PPO behavior."
autonomous: true
files_modified:
  - src/mjlab_microduck/evaluation/protocol.py
  - src/mjlab_microduck/tasks/adaptive_runner.py
  - src/mjlab_microduck/tasks/adaptive_curriculum.py
  - src/mjlab_microduck/tasks/microduck_adaptive_velocity_env_cfg.py
  - tests/test_adaptive_runner_windows.py
  - tests/test_adaptive_checkpoint_resume.py
  - tests/test_adaptive_rollback.py
---

<objective>
Make the adaptive runner the authoritative controller for periodic frozen
evaluation, legal stage updates, checkpoint provenance, rollback, and resume.

This phase uses a synchronous typed evaluator protocol. A subprocess adapter is
permitted as an implementation detail after the in-process fake evaluator is
proven; asynchronous evaluation and Codex advice are out of scope.
</objective>

<scope_fence>
Do not alter the canonical task, PPO settings, reward weights, actor/action ABI,
or deployment export semantics. Do not run long matched-budget jobs until this
plan passes. The existing shell pipeline remains runnable for audit but loses
authority over checkpoint selection and stage state.
</scope_fence>

<contract>

### Evaluator protocol

```python
class FrozenCapabilityEvaluator(Protocol):
    def evaluate(
        self,
        *, checkpoint_path: Path, task_id: str, axis_mode: str,
        curriculum_state: Mapping[str, object], iteration: int,
        seed_set_id: str,
    ) -> CapabilityReport: ...
```

The default runner path is synchronous. It provides an explicit pre-evaluation
checkpoint path to the evaluator and accepts only a validated report v2 whose
task, checkpoint hash, mode, allowlist, schema and source/config provenance
match. A timeout, non-zero subprocess exit, missing artifact, invalid report,
or mismatched provenance becomes an `evaluation_error` hold: it cannot advance,
regress, partially mutate live event terms, or overwrite last-known-good.

### Checkpoint payload

`infos["adaptive_curriculum"]` carries a versioned payload containing enabled
axes, current stage values, `CapabilityGate.state_dict()`, report hash/schema,
evaluation iteration and seed set, `last_known_good_checkpoint`, and typed
transition/rollback events. The runner separately captures Python `random`,
NumPy, Torch CPU, and available Torch CUDA RNG states. Existing `alg.save()`
remains responsible for policy and optimizer state.

A checkpoint is saved immediately before evaluation. It becomes
last-known-good only after a valid report passes preservation/frontier checks;
a stage transition is followed by another checkpoint that records the applied
manager state. Rollback loads the complete known-good checkpoint, restores RNG,
and reapplies every enabled axis through `EventManager.get_term_cfg(...)`.

`GateDecision` must distinguish `hold`, `advance`, `regress`,
`preservation_failure`, and `evaluation_error`; `None` alone is insufficient for
runner rollback semantics. Inactive axes are never touched.
</contract>

<tasks>

<task type="implementation">
  <name>Add typed evaluator protocol and deterministic runner window hook</name>
  <files>src/mjlab_microduck/evaluation/protocol.py, src/mjlab_microduck/tasks/adaptive_runner.py, src/mjlab_microduck/tasks/microduck_adaptive_velocity_env_cfg.py, tests/test_adaptive_runner_windows.py</files>
  <action>
    Add the protocol, a report validator/adapter, and an evaluator factory
    injection seam. Extend `AdaptiveMicroduckOnPolicyRunner.learn()` against the
    installed RSL-RL `OnPolicyRunner.learn()` lifecycle: preserve rollout,
    update, logging, and save behavior, then invoke the controller at an
    explicit interval after a PPO update. Use iteration units consistently with
    `NUM_STEPS_PER_ENV=24`; record both iteration and environment-step values.

    Add adaptive-only config fields with safe defaults: evaluation interval `0`
    means disabled, evaluator factory/command, report schema version, seed-set
    ID, and timeout. Verify they survive the task config-to-`asdict` path without
    changing PPO config. Use fake environment/event manager/evaluator tests to
    prove cadence, disabled behavior, valid hold, valid advance after required
    windows, one-axis stage application, malformed report rejection, and no
    evaluation outside an interval.
  </action>
  <verify>
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_adaptive_runner_windows.py tests/test_adaptive_curriculum.py
    uv run ruff check src/mjlab_microduck/evaluation/protocol.py src/mjlab_microduck/tasks/adaptive_runner.py tests/test_adaptive_runner_windows.py
  </verify>
  <done>
    The runner can synchronously evaluate an explicitly saved checkpoint and
    make a deterministic legal decision. With the feature disabled, it does not
    invoke an evaluator or alter normal PPO control flow.
  </done>
</task>

<task type="implementation">
  <name>Implement complete checkpoint, resume, and rollback semantics</name>
  <files>src/mjlab_microduck/tasks/adaptive_curriculum.py, src/mjlab_microduck/tasks/adaptive_runner.py, tests/test_adaptive_checkpoint_resume.py, tests/test_adaptive_rollback.py</files>
  <depends_on>Add typed evaluator protocol and deterministic runner window hook</depends_on>
  <action>
    Extend the gate/controller result API so the runner can record causal
    decisions rather than inferring failure from a missing transition. Add the
    versioned metadata and RNG capture/restore contract above. Ensure save/load
    rejects incompatible enabled axes and evaluator schema/report provenance.
    Implement rollback only from an explicit known-good checkpoint; absent
    known-good state is a safe hold/error, not a partial reset. Reapply all
    enabled live manager values after every load and include a rollback event in
    the trace.

    Test with deterministic fake policy/optimizer state, fake event manager,
    and injected reports: save/resume yields identical gate trace/ranges;
    preservation failure restores policy/optimizer/RNG/gate/ranges; evaluator
    failure leaves state untouched; stage-file edits cannot influence runner
    resume. Maintain compatibility for offline `record_capability_metrics()` if
    existing scripts still use it, but route it through the same validation and
    allowlist logic.
  </action>
  <verify>
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_adaptive_runner_windows.py tests/test_adaptive_checkpoint_resume.py tests/test_adaptive_rollback.py tests/test_adaptive_curriculum.py
    uv run ruff check src/mjlab_microduck/tasks/adaptive_curriculum.py src/mjlab_microduck/tasks/adaptive_runner.py tests/test_adaptive_checkpoint_resume.py tests/test_adaptive_rollback.py
    git diff --check
  </verify>
  <done>
    A restored checkpoint plus the same validated report reproduces the same
    decision, trace, and live ranges. Invalid/evaluation-failure paths leave
    stages unchanged; a valid rollback restores the complete known-good state.
  </done>
</task>

<task type="verification">
  <name>Prove runner integration without changing the canonical task</name>
  <files>tests/test_adaptive_runner_windows.py, tests/test_adaptive_checkpoint_resume.py, tests/test_adaptive_rollback.py</files>
  <depends_on>Implement complete checkpoint, resume, and rollback semantics</depends_on>
  <action>
    Run the focused suite, then run the required adaptive 64-environment,
    five-iteration smoke with evaluation disabled. Add a CPU fake-evaluator
    smoke that enters a window without creating ONNX artifacts. Confirm that
    canonical Velocity config/registration has no diff, then run the final
    deployment command only after a policy is selected in Phase 2; do not claim
    deployment readiness here.
  </action>
  <verify>
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest tests/test_capability_metrics.py tests/test_adaptive_curriculum.py tests/test_adaptive_axis_isolation.py tests/test_adaptive_runner_windows.py tests/test_adaptive_checkpoint_resume.py tests/test_adaptive_rollback.py tests/test_mjlab_adaptive_velocity_config.py
    WANDB_MODE=disabled PYTHONPATH=$PWD/src uv run train Mjlab-Velocity-Flat-Adaptive-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard --agent.experiment-name /tmp/microduck-adaptive-runner-smoke
    git diff --exit-code -- src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py
  </verify>
  <done>
    The smoke has finite rewards, 61D actor observations, and 14D actions. The
    runner-owned state contract passes all failure-injection and replay tests,
    and the canonical task remains unchanged.
  </done>
</task>

</tasks>

<verification>

1. Execute Phase 0's full verification before this plan.
2. Execute the focused runner/checkpoint/rollback suite.
3. Run `ruff`, `git diff --check`, and the mandatory smoke64/5.
4. Inspect a generated checkpoint to confirm all state fields are co-located.
5. Do not submit matched-budget CloudML training before every success criterion
   below passes.

</verification>

<success_criteria>

- Evaluation cadence, stage mutation, and rollback live in the runner.
- One explicit checkpoint is the resume authority; shell globs and stage files
  cannot determine a resumed stage.
- The evaluator/report error path fails closed with an audit event.
- Resume reproduces the trace and event-manager configuration.
- Disabled adaptive evaluation preserves normal PPO execution.
- Phase 2 is the first phase permitted to request long cluster experiments.

</success_criteria>

## Current status

The typed evaluator seam, cadence hook, provenance validation, typed gate
decisions, checkpoint metadata, RNG capture, and explicit rollback boundary are
implemented. The remaining gate is dedicated fake-runner save/load replay proof
for PPO state, live ranges, RNG, and transition trace.
