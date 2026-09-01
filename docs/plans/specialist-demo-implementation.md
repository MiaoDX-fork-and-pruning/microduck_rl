# Specialist demo implementation plan

Status: **ready for execution**

This plan turns the accepted specialist inventory into a reproducible,
scenario-driven demo. It is an implementation contract, not a training plan.

## Goal

Produce a reviewable demo package containing deterministic switching scenarios,
deployment-style PT/ONNX rehearsals, integrated reels, per-policy diagnostics,
and an HTML gallery indexed by immutable artifact hashes.

## Boundaries

- The accepted 13-policy inventory is fixed; do not retrain or silently replace
  checkpoints.
- Track A is the primary locomotion/pose reel. Track B is a separate showcase
  for the roller/dynamic policies and does not share an unsafe cross-model
  physics scene.
- Unsupported transitions are explicit non-executable records. They must never
  be silently mapped to another policy.
- No runtime repository, robot hardware, Generalist 71D schema, or
  Rough/Backlash expansion work is included.

## Phases

### D0. Freeze inputs and stage artifacts

Owner: artifact/package maintainer.

- Resolve all 13 entries from
  `cloudml/specialist-final-checkpoints-remediated-facd4f4.json`.
- Stage `.pt`, `.onnx`, metadata, evaluation report, parity report, and
  diagnostic video under `artifacts/specialists/<policy_id>/` (or record the
  agreed immutable JuiceFS paths).
- Generate a real manifest with paths and SHA-256 values; do not use the empty
  example hashes as evidence.
- Record source commit, image digest, export command, manifest hash, and artifact
  provenance.

Stop gate: every accepted policy has all required artifacts and matching hashes.

### D1. Define executable scenario contracts

Owner: scenario/router maintainer.

- Keep `docs/specialist_demo_scenario.json` as the Track A source of truth.
- Add a separate Track B scenario only after its compatible model/session and
  scene requirements are explicit.
- Define `seed`, 50 Hz frame grid, policy IDs, command blocks, switch times,
  dwell/non-interruptible windows, and measurable outcome fields.
- Represent unsupported transitions outside executable transitions with a reason
  and test that the compiler skips them rather than attempting them.
- Add validator coverage for required seed and scenario/manifest policy
  alignment.

Stop gate: scenarios compile deterministically and every executable edge is
legal for the selected scene/router; all other edges are explicit unsupported
records.

### D2. Extend and rehearse the router

Owner: inference maintainer.

- Verify the existing Track A route with the real staged policies.
- Add explicit Track B sessions only where the model and MuJoCo scene are
  compatible; otherwise keep Track B as an independently rendered scenario.
- Preserve 61D observations, command-slot semantics, finite `[14]` actions,
  no fallback, and action-history/reset behavior.
- Test switch guards for sit/stand, ground-pick, kick, roulade, slope, and
  roller transitions.

Stop gate: router tests cover every policy included in a scenario and reject
  missing or unsupported policy IDs without mutation or fallback.

### D3. Export and end-to-end parity

Owner: deployment/evidence maintainer.

- Export every included checkpoint through `scripts/export.py` so normalization
  is baked into ONNX.
- Run golden observation parity for zero-command and command-extreme cases.
- Run the same scenario with PT and ONNX paths and compare seed, frame count,
  transition events, command blocks, reset count, and action tolerances.
- Persist parity JSON next to each reel.

Stop gate: per-policy and scenario-level parity pass, or a recorded, reviewed
reason for an intentionally PT-only artifact.

### D4. Render reels and diagnostics

Owner: demo renderer.

- Render Track A and Track B as separate 60--90 second reels from their scenario
  files, with fixed seed and 50 Hz control.
- Produce PT and ONNX versions where D3 passes.
- Record transition events, duration, resets, outcome metrics, decoder status,
  and failure notes. Keep the existing per-policy diagnostic clips.

Stop gate: every included policy appears in a reel or has an explicit
unsupported/deferred record; all produced videos decode and match scenario
metadata.

### D5. Build the reproducibility gallery

Owner: package maintainer.

- Extend `scripts/build_video_gallery.py` input/indexing as needed for
  integrated reels, scenario metadata, PT/ONNX labels, hashes, metrics, and
  failure/unsupported notes.
- Generate the HTML gallery from the validated manifest and scenario files.
- Include exact commands and source/manifest/scenario hashes.

Stop gate: gallery references only validated artifacts and exposes enough
metadata to reproduce each reel.

### D6. Final acceptance and handoff

Owner: primary maintainer.

Run:

```bash
uv run python scripts/validate_specialist_artifacts.py \
  artifacts/specialist_artifact_manifest.json \
  docs/specialist_demo_scenario.json
uv run --with pytest pytest tests/
```

Then verify video decoding, scenario/manifest coverage, PT/ONNX parity reports,
and produce `specialist-demo-completion.json`. Update
`docs/status/active/specialist-policy-demo.md` from pending to completed only
after all stop gates pass.

## Acceptance criteria

1. 13 accepted checkpoints remain immutable and are traceable to their source
   inventory.
2. Every included artifact has valid metadata, hashes, evaluation, and parity
   evidence.
3. Track A and Track B scenarios are deterministic, legal, and separately
   reproducible; unsupported edges are non-executable and documented.
4. Reels, diagnostics, and gallery are decodable, indexed, and tied to exact
   scenario and manifest hashes.
5. Tests and offline validation pass without changing the 61D/14D contract.

## Plan Ledger

- 2026-08-31: split from the historical specialist training plan after
  planning-loop review; D0--D6 are pending.
- Planning-loop decision: do not promise one 13-policy reel or infer Track B
  router support; prove compatible slices independently.
