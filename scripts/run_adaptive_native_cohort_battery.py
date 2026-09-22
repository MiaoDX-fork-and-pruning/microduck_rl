#!/usr/bin/env python3
"""Evaluate a checkpoint over a fixed native reset/DR seed cohort.

The ordinary native checkpoint battery remains the single-seed primitive.  This
wrapper runs that primitive for a deterministic cohort and rebuilds one schema
v2 report whose raw evidence for each bucket comes from the worst cohort
member.  The conservative report is what the adaptive gate consumes; the
member reports and selected evidence remain available for audit.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from mjlab_microduck.evaluation.capability import (
    BUCKETS,
    CapabilityReport,
    build_capability_report,
    canonical_sha256,
)

from run_adaptive_native_battery import run_native


COHORT_MANIFEST_VERSION = "native-reset-dr-cohort-v1"
_TRAINING_ONLY_ENV_NAMES = (
    "MICRODUCK_ADAPTIVE_TRANSITION_PROBABILITY",
    "MICRODUCK_ADAPTIVE_TRANSITION_OVERRIDE",
    "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_MODE",
    "MICRODUCK_ADAPTIVE_TRANSITION_BOOTSTRAP_OVERRIDE",
    "MICRODUCK_ADAPTIVE_RESUME_CHECKPOINT",
    "MICRODUCK_ADAPTIVE_SENSOR_CORNER_FRACTION",
    "MICRODUCK_ADAPTIVE_SENSOR_RESET_FRACTION",
)
# Keep the old name as a compatibility alias for existing contract tests and
# callers that only know about transition overrides.
_TRANSITION_ENV_NAMES = _TRAINING_ONLY_ENV_NAMES


def resolve_cohort_seeds(
    evaluation_seed: int,
    *,
    cohort_size: int = 1,
    cohort_seeds: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Return a deterministic, duplicate-free seed tuple.

    An explicit list is useful for a frozen manifest.  Without one, the
    evaluation seed is the first member and subsequent members are contiguous
    seeds.  The default size of one preserves the old single-seed behavior.
    """
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
        raise ValueError("evaluation_seed must be an integer")
    if cohort_seeds is None:
        if isinstance(cohort_size, bool) or not isinstance(cohort_size, int) or cohort_size < 1:
            raise ValueError("cohort_size must be a positive integer")
        seeds = tuple(evaluation_seed + offset for offset in range(cohort_size))
    else:
        parsed: list[int] = []
        for seed in cohort_seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("cohort seeds must be integers")
            parsed.append(seed)
        seeds = tuple(parsed)
        if not seeds:
            raise ValueError("cohort_seeds must not be empty")
        if cohort_size != 1 and len(seeds) != cohort_size:
            raise ValueError("explicit cohort seeds disagree with cohort_size")
    if len(set(seeds)) != len(seeds):
        raise ValueError("cohort seeds must be unique")
    if seeds[0] != evaluation_seed:
        raise ValueError("the first cohort seed must equal evaluation_seed")
    return seeds


@contextmanager
def _native_evaluator_environment():
    """Keep training-only acquisition and sensor coverage out of native members."""
    saved = {name: os.environ.get(name) for name in _TRAINING_ONLY_ENV_NAMES}
    try:
        for name in _TRAINING_ONLY_ENV_NAMES:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _member_report_sha(payload: Mapping[str, object]) -> str:
    existing = payload.get("report_sha256")
    if isinstance(existing, str) and existing:
        return existing
    return canonical_sha256(payload)


def _validate_member_artifacts(
    payload: Mapping[str, object],
    *,
    seed: int,
    task: str,
    axis_mode: str,
    seed_set_id: str,
    checkpoint: Path,
    checkpoint_sha: str,
    source_sha: str,
) -> CapabilityReport:
    """Validate one member before it can influence worst-case selection."""
    report = CapabilityReport.from_dict(payload)
    metadata = report.payload["metadata"]
    if (
        metadata.get("task_id") != task
        or report.payload.get("axis_mode") != axis_mode
        or metadata.get("seed_set_id") != seed_set_id
        or metadata.get("checkpoint") != str(checkpoint)
        or metadata.get("checkpoint_sha256") != checkpoint_sha
        or metadata.get("source_sha") != source_sha
        or metadata.get("evaluation_seed") != seed
    ):
        raise ValueError("cohort member provenance mismatch")
    config_hash = canonical_sha256(report.payload.get("evaluator_config", {}))
    if metadata.get("evaluator_config_sha256") != config_hash:
        raise ValueError("cohort member evaluator config hash mismatch")
    cases = payload.get("cases")
    if not isinstance(cases, list) or {case.get("bucket") for case in cases if isinstance(case, Mapping)} != set(BUCKETS):
        raise ValueError("cohort member trace cases are incomplete")
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("invalid cohort member trace case")
        trace = case.get("trace")
        trace_hash = case.get("trace_sha256")
        if not isinstance(trace, str) or not isinstance(trace_hash, str):
            raise ValueError("cohort member trace provenance is missing")
        trace_path = Path(trace)
        if not trace_path.exists() or hashlib.sha256(trace_path.read_bytes()).hexdigest() != trace_hash:
            raise ValueError("cohort member trace hash mismatch")
    return report


def aggregate_cohort_reports(
    members: Iterable[Mapping[str, object]],
    *,
    checkpoint: Path,
    task: str,
    axis_mode: str,
    seed_set_id: str,
    cohort_seeds: tuple[int, ...],
    source_sha: str,
    output: Path | None = None,
) -> dict:
    """Build one conservative report from validated single-seed reports."""
    records = list(members)
    if not records or tuple(record.get("seed") for record in records) != cohort_seeds:
        raise ValueError("cohort member order or seed manifest does not match")
    if len({record.get("seed") for record in records}) != len(records):
        raise ValueError("cohort member seeds must be unique")
    checkpoint = checkpoint.resolve()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    validated: list[tuple[int, dict, CapabilityReport]] = []
    config_hash: str | None = None
    for record in records:
        seed = record.get("seed")
        payload = record.get("payload")
        if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(payload, Mapping):
            raise ValueError("invalid cohort member")
        report = _validate_member_artifacts(
            payload,
            seed=seed,
            task=task,
            axis_mode=axis_mode,
            seed_set_id=seed_set_id,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
            source_sha=source_sha,
        )
        member_config_hash = canonical_sha256(report.payload.get("evaluator_config", {}))
        if config_hash is None:
            config_hash = member_config_hash
        elif member_config_hash != config_hash:
            raise ValueError("cohort member evaluator config mismatch")
        # Keep the original artifact for trace/case evidence.  CapabilityReport
        # normalizes numeric values for schema comparison, which would turn
        # integer case lengths and seed IDs into floats in the merged native
        # audit artifact.
        validated.append((seed, deepcopy(dict(payload)), report))

    # Stable ordering makes equal-score ties reproducible and auditable.
    selected: dict[str, tuple[float, int, dict, dict | None]] = {}
    for bucket in BUCKETS:
        candidates = []
        for seed, payload, _ in validated:
            bucket_payload = payload["buckets"][bucket]
            candidates.append((float(bucket_payload["score"]), seed, bucket_payload["raw"], payload))
        selected[bucket] = min(candidates, key=lambda item: (item[0], item[1]))

    first_metadata = deepcopy(validated[0][1]["metadata"])
    first_config = deepcopy(validated[0][1]["evaluator_config"])
    first_config.update({
        "cohort_aggregation": "worst_bucket_score_v1",
        "cohort_size": len(cohort_seeds),
        "cohort_seeds": list(cohort_seeds),
        "member_evaluator_reports": [
            {
                "seed": int(record["seed"]),
                "path": str(record.get("path", "")),
                "sha256": _member_report_sha(record["payload"]),
            }
            for record in records
        ],
    })
    first_metadata.update({
        "task_id": task,
        "source_sha": source_sha,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seed_set_id": seed_set_id,
        "evaluation_seed": cohort_seeds[0],
        "cohort": {
            "version": COHORT_MANIFEST_VERSION,
            "seeds": list(cohort_seeds),
            "selected_seed_by_bucket": {
                bucket: int(selected[bucket][1]) for bucket in BUCKETS
            },
        },
    })
    # build_capability_report adds normalized defaults to the evaluator config;
    # hash that final config afterward so the report remains self-verifying.
    report = build_capability_report(
        {bucket: selected[bucket][2] for bucket in BUCKETS},
        axis_mode=axis_mode,
        evaluator_config=first_config,
        metadata={**first_metadata, "evaluator_config_sha256": "pending"},
    )
    payload = report.payload
    payload["metadata"]["evaluator_config_sha256"] = canonical_sha256(payload["evaluator_config"])

    member_by_seed = {seed: item for seed, item, _ in validated}
    selected_cases = []
    selected_seed_cases = []
    for bucket in BUCKETS:
        seed = selected[bucket][1]
        member = member_by_seed[seed]
        cases = {case.get("bucket"): case for case in member.get("cases", [])}
        seed_cases = {
            case.get("bucket"): case for case in member.get("seed_manifest", {}).get("cases", [])
        }
        if bucket not in cases or bucket not in seed_cases:
            raise ValueError(f"cohort member lacks native evidence for {bucket}")
        selected_case = deepcopy(cases[bucket])
        selected_case["cohort_seed"] = seed
        selected_cases.append(selected_case)
        selected_seed_case = deepcopy(seed_cases[bucket])
        selected_seed_case["cohort_seed"] = seed
        selected_seed_cases.append(selected_seed_case)

    payload["cases"] = selected_cases
    payload["seed_manifest"] = {
        "version": COHORT_MANIFEST_VERSION,
        "seed_set_id": seed_set_id,
        "evaluation_seed": cohort_seeds[0],
        "cohort_seeds": list(cohort_seeds),
        "cases": selected_seed_cases,
        "selected_seed_by_bucket": {
            bucket: int(selected[bucket][1]) for bucket in BUCKETS
        },
        "members": [
            {
                "seed": int(record["seed"]),
                "path": str(record.get("path", "")),
                "sha256": _member_report_sha(record["payload"]),
            }
            for record in records
        ],
        "consumed_state_fields": member_by_seed[cohort_seeds[0]].get("seed_manifest", {}).get(
            "consumed_state_fields", []
        ),
    }
    payload["cohort_manifest"] = deepcopy(payload["metadata"]["cohort"])
    payload["report_sha256"] = canonical_sha256(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_cohort(
    checkpoint: Path,
    output: Path,
    *,
    task: str,
    evaluation_seed: int,
    seed_set_id: str,
    axis_mode: str,
    source_sha: str = "unknown",
    cohort_size: int = 1,
    cohort_seeds: Iterable[int] | None = None,
    steps: int = 300,
    device: str = "cuda:0",
    onnx: Path | None = None,
    distribution: str = "final",
    zero_mode: str = "nominal",
) -> dict:
    seeds = resolve_cohort_seeds(
        evaluation_seed, cohort_size=cohort_size, cohort_seeds=cohort_seeds
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with _native_evaluator_environment():
        if len(seeds) == 1:
            payload = run_native(
                checkpoint, output.parent / f"{output.stem}.native", task=task,
                seed=seeds[0], steps=steps, device=device, onnx=onnx,
                seed_set_id=seed_set_id, axis_mode=axis_mode, source_sha=source_sha,
                distribution=distribution, zero_mode=zero_mode,
            )
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return payload

        members = []
        for seed in seeds:
            member_dir = output.parent / f"{output.stem}.cohort-{seed}"
            payload = run_native(
                checkpoint, member_dir, task=task, seed=seed, steps=steps,
                device=device, onnx=onnx, seed_set_id=seed_set_id,
                axis_mode=axis_mode, source_sha=source_sha,
                distribution=distribution, zero_mode=zero_mode,
            )
            members.append({
                "seed": seed,
                "path": str((member_dir / "native_capability.json").resolve()),
                "payload": payload,
            })
        return aggregate_cohort_reports(
            members, checkpoint=checkpoint, task=task, axis_mode=axis_mode,
            seed_set_id=seed_set_id, cohort_seeds=seeds, source_sha=source_sha,
            output=output,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--seed-set-id", required=True)
    parser.add_argument("--axis-mode", choices=("all_static", "com", "head_com", "composed"), required=True)
    parser.add_argument("--source-sha", default=os.environ.get("MICRODUCK_SOURCE_SHA", "unknown"))
    parser.add_argument("--cohort-size", type=int, default=1)
    parser.add_argument("--cohort-seeds", help="comma-separated explicit seeds")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default=os.environ.get("MICRODUCK_NATIVE_DEVICE", "cuda:0"))
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--distribution", choices=("initial", "final"), default="final")
    parser.add_argument("--zero-mode", choices=("nominal", "push"), default="nominal")
    args = parser.parse_args()
    explicit = None
    if args.cohort_seeds:
        try:
            explicit = tuple(int(value.strip()) for value in args.cohort_seeds.split(","))
        except ValueError as exc:
            parser.error(f"invalid --cohort-seeds: {exc}")
    payload = run_cohort(**{**vars(args), "cohort_seeds": explicit})
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
