#!/usr/bin/env python3
"""Assemble Phase 2A evidence, failing closed on incomplete diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from mjlab_microduck.evaluation.capability import (
    BUCKETS,
    CapabilityReport,
    DEFAULT_TRACKING_TAU_S,
    TRACKING_METRIC_SAMPLEWISE,
    TRACKING_METRIC_SIGNED_EMA,
    canonical_sha256,
    tracking_error_metrics,
)

REQUIRED_ITERS = (500, 1000, 2000, 3999)
REQUIRED_BRANCHES = ("fixed", "static")
REQUIRED_SEEDS = (17, 23)
MIN_STEPS = 300
FIXED_TASK = "Mjlab-Velocity-Flat-MicroDuck"
STATIC_TASK = "Mjlab-Velocity-Flat-Adaptive-Static-MicroDuck"
CANONICAL_COMMANDS = {
    "zero": (0.0, 0.0, 0.0), "forward": (0.12, 0.0, 0.0),
    "lateral": (0.0, 0.12, 0.0), "yaw": (0.0, 0.0, 0.8),
    "turn-left": (0.08, 0.0, 0.8), "turn-right": (0.08, 0.0, -0.8),
}
CONSUMED_STATE_FIELDS = (
    "initial_root_position", "initial_joint_pos", "initial_joint_vel",
    "reset_body_ipos", "reset_dof_armature", "reset_friction_scale",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("report must be an object")
    return payload


def _trace_raw(
    trace: np.lib.npyio.NpzFile,
    bucket: str,
    requested_steps: int,
    dt: float,
    *,
    tracking_metric: str = TRACKING_METRIC_SAMPLEWISE,
    tracking_tau_s: float = DEFAULT_TRACKING_TAU_S,
) -> dict[str, float]:
    """Recompute gate inputs from the trace instead of trusting report summaries."""
    n = len(trace["action"])
    terminated = np.asarray(trace["terminated"], dtype=bool)
    truncated = np.asarray(trace["truncated"], dtype=bool)
    if terminated.shape != (n,) or truncated.shape != (n,):
        raise ValueError("native termination trace length mismatch")
    quat = np.asarray(trace["root_quaternion_wxyz"])
    if quat.shape != (n, 4):
        raise ValueError("native quaternion trace shape mismatch")
    tilt = np.arccos(np.clip(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1.0, 1.0))
    raw: dict[str, float] = {
        "survival_fraction": (n - int(np.any(terminated))) / requested_steps,
        "tilt_p95_rad": float(np.percentile(tilt, 95)),
        "episode_length_mean": n * dt,
        "fall_rate": float(np.any(terminated)),
        "action_magnitude_mean": float(np.abs(trace["action"]).mean()),
    }
    if bucket == "zero":
        position = np.asarray(trace["root_position"])
        initial = np.asarray(trace["initial_root_position"])
        if position.shape != (n, 3) or initial.shape != (3,):
            raise ValueError("native position trace shape mismatch")
        raw["zero_drift_m"] = float(np.linalg.norm(position[-1, :2] - initial[:2]))
    else:
        command = np.asarray(trace["command"])
        velocity = np.asarray(trace["linear_velocity_b"])
        angular = np.asarray(trace["angular_velocity_b"])
        if command.shape != (n, 13) or velocity.shape != (n, 3) or angular.shape != (n, 3):
            raise ValueError("native command/state trace shape mismatch")
        if tracking_metric not in (TRACKING_METRIC_SAMPLEWISE, TRACKING_METRIC_SIGNED_EMA):
            raise ValueError("unsupported native tracking metric")
        if bucket in ("forward", "lateral"):
            axis = 0 if bucket == "forward" else 1
            samplewise, signed_ema = tracking_error_metrics(
                velocity[:, axis], command[:, axis], dt=dt, tau_s=tracking_tau_s
            )
            raw["tracking_error_m_s"] = float(
                signed_ema if tracking_metric == TRACKING_METRIC_SIGNED_EMA else samplewise
            )
            if tracking_metric == TRACKING_METRIC_SIGNED_EMA:
                raw["tracking_error_samplewise_m_s"] = float(samplewise)
        else:
            samplewise, signed_ema = tracking_error_metrics(
                angular[:, 2], command[:, 2], dt=dt, tau_s=tracking_tau_s
            )
            raw["angular_tracking_error_rad_s"] = float(
                signed_ema if tracking_metric == TRACKING_METRIC_SIGNED_EMA else samplewise
            )
            if tracking_metric == TRACKING_METRIC_SIGNED_EMA:
                raw["angular_tracking_error_samplewise_rad_s"] = float(samplewise)
    return raw


def _assert_raw_matches(payload: dict, bucket: str, trace: np.lib.npyio.NpzFile, requested_steps: int) -> None:
    config = payload["evaluator_config"]
    dt = config.get("step_dt", 0.02)
    if not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid native step_dt")
    tracking_metric = config.get("tracking_metric", TRACKING_METRIC_SAMPLEWISE)
    tracking_tau_s = config.get("tracking_metric_tau_s", DEFAULT_TRACKING_TAU_S)
    if not isinstance(tracking_tau_s, (int, float)) or not np.isfinite(tracking_tau_s) or tracking_tau_s <= 0:
        raise ValueError("invalid native tracking metric tau")
    expected = _trace_raw(
        trace,
        bucket,
        requested_steps,
        dt,
        tracking_metric=tracking_metric,
        tracking_tau_s=float(tracking_tau_s),
    )
    reported = payload["buckets"][bucket]["raw"]
    for key, value in expected.items():
        if key not in reported or not np.isclose(float(reported[key]), value, rtol=1e-5, atol=1e-6):
            raise ValueError(f"native raw metric mismatch: {bucket}.{key}")


def validate_native(path: Path, *, expected_task: str | None = None, require_parity: bool = False) -> dict:
    """Validate the canonical gate and the underlying native trace evidence."""
    payload = _read(path)
    if payload.get("schema_version") != 2:
        raise ValueError("legacy native schema rejected; schema 2 is required")
    if "report_sha256" in payload and payload["report_sha256"] != canonical_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    ):
        raise ValueError("native report hash mismatch")
    CapabilityReport.from_dict(payload)
    if not payload["aggregate"]["valid"]:
        raise ValueError("native aggregate is invalid")
    metadata = payload["metadata"]
    config = payload.get("evaluator_config", {})
    if config.get("name") != "native_mjlab_bam_v2":
        raise ValueError("native evaluator contract is not native_mjlab_bam_v2")
    if config.get("environment_profile") != "training":
        raise ValueError("native evaluation must use the training environment profile")
    if config.get("bucket_isolation") != "fresh_environment":
        raise ValueError("native buckets require fresh_environment isolation")
    if config.get("tracking_metric") != TRACKING_METRIC_SIGNED_EMA:
        raise ValueError("native evaluator must use signed_ema_v1 tracking")
    if config.get("tracking_metric_tau_s") != DEFAULT_TRACKING_TAU_S:
        raise ValueError("native evaluator tracking tau must be 0.5 s")
    if config.get("zero_mode") != "nominal":
        raise ValueError("product native gate requires nominal zero without pushes")
    if canonical_sha256(config.get("commands")) != canonical_sha256(CANONICAL_COMMANDS):
        raise ValueError("native evaluator commands differ from canonical product commands")
    if expected_task is not None and metadata.get("task_id") != expected_task:
        raise ValueError("native task does not match ladder branch")
    if metadata.get("evaluator_config_sha256") != canonical_sha256(config):
        raise ValueError("evaluator config hash mismatch")
    checkpoint = Path(metadata["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint missing or hash mismatch")
    if not metadata.get("source_sha") or metadata["source_sha"] == "unknown":
        raise ValueError("native source provenance is missing")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != len(BUCKETS):
        raise ValueError("exactly six native trace cases are required")
    if {case.get("bucket") for case in cases} != set(BUCKETS):
        raise ValueError("native cases must contain each canonical bucket once")
    seed_manifest = payload.get("seed_manifest", {})
    if seed_manifest.get("version") != "native-reset-dr-v3":
        raise ValueError("native seed manifest version missing or unsupported")
    evaluation_seed = metadata.get("evaluation_seed")
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, (int, float)) or not np.isfinite(evaluation_seed) or not float(evaluation_seed).is_integer():
        raise ValueError("native evaluation seed must be an integer")
    if seed_manifest.get("startup_seed") != evaluation_seed or seed_manifest.get("seed_set_id") != metadata["seed_set_id"]:
        raise ValueError("native seed manifest startup seed or seed set mismatch")
    seed_cases = seed_manifest.get("cases")
    if not isinstance(seed_cases, list) or len(seed_cases) != len(BUCKETS) or {case.get("bucket") for case in seed_cases} != set(BUCKETS):
        raise ValueError("native seed manifest requires exactly six canonical buckets")
    seed_cases = {case["bucket"]: case for case in seed_cases}
    if seed_manifest.get("consumed_state_fields") != list(CONSUMED_STATE_FIELDS):
        raise ValueError("native consumed state fields do not match reset contract")
    for offset, bucket in enumerate(BUCKETS):
        if seed_cases[bucket].get("reset_seed") != evaluation_seed + offset:
            raise ValueError("native bucket reset seed mismatch")
    for case in cases:
        steps = case.get("steps", 0)
        requested_steps = case.get("requested_steps", config.get("steps", 0))
        if not isinstance(requested_steps, (int, float)) or not float(requested_steps).is_integer() or requested_steps < MIN_STEPS:
            raise ValueError(f"native request is shorter than product window: {case.get('bucket')}")
        if requested_steps != config.get("steps"):
            raise ValueError("native requested steps disagree with evaluator config")
        if not isinstance(steps, int) or steps < 1 or steps > requested_steps:
            raise ValueError(f"empty native trace: {case.get('bucket')}")
        if not case.get("finite_61d_14d"):
            raise ValueError("native case lacks finite 61D/14D evidence")
        trace_path = Path(case.get("trace", ""))
        if not trace_path.is_absolute():
            trace_path = path.parent / trace_path
        if not trace_path.is_file() or sha256(trace_path) != case.get("trace_sha256"):
            raise ValueError("native trace missing or hash mismatch")
        with np.load(trace_path, allow_pickle=False) as trace:
            required = ("terminated", "truncated", "root_quaternion_wxyz", "root_position",
                        "initial_root_position", "command", "linear_velocity_b", "angular_velocity_b")
            if any(field not in trace for field in required):
                raise ValueError("native trace lacks raw state or termination evidence")
            for field in trace.files:
                if trace[field].dtype.kind in "fci" and not np.isfinite(trace[field]).all():
                    raise ValueError(f"nonfinite native {field}")
            if any(field not in trace or trace[field].size == 0 for field in CONSUMED_STATE_FIELDS):
                raise ValueError("native trace lacks consumed reset state")
            consumed_hash = canonical_sha256({field: trace[field].tolist() for field in CONSUMED_STATE_FIELDS})
            if seed_cases[case["bucket"]].get("consumed_state_sha256") != consumed_hash:
                raise ValueError("native consumed state hash mismatch")
            for field, width in (("observation", 61), ("action", 14)):
                if field not in trace or trace[field].shape != (steps, width):
                    raise ValueError(f"native trace {field} shape mismatch")
                if not np.isfinite(trace[field]).all():
                    raise ValueError(f"nonfinite native {field}")
            if trace["observation"].shape[0] != steps:
                raise ValueError("native trace row count does not match case steps")
            if steps < requested_steps:
                terminated = trace.get("terminated")
                truncated = trace.get("truncated")
                terminal = bool(
                    (terminated is not None and len(terminated) and bool(terminated[-1]))
                    or (truncated is not None and len(truncated) and bool(truncated[-1]))
                )
                if not terminal:
                    raise ValueError(f"short native trace without terminal failure: {case.get('bucket')}")
            if "terminated" not in trace or "truncated" not in trace:
                raise ValueError("native trace lacks termination markers")
            if len(trace["terminated"]) != steps or len(trace["truncated"]) != steps:
                raise ValueError("native termination trace length mismatch")
            if np.any(trace["terminated"][:-1]) or np.any(trace["truncated"][:-1]):
                raise ValueError("native trace continues after terminal state")
            if trace["terminated"].dtype.kind != "b" or trace["truncated"].dtype.kind != "b":
                raise ValueError("native termination markers must be boolean")
            for field, shape in (("root_position", (steps, 3)), ("initial_root_position", (3,)),
                                 ("linear_velocity_b", (steps, 3)), ("angular_velocity_b", (steps, 3))):
                if trace[field].shape != shape:
                    raise ValueError(f"native trace {field} shape mismatch")
            if trace["command"].shape != (steps, 13) or not np.array_equal(trace["observation"][:, 48:], trace["command"]):
                raise ValueError("native observation command block mismatch")
            if case["bucket"] in config.get("commands", {}):
                command = np.array((*config["commands"][case["bucket"]], *([0.0] * 10)), dtype=trace["command"].dtype)
                if not np.all(trace["command"] == command):
                    raise ValueError("native frozen command mismatch")
            _assert_raw_matches(payload, case["bucket"], trace, requested_steps)
            if require_parity:
                if "onnx_action" not in trace or trace["onnx_action"].shape != (steps, 14):
                    raise ValueError("ONNX parity trace missing or shape mismatch")
                if not np.isfinite(trace["onnx_action"]).all() or not np.allclose(
                    trace["action"], trace["onnx_action"], atol=1e-4, rtol=1e-3
                ):
                    raise ValueError("ONNX parity trace fails tolerance")
                if not isinstance(case.get("parity"), dict) or not case["parity"].get("passed"):
                    raise ValueError("ONNX parity case missing or failed")
    return payload


def _artifact(path: Path, errors: list[str], *, native: bool = False, **kwargs) -> tuple[dict, dict | None]:
    item = {"path": str(path), "sha256": sha256(path) if path.is_file() else None}
    try:
        payload = validate_native(path, **kwargs) if native else _read(path)
        if not native:
            CapabilityReport.from_dict(payload)
            if not payload["aggregate"]["valid"]:
                raise ValueError("capability aggregate is invalid")
        item["aggregate"] = payload["aggregate"]
        item["validated"] = True
        return item, payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"{path}: {exc}")
        item["validated"] = False
        return item, None


def _case_trace(report_path: Path, case: dict) -> Path:
    path = Path(case["trace"])
    return path if path.is_absolute() else report_path.parent / path


def validate_replay(replay_a: Path, replay_b: Path, replay_different: Path) -> dict:
    """Compare consumed arrays, not trace labels or caller-supplied pass flags."""
    paths = (replay_a, replay_b, replay_different)
    a, b, different = [validate_native(path) for path in paths]
    for other in (b, different):
        for key in ("checkpoint_sha256", "evaluator_config_sha256", "task_id", "source_sha"):
            if a["metadata"][key] != other["metadata"][key]:
                raise ValueError(f"replay {key} mismatch")
        if a["axis_mode"] != other["axis_mode"]:
            raise ValueError("replay axis mode mismatch")
    if canonical_sha256(a["seed_manifest"]) != canonical_sha256(b["seed_manifest"]):
        raise ValueError("same-seed replay manifests differ")
    if a["metadata"]["evaluation_seed"] != b["metadata"]["evaluation_seed"]:
        raise ValueError("same-seed replay evaluation seeds differ")
    if a["metadata"]["seed_set_id"] == different["metadata"]["seed_set_id"]:
        raise ValueError("different replay requires distinct seed-set labels")
    seeds = [{report["seed_manifest"]["startup_seed"], *(case["reset_seed"] for case in report["seed_manifest"]["cases"])} for report in (a, different)]
    if seeds[0] & seeds[1]:
        raise ValueError("gate and held-out consumed seed sets overlap")
    case_maps = [{case["bucket"]: case for case in report["cases"]} for report in (a, b, different)]
    comparisons = []
    for bucket in BUCKETS:
        with np.load(_case_trace(replay_a, case_maps[0][bucket]), allow_pickle=False) as trace_a, np.load(
            _case_trace(replay_b, case_maps[1][bucket]), allow_pickle=False
        ) as trace_b, np.load(_case_trace(replay_different, case_maps[2][bucket]), allow_pickle=False) as trace_different:
            if set(trace_a.files) != set(trace_b.files):
                raise ValueError(f"same-seed replay trace fields differ: {bucket}")
            for field in trace_a.files:
                left, right = trace_a[field], trace_b[field]
                if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
                    raise ValueError(f"same-seed replay trace mismatch: {bucket}.{field}")
            changed = [field for field in CONSUMED_STATE_FIELDS if not np.array_equal(trace_a[field], trace_different[field])]
            if not changed:
                raise ValueError(f"different seed did not change consumed state: {bucket}")
            comparisons.append({"bucket": bucket, "exact_replay_fields": sorted(trace_a.files), "different_seed_changed_consumed_fields": changed})
    return {
        "status": "validated", "reports": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "seed_sets": [{"seed_set_id": report["metadata"]["seed_set_id"], "consumed_seeds": sorted(group)} for report, group in zip((a, different), seeds, strict=True)],
        "buckets": comparisons,
    }


def validate_ownership_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    if root.tag not in ("testsuite", "testsuites"):
        raise ValueError("ownership evidence is not JUnit XML")
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("ownership JUnit contains no tests")
    for suite in root.iter("testsuite"):
        if int(suite.get("failures", "0")) or int(suite.get("errors", "0")):
            raise ValueError("ownership JUnit reports failures or errors")
        direct_cases = suite.findall("testcase")
        if direct_cases and int(suite.get("tests", "-1")) != len(direct_cases):
            raise ValueError("ownership JUnit test count mismatch")
    if any(case.find("failure") is not None or case.find("error") is not None for case in cases):
        raise ValueError("ownership JUnit contains failed tests")
    required = {"test_adaptive_factory_is_separate_static_initial_slice", "test_adaptive_factory_exposes_explicit_axis_modes"}
    passed = {
        case.get("name") for case in cases
        if case.get("classname", "").split(".")[-1] == "test_mjlab_adaptive_velocity_config"
        and case.find("skipped") is None
    }
    if not required <= passed:
        raise ValueError("ownership JUnit lacks passing exact curriculum term-set tests")
    return {"status": "validated", "path": str(path), "sha256": sha256(path), "test_count": len(cases), "required_passed_tests": sorted(required)}


def assemble(native_ladder: Path, native_final: Path, cpu_ladder: Path, calibration: Path, parity: Path, output: Path,
             replay_a: Path | None = None, replay_b: Path | None = None, replay_different: Path | None = None,
             ownership_junit: Path | None = None) -> dict:
    errors: list[str] = []
    entries, final, native_payloads, cpu_payloads = [], [], [], []
    for branch in REQUIRED_BRANCHES:
        task = FIXED_TASK if branch == "fixed" else STATIC_TASK
        for seed in REQUIRED_SEEDS:
            for iteration in REQUIRED_ITERS:
                name = f"{branch}-s{seed}-i{iteration}"
                native, native_payload = _artifact(native_ladder / name / "native_capability.json", errors, native=True, expected_task=task)
                cpu, cpu_payload = _artifact(cpu_ladder / name / "capability.json", errors)
                if native_payload is not None:
                    native_payloads.append(native_payload)
                if cpu_payload is not None:
                    cpu_payloads.append(cpu_payload)
                entries.append({"branch": branch, "training_seed": seed, "checkpoint_index": iteration, "native": native, "cpu_transfer": cpu})
    for branch in REQUIRED_BRANCHES:
        for seed in REQUIRED_SEEDS:
            name = f"{branch}{seed}"
            artifact, _ = _artifact(native_final / name / "native_capability.json", errors, native=True, expected_task=FIXED_TASK if branch == "fixed" else STATIC_TASK)
            final.append({"name": name, **artifact})
    parity_artifact, parity_payload = _artifact(parity, errors, native=True, require_parity=True)
    calibration_artifact = {"path": str(calibration), "sha256": sha256(calibration) if calibration.is_file() else None}
    try:
        calibration_payload = _read(calibration)
        if calibration_payload.get("status") != "complete":
            raise ValueError("calibration is incomplete")
        for key in ("ema_alpha", "dwell", "preservation_tolerance", "curriculum_thresholds"):
            if key not in calibration_payload.get("calibrated_parameters", {}):
                raise ValueError(f"calibration lacks {key}")
        digest_input = {key: value for key, value in calibration_payload.items() if key != "config_sha256"}
        if calibration_payload.get("config_sha256") != canonical_sha256(digest_input):
            raise ValueError("calibration hash mismatch")
        calibration_artifact["config_sha256"] = calibration_payload["config_sha256"]
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"{calibration}: {exc}")
    ownership_evidence, replay_evidence = None, None
    try:
        if ownership_junit is None:
            raise ValueError("ownership test evidence is not attached")
        ownership_evidence = validate_ownership_junit(ownership_junit)
    except (OSError, ValueError, ET.ParseError) as exc:
        errors.append(f"ownership evidence: {exc}")
    try:
        if replay_a is None or replay_b is None or replay_different is None:
            raise ValueError("same-seed replay and disjoint consumed gate/held-out seed evidence are not attached")
        replay_evidence = validate_replay(replay_a, replay_b, replay_different)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"seed replay evidence: {exc}")
    complete_ladder = len(native_payloads) == len(entries)
    native_verdict = "incomplete"
    if complete_ladder:
        passed = [bool(item["aggregate"]["passed"]) for item in native_payloads]
        native_verdict = "all_pass" if all(passed) else "mixed" if any(passed) else "all_fail"
    cpu_verdict = "incomplete"
    if len(cpu_payloads) == len(entries):
        passed = [bool(item["aggregate"]["passed"]) for item in cpu_payloads]
        cpu_verdict = "all_pass" if all(passed) else "mixed" if any(passed) else "all_fail"
    manifest = {
        "schema_version": 2, "phase": "02A", "status": "incomplete" if errors else "valid", "errors": errors,
        "checkpoint_index_convention": "model_3999 is the zero-indexed endpoint of 4000 training iterations",
        "ladder": entries, "native_final": final, "calibration": calibration_artifact, "onnx_parity": parity_artifact,
        "ownership_evidence": ownership_evidence, "seed_replay_evidence": replay_evidence,
        "verdicts": {"adaptive_config_ownership": "pass" if ownership_evidence else "unverified", "seed_consumption": "pass" if replay_evidence else "unverified", "native_learning_diagnostic": native_verdict, "export_observation_parity": "pass" if parity_payload is not None else "unverified", "cpu_actuator_transfer": cpu_verdict, "matched_budget_campaign_authorized": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-ladder", type=Path, required=True)
    parser.add_argument("--native-final", type=Path, required=True)
    parser.add_argument("--cpu-ladder", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-a", type=Path)
    parser.add_argument("--replay-b", type=Path)
    parser.add_argument("--replay-different", type=Path)
    parser.add_argument("--ownership-junit", type=Path)
    args = parser.parse_args()
    manifest = assemble(**vars(args))
    print(json.dumps({"status": manifest["status"], "errors": manifest["errors"], "output": str(args.output)}, indent=2))
    return 0 if manifest["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
