#!/usr/bin/env python3
"""Compose deterministic specialist evidence reels from reviewed diagnostics.

This is a packaging compositor, not a physics simulator: each segment is an
immutable reviewed diagnostic clip. The metadata makes that distinction
explicit while preserving the canonical scenario event schedule.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from specialist_scenario import load_scenario, scenario_events


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compose(manifest_path: Path, scenario_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    scenario, frames = load_scenario(scenario_path)
    policies = {p["id"]: p for p in manifest["policies"]}
    events = scenario_events(frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"track-{scenario['track'].lower()}-stitched.mp4"
    writer = None
    segments = []
    try:
        for i, event in enumerate(events):
            policy = policies.get(event.policy_id)
            if policy is None:
                raise ValueError(f"scenario policy missing from manifest: {event.policy_id}")
            source = Path(policy["artifacts"]["diagnostic_video"])
            cap = imageio.get_reader(str(source))
            meta = cap.get_meta_data()
            first = next(iter(cap), None)
            if first is None:
                cap.close()
                raise ValueError(f"cannot decode diagnostic video: {source}")
            height, width = first.shape[:2]
            if writer is None:
                writer = imageio.get_writer(str(output), fps=50, codec="libx264", quality=7)
            stop = events[i + 1].step if i + 1 < len(events) else len(frames)
            count = stop - event.step
            written = 0
            frame = first
            while written < count:
                writer.append_data(frame)
                written += 1
                try:
                    frame = next(iter(cap))
                except StopIteration:
                    cap.close()
                    cap = imageio.get_reader(str(source))
                    frame = next(iter(cap))
            cap.close()
            segments.append({"policy_id": event.policy_id, "start_step": event.step,
                             "frames": written, "source": str(source),
                             "source_sha256": sha256(source),
                             "expected_outcome": event.expected_outcome})
    finally:
        if writer is not None:
            writer.close()
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("reel was not written")
    metadata = {
        "schema_version": 1, "track": scenario["track"], "stitched": True,
        "continuous_physics_rollout": False, "command_rate_hz": 50,
        "seed": scenario["seed"], "frames": sum(s["frames"] for s in segments),
        "duration_s": sum(s["frames"] for s in segments) / 50.0,
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "scenario": str(scenario_path), "scenario_sha256": sha256(scenario_path),
        "segments": segments, "reel_sha256": sha256(output),
        "failure_note": "stitched reviewed diagnostics; continuous headless switching is deferred",
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"reel": str(output), "metadata": str(metadata_path), "frames": metadata["frames"], "duration_s": metadata["duration_s"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(compose(args.manifest, args.scenario, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
