#!/usr/bin/env python3
"""Compile the canonical specialist demo scenario into a deterministic schedule."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMMAND_DIM = 13
TWIST_DIM = 3


@dataclass(frozen=True)
class ScenarioFrame:
    step: int
    time_s: float
    policy_id: str
    command: tuple[float, ...]
    expected_outcome: str


def _command_block(raw: Any, location: str) -> tuple[float, ...]:
    if raw is None:
        values: list[Any] = []
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError(f"{location}.command must be an array")
    if len(values) not in (0, TWIST_DIM, COMMAND_DIM):
        raise ValueError(f"{location}.command must contain 3 or 13 values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{location}.command values must be finite numbers")
    command = tuple(float(value) for value in values)
    if any(value != value or value in (float("inf"), float("-inf")) for value in command):
        raise ValueError(f"{location}.command values must be finite numbers")
    return command + (0.0,) * (COMMAND_DIM - len(command))


def compile_scenario(scenario: dict[str, Any]) -> list[ScenarioFrame]:
    rate = scenario.get("command_rate_hz")
    duration = scenario.get("duration_s")
    transitions = scenario.get("transitions")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("scenario.command_rate_hz must be a positive integer")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("scenario.duration_s must be positive")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("scenario.transitions must contain at least one transition")

    total_steps_float = float(duration) * rate
    if not total_steps_float.is_integer():
        raise ValueError("scenario.duration_s must align to command_rate_hz")
    total_steps = int(total_steps_float)
    events: list[tuple[int, dict[str, Any], tuple[float, ...]]] = []
    previous_to: str | None = None
    previous_step = -1
    for index, transition in enumerate(transitions):
        location = f"scenario.transitions[{index}]"
        if not isinstance(transition, dict):
            raise ValueError(f"{location} must be an object")
        at_s = transition.get("at_s")
        if not isinstance(at_s, (int, float)) or isinstance(at_s, bool):
            raise ValueError(f"{location}.at_s must be numeric")
        step_float = float(at_s) * rate
        if not step_float.is_integer():
            raise ValueError(f"{location}.at_s must align to command_rate_hz")
        step = int(step_float)
        if step <= previous_step or step < 0 or step >= total_steps:
            raise ValueError(f"{location}.at_s must increase within the scenario")
        source, target = transition.get("from"), transition.get("to")
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise ValueError(f"{location}.from and .to must be policy ids")
        if previous_to is not None and source != previous_to:
            raise ValueError(
                f"{location}.from {source!r} does not continue previous policy {previous_to!r}"
            )
        outcome = transition.get("expected_outcome")
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError(f"{location}.expected_outcome must be non-empty")
        events.append((step, transition, _command_block(transition.get("command"), location)))
        previous_step, previous_to = step, target

    if events[0][0] != 0:
        raise ValueError("scenario.transitions[0].at_s must be 0")

    frames: list[ScenarioFrame] = []
    for event_index, (start, event, command) in enumerate(events):
        stop = events[event_index + 1][0] if event_index + 1 < len(events) else total_steps
        frames.extend(
            ScenarioFrame(
                step=step,
                time_s=step / rate,
                policy_id=event["to"],
                command=command,
                expected_outcome=event["expected_outcome"],
            )
            for step in range(start, stop)
        )
    return frames


def load_scenario(path: Path) -> tuple[dict[str, Any], list[ScenarioFrame]]:
    scenario = json.loads(path.read_text(encoding="utf-8"))
    return scenario, compile_scenario(scenario)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, help="Write the expanded frame schedule as JSON")
    args = parser.parse_args()
    scenario, frames = load_scenario(args.scenario)
    summary = {
        "scenario": str(args.scenario),
        "seed": scenario.get("seed"),
        "frames": len(frames),
        "duration_s": scenario["duration_s"],
        "command_rate_hz": scenario["command_rate_hz"],
        "policy_ids": list(dict.fromkeys(frame.policy_id for frame in frames)),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([frame.__dict__ for frame in frames], indent=2) + "\n",
            encoding="utf-8",
        )
        summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
