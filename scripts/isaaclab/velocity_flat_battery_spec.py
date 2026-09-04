"""Single source of truth for cross-backend Velocity-Flat evaluation."""

from __future__ import annotations

from dataclasses import dataclass


SEED = 2026
CONTROL_HZ = 50
STEPS_PER_CASE = 300
NUM_ENVS = 16
RESET_BETWEEN_CASES = True


@dataclass(frozen=True)
class CommandCase:
    name: str
    command: tuple[float, float, float]
    min_response: float
    response_axis: int


CASES = (
    CommandCase("zero", (0.0, 0.0, 0.0), 0.0, 0),
    CommandCase("forward", (0.20, 0.0, 0.0), 0.02, 0),
    CommandCase("lateral", (0.0, 0.20, 0.0), 0.02, 1),
    CommandCase("yaw", (0.0, 0.0, 0.50), 0.05, 2),
    CommandCase("turn_left", (0.0, 0.0, -0.70), 0.05, 2),
    CommandCase("turn_right", (0.0, 0.0, 0.70), 0.05, 2),
)

# These are safety/response gates, not simulator-equality tolerances.  Exact
# trajectories are intentionally excluded because MuJoCo and PhysX solvers
# differ.  Both backends must nevertheless use the same gates.
MAX_TILT_RAD = 1.05
MAX_RESET_FRACTION = 0.005
MAX_ZERO_XY_SPEED_M_S = 0.08


def evaluate_case(
    case: CommandCase,
    *,
    finite: bool,
    mean_actual_xy: tuple[float, float] | list[float],
    mean_actual_yaw: float,
    max_tilt: float,
    reset_fraction: float,
) -> tuple[bool, list[str]]:
    """Apply identical behavioral gates to either backend's metrics."""

    failures: list[str] = []
    if not finite:
        failures.append("nonfinite")
    if max_tilt > MAX_TILT_RAD:
        failures.append("tilt")
    if reset_fraction > MAX_RESET_FRACTION:
        failures.append("resets")
    if case.name == "zero":
        if (mean_actual_xy[0] ** 2 + mean_actual_xy[1] ** 2) ** 0.5 > MAX_ZERO_XY_SPEED_M_S:
            failures.append("zero_drift")
    else:
        response = mean_actual_yaw if case.response_axis == 2 else mean_actual_xy[case.response_axis]
        requested = case.command[case.response_axis]
        if response * requested <= 0 or abs(response) < case.min_response:
            failures.append("command_response")
    return not failures, failures


__all__ = [
    "CASES", "CONTROL_HZ", "MAX_RESET_FRACTION", "MAX_TILT_RAD",
    "NUM_ENVS", "RESET_BETWEEN_CASES", "SEED", "STEPS_PER_CASE",
    "CommandCase", "evaluate_case",
]
