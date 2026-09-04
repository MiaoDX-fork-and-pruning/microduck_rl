"""IsaacLab task discovery with lazy simulator imports."""

from __future__ import annotations

import importlib.util

TASKS: tuple[str, ...] = (
    "IsaacLab-Velocity-Flat-MicroDuck",
    "IsaacLab-Velocity-Flat-MicroDuck-Adapted",
)


def available_tasks() -> tuple[str, ...]:
    """Return registered task names without starting Isaac Sim."""

    return TASKS


def require_isaaclab() -> None:
    """Raise an actionable error when simulator-backed code is requested."""

    if importlib.util.find_spec("isaaclab") is None:
        raise RuntimeError(
            "IsaacLab is unavailable. Run scripts/isaaclab/docker-run.sh "
            "after pulling the pinned Isaac Sim image."
        )
