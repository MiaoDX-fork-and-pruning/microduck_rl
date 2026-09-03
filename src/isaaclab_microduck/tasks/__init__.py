"""IsaacLab task registrations.

Framework registration is deferred until an IsaacLab runtime explicitly asks
for it; importing this module must remain safe for mjlab-only workflows.
"""

from __future__ import annotations

from ..registry import available_tasks, require_isaaclab


def register_tasks() -> None:
    """Validate the simulator runtime before framework registration."""

    require_isaaclab()


__all__ = ["available_tasks", "register_tasks"]
