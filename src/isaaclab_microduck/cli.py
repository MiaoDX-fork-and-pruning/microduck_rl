"""Command-line entry points that do not require Isaac Sim."""

from __future__ import annotations

from .registry import available_tasks


def list_tasks() -> None:
    """Print registered IsaacLab task names."""

    for task in available_tasks():
        print(task)
