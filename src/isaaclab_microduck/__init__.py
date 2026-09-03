"""Microduck IsaacLab backend.

The package is intentionally importable without Isaac Sim installed so mjlab
tools and CI can inspect the backend metadata without loading Omniverse.
"""

from .registry import available_tasks, require_isaaclab
from . import tasks

__all__ = ["available_tasks", "require_isaaclab", "tasks"]
