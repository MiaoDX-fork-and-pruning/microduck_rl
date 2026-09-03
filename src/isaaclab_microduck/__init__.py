"""Microduck IsaacLab backend.

The package is intentionally importable without Isaac Sim installed so mjlab
tools and CI can inspect the backend metadata without loading Omniverse.
"""

from .registry import available_tasks, require_isaaclab

__all__ = ["available_tasks", "require_isaaclab"]
