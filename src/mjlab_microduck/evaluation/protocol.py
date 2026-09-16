"""Typed seam between the trainer and frozen capability evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from .capability import CapabilityReport


class FrozenCapabilityEvaluator(Protocol):
    def evaluate(
        self,
        *,
        checkpoint_path: Path,
        task_id: str,
        axis_mode: str,
        curriculum_state: Mapping[str, object],
        iteration: int,
        seed_set_id: str,
    ) -> CapabilityReport: ...
