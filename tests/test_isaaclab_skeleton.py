from __future__ import annotations

import pytest

from isaaclab_microduck import available_tasks, require_isaaclab


def test_task_registry_is_available_without_simulator() -> None:
    assert available_tasks() == ("IsaacLab-Velocity-Flat-MicroDuck",)


def test_simulator_import_is_lazy() -> None:
    with pytest.raises(RuntimeError, match="IsaacLab is unavailable"):
        require_isaaclab()
