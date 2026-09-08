from __future__ import annotations

import pytest

from isaaclab_microduck import available_tasks, require_isaaclab
from isaaclab_microduck.cli import list_tasks
from isaaclab_microduck.tasks import register_tasks


def test_task_registry_is_available_without_simulator() -> None:
    assert available_tasks() == (
        "IsaacLab-Velocity-Flat-MicroDuck",
        "IsaacLab-Velocity-Flat-MicroDuck-Adapted",
    )


def test_simulator_import_is_lazy() -> None:
    with pytest.raises(RuntimeError, match="IsaacLab is unavailable"):
        require_isaaclab()


def test_task_cli_is_importable_without_simulator(capsys: pytest.CaptureFixture[str]) -> None:
    list_tasks()

    assert capsys.readouterr().out == (
        "IsaacLab-Velocity-Flat-MicroDuck\n"
        "IsaacLab-Velocity-Flat-MicroDuck-Adapted\n"
    )


def test_task_registration_is_lazy() -> None:
    with pytest.raises(RuntimeError, match="IsaacLab is unavailable"):
        register_tasks()
