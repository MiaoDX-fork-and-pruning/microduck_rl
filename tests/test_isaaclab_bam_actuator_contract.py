from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "src/isaaclab_microduck/actuators/bam_actuator.py"


def test_bam_actuator_declares_explicit_isaaclab_extension() -> None:
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "ActuatorBase" in names
    assert "ArticulationActions" in names
    assert "voltage_torque" in names
    assert "friction_budget" in names
    assert "class_type" in source


def test_bam_actuator_does_not_hide_missing_physx_friction_bridge() -> None:
    source = SCRIPT.read_text()
    assert "silently" in source
    assert "PhysX-side friction bridge" in source
    assert "def friction_budget" in source


def test_actuator_package_keeps_math_importable_without_isaaclab() -> None:
    package = (SCRIPT.parent / "__init__.py").read_text()
    assert "def __getattr__" in package
    assert "from .bam_actuator import" in package


def test_bam_wrapper_uses_measured_velocity_for_back_emf() -> None:
    source = SCRIPT.read_text()
    assert "# BAM's position controller" in source
    assert "            joint_vel,\n            params=self._params" in source
    assert "joint_vel - target_vel" not in source


def test_bam_voltage_sag_uses_previous_motor_effort_not_feedforward() -> None:
    source = SCRIPT.read_text()
    # mjlab's ``_prev_motor_torque`` is motor-only.  Keeping a separate
    # state buffer prevents ArticulationActions.joint_efforts (feedforward)
    # from changing the next-step battery sag.
    assert "self._previous_motor_effort = torch.zeros_like(self._last_target)" in source
    assert "previous = self._previous_motor_effort" in source
    assert "self._previous_motor_effort.copy_(motor_effort.detach())" in source
    assert "self._previous_motor_effort[ids] = 0.0" in source


def test_bam_friction_scale_supports_per_environment_domain_randomization() -> None:
    source = SCRIPT.read_text()
    assert "env_ids: Sequence[int] | torch.Tensor | slice | None = None" in source
    assert "subset friction scale must be scalar" in source
