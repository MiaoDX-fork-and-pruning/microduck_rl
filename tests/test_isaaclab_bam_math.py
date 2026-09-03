from __future__ import annotations

import torch

from isaaclab_microduck.actuators.bam_math import (
    XL330_M6,
    friction_budget,
    voltage_torque,
)


def test_voltage_torque_matches_reference_bam() -> None:
    from bam.model import load_model
    from bam.actuator import TorchBackend

    reference = load_model(motor_name="xl330", model="m6")
    reference.actuator.kp = 200.0
    reference.actuator.backend = TorchBackend()
    target = torch.tensor([[-0.8, 0.0, 0.8], [0.2, -0.4, 1.1]])
    position = torch.tensor([[-0.2, 0.1, 0.4], [0.0, -0.5, 0.7]])
    velocity = torch.tensor([[0.0, 2.0, -8.0], [1.0, -4.0, 9.0]])
    control = reference.actuator.compute_control(target, position, velocity, 0.005)
    expected_torque = reference.actuator.compute_torque(control, True, position, velocity)
    voltage, actual_torque = voltage_torque(target, position, velocity)

    assert torch.allclose(voltage, control, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_torque, expected_torque, atol=1e-6, rtol=1e-6)


def test_m6_friction_budget_matches_reference_scalar_sweep() -> None:
    from bam.model import load_model

    reference = load_model(motor_name="xl330", model="m6")
    motor = torch.tensor([[-0.2, 0.0, 0.5], [1.1, -0.8, 0.3]])
    external = torch.tensor([[0.1, -0.4, 0.9], [-0.7, 0.0, 0.2]])
    velocity = torch.tensor([[0.0, 0.4, 3.0], [1.0, -2.0, 8.0]])
    actual = friction_budget(motor, external, velocity)
    expected = torch.tensor(
        [[reference.compute_frictions(float(m), float(e), float(v))[0] for m, e, v in zip(motor[0], external[0], velocity[0])],
         [reference.compute_frictions(float(m), float(e), float(v))[0] for m, e, v in zip(motor[1], external[1], velocity[1])]]
    , dtype=actual.dtype)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_friction_scale_is_multiplicative_and_broadcastable() -> None:
    motor = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    external = torch.zeros_like(motor)
    velocity = torch.ones_like(motor)
    base = friction_budget(motor, external, velocity)
    scaled = friction_budget(motor, external, velocity, friction_scale=torch.tensor([[1.0], [1.5]]))
    assert torch.allclose(scaled[0], base[0])
    assert torch.allclose(scaled[1], base[1] * 1.5)


def test_reference_fixture_parameters_are_explicit() -> None:
    assert XL330_M6.kt > 0
    assert XL330_M6.resistance > 0
    assert XL330_M6.kp_fw == 200.0
    assert XL330_M6.max_current == 1.75
