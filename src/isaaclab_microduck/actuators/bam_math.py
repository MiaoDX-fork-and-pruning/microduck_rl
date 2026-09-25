"""Torch implementation of the BAM XL330 M6 numerical model.

The functions in this module have no IsaacLab dependency.  They are the
portable part of the actuator port and are intentionally shaped for batched
``(num_envs, num_joints)`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BamParameters:
    kt: float
    resistance: float
    error_gain: float
    kp_fw: float
    vin: float
    max_pwm: float
    max_current: float
    friction_base: float
    friction_stribeck: float
    load_friction_motor: float
    load_friction_external: float
    load_friction_motor_stribeck: float
    load_friction_external_stribeck: float
    load_friction_motor_quad: float
    load_friction_external_quad: float
    dtheta_stribeck: float
    alpha: float
    friction_viscous: float


# Values are the checked-in BAM xl330/m6 fit used by the mjlab backend.
XL330_M6 = BamParameters(
    kt=0.36601349688984386,
    resistance=2.8113923539223227,
    error_gain=0.0028773775022263564,
    # Microduck overrides the bundled XL330 firmware gain in
    # ``microduck_constants.py``; the bundled BAM JSON default is 400.
    kp_fw=200.0,
    vin=7.5,
    max_pwm=1.0,
    max_current=1.75,
    friction_base=0.004771183165566,
    friction_stribeck=0.004676345799486616,
    load_friction_motor=0.2667860954283698,
    load_friction_external=8.515871897059342e-06,
    load_friction_motor_stribeck=1.0722918395099123e-05,
    load_friction_external_stribeck=0.08077928978935671,
    load_friction_motor_quad=0.009972471242139415,
    load_friction_external_quad=0.004902565732332559,
    dtheta_stribeck=2.890372094130307,
    alpha=8.683259907618984,
    friction_viscous=0.005359668274599504,
)


def voltage_torque(
    position_target: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    *,
    params: BamParameters = XL330_M6,
    kp_scale: torch.Tensor | float = 1.0,
    vin: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(voltage, motor_torque)`` for the BAM voltage controller.

    This follows ``VoltageControlledActuator.compute_control`` and
    ``compute_torque`` in the reference BAM package, including its attempted
    current limiter and final physical PWM clamp.
    """

    supply = torch.as_tensor(
        params.vin if vin is None else vin,
        dtype=position.dtype,
        device=position.device,
    )
    gain = torch.as_tensor(kp_scale, dtype=position.dtype, device=position.device)
    duty = (position_target - position) * params.kp_fw * gain * params.error_gain
    back_emf = params.kt * velocity
    duty_span = params.resistance * params.max_current / supply
    duty_center = back_emf / supply
    duty = torch.maximum(duty, duty_center - duty_span)
    duty = torch.minimum(duty, duty_center + duty_span)
    duty = torch.clamp(duty, -params.max_pwm, params.max_pwm)
    voltage = supply * duty
    torque = params.kt * voltage / params.resistance
    torque = torque - (params.kt**2) * velocity / params.resistance
    return voltage, torque


def friction_budget(
    motor_torque: torch.Tensor,
    external_torque: torch.Tensor,
    velocity: torch.Tensor,
    *,
    params: BamParameters = XL330_M6,
    friction_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Return the M6 velocity-independent friction budget in N-m."""

    dtype = motor_torque.dtype
    device = motor_torque.device
    scale = torch.as_tensor(friction_scale, dtype=dtype, device=device)
    stribeck = torch.exp(
        -torch.pow(torch.abs(velocity / params.dtheta_stribeck), params.alpha)
    )
    gearbox = torch.abs(
        external_torque * params.load_friction_external
        - motor_torque * params.load_friction_motor
    )
    gearbox_stribeck = torch.abs(
        external_torque * params.load_friction_external_stribeck
        - motor_torque * params.load_friction_motor_stribeck
    )
    drive_side = (torch.abs(motor_torque) > torch.abs(external_torque)).to(dtype)
    quadratic = drive_side * params.load_friction_external_quad * torch.abs(external_torque) ** 2
    quadratic = quadratic + (1.0 - drive_side) * params.load_friction_motor_quad * torch.abs(motor_torque) ** 2
    budget = params.friction_base + stribeck * params.friction_stribeck
    budget = budget + gearbox + stribeck * gearbox_stribeck
    budget = budget + stribeck * quadratic * (torch.sign(external_torque) != torch.sign(motor_torque)).to(dtype)
    return budget * scale


def resisting_torque(
    motor_torque: torch.Tensor,
    external_torque: torch.Tensor,
    velocity: torch.Tensor,
    *,
    params: BamParameters = XL330_M6,
    friction_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Approximate dynamic friction torque for explicit-actuator backends.

    IsaacLab's explicit actuator callback does not expose the solved external
    joint torque.  This helper is therefore only a numerical fallback for
    benches; simulator parity still requires a PhysX-side friction integration.
    """

    budget = friction_budget(
        motor_torque,
        external_torque,
        velocity,
        params=params,
        friction_scale=friction_scale,
    )
    return budget * torch.tanh(velocity / 1.0) + params.friction_viscous * velocity
