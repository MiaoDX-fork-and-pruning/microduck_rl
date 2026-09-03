"""IsaacLab explicit actuator wrapper for Microduck's BAM XL330 model.

The wrapper owns the voltage-controller part of BAM.  IsaacLab's actuator
callback receives joint state but not the solved external joint torque, so the
load-dependent friction budget is exposed for the dynamic bench and is not
silently folded into the returned motor effort.  A PhysX-side friction bridge
is required before claiming full BAM simulator parity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

from .bam_math import XL330_M6, BamParameters, friction_budget, voltage_torque

@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the explicit Microduck BAM actuator."""

    class_type: type["BamActuator"] | str = "isaaclab_microduck.actuators.bam_actuator:BamActuator"
    kp_fw: float = XL330_M6.kp_fw
    vin: float = XL330_M6.vin
    max_pwm: float = XL330_M6.max_pwm
    max_current: float = XL330_M6.max_current
    vin_range: tuple[float, float] | None = None
    friction_scale: float = 1.0
    stiffness: float = 0.0
    damping: float = 0.0
    effort_limit: float = 1.0e9
    effort_limit_sim: float = 1.0e9
    velocity_limit: float = 100.0
    velocity_limit_sim: float = 100.0


class BamActuator(ActuatorBase):
    """Vectorized BAM voltage control with explicit effort output."""

    cfg: BamActuatorCfg

    def __init__(self, cfg: BamActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._params: BamParameters = replace(
            XL330_M6,
            kp_fw=cfg.kp_fw,
            vin=cfg.vin,
            max_pwm=cfg.max_pwm,
            max_current=cfg.max_current,
        )
        self._supply_voltage = torch.full(
            (self._num_envs, 1), cfg.vin, dtype=torch.float32, device=self._device
        )
        if cfg.vin_range is not None:
            self._supply_voltage.uniform_(*cfg.vin_range)
        self._friction_scale = torch.full(
            (self._num_envs, 1), cfg.friction_scale, dtype=torch.float32, device=self._device
        )

    def reset(self, env_ids: Sequence[int]):
        # Voltage and friction scales are episode-stable DR values.
        del env_ids

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("BamActuator requires joint position targets")
        voltage, motor_effort = voltage_torque(
            control_action.joint_positions,
            joint_pos,
            # BAM's position controller consumes measured dq for back-EMF;
            # velocity targets are not part of its firmware command.
            joint_vel,
            params=self._params,
            vin=self._supply_voltage,
        )
        del voltage  # retained by the math bench; PhysX consumes effort here
        feedforward = control_action.joint_efforts
        if feedforward is None:
            feedforward = torch.zeros_like(motor_effort)
        self.computed_effort = motor_effort + feedforward
        self.applied_effort = self.computed_effort
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def friction_budget(self, motor_effort: torch.Tensor, external_effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """Return the M6 budget for a measured/bench-provided external load."""

        return friction_budget(
            motor_effort,
            external_effort,
            joint_vel,
            params=self._params,
            friction_scale=self._friction_scale,
        )
