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
from isaaclab_microduck.tasks.parity import ControlStepDelay, effective_supply_voltage

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
    # BAM firmware command latency follows mjlab's DelayBuffer: with the
    # default update period of zero, a per-environment lag is sampled on each
    # actuator command application. These are simulator callback steps; the
    # task must call this actuator at the same cadence as the reference path.
    delay_min_lag: int = 3
    delay_max_lag: int = 6
    delay_hold_prob: float = 0.0
    delay_update_period: int = 0
    delay_per_env_phase: bool = True
    vin_drop_gain_range: tuple[float, float] = (0.0, 0.2)
    vin_min: float = 6.0
    action_clip: float = 1.0


class BamActuator(ActuatorBase):
    """Vectorized BAM voltage control with explicit effort output."""

    cfg: BamActuatorCfg

    def __init__(self, cfg: BamActuatorCfg, *args, **kwargs):
        # IsaacLab 3.0 resolves joint stiffness/damping in the articulation
        # control layer and no longer accepts them in ActuatorBase.__init__.
        # Keep accepting the resolved kwargs so the same wrapper remains
        # constructible from the actuator collection.
        kwargs.pop("stiffness", None)
        kwargs.pop("damping", None)
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
        self._vin_drop_gain = torch.empty(
            (self._num_envs, 1), dtype=torch.float32, device=self._device
        ).uniform_(*cfg.vin_drop_gain_range)
        resolved_num_joints = getattr(self, "_num_joints", None)
        if resolved_num_joints is None:
            indices = getattr(self, "joint_indices", ())
            resolved_num_joints = len(indices) if not isinstance(indices, slice) else len(self.joint_names)
        num_joints = int(resolved_num_joints)
        if num_joints <= 0:
            raise RuntimeError("BamActuator could not resolve its joint count")
        self._delay = ControlStepDelay(
            self._num_envs,
            num_joints,
            min_lag=cfg.delay_min_lag,
            max_lag=cfg.delay_max_lag,
            device=self._device,
            hold_prob=cfg.delay_hold_prob,
            update_period=cfg.delay_update_period,
            per_env_phase=cfg.delay_per_env_phase,
            sample_lag_each_push=True,
        )
        self._delay_initialized = torch.zeros(
            (self._num_envs,), dtype=torch.bool, device=self._device
        )
        # Keep the most recent absolute target per environment.  mjlab's
        # delayed actuator history is reset to the episode's initial command,
        # never to zero; retaining this value lets a partial IsaacLab reset
        # seed its FIFO without a transient HOME-to-zero command.
        self._last_target = torch.zeros(
            (self._num_envs, num_joints), dtype=torch.float32, device=self._device
        )
        # Keep the motor-only effort separately from the final articulation
        # effort.  mjlab's voltage sag uses the previous BAM motor torque;
        # feedforward effort is an external command and must not contribute to
        # the battery-load estimate.
        self._previous_motor_effort = torch.zeros_like(self._last_target)
        self._applied_effort = torch.zeros_like(self._last_target)
        # Runtime parity probes read this diagnostic to distinguish the raw
        # absolute target from the target visible after BAM's control-step
        # FIFO.  It is observational only and never participates in effort
        # computation.
        self._delayed_target = torch.zeros_like(self._last_target)
        self._friction_scale = torch.full(
            (self._num_envs, 1), cfg.friction_scale, dtype=torch.float32, device=self._device
        )

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        # vin and drop gain are startup-randomized in mjlab and remain stable
        # across episode resets.  Only dynamic controller state is reset here.
        if env_ids is None:
            ids = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self._num_envs, device=self._device, dtype=torch.long)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        if ids.numel() == 0:
            return
        # mjlab's DelayBuffer reset clears the episode rows; the next command
        # append backfills those rows with the first command of the episode.
        self._delay.reset(ids)
        self._previous_motor_effort[ids] = 0.0
        self._applied_effort[ids] = 0.0
        self._delayed_target[ids] = self._last_target[ids]
        self.applied_effort = self._applied_effort
        self._delay_initialized[ids] = False

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("BamActuator requires joint position targets")
        # The RSL-RL/VecEnv boundary clips raw policy actions before the
        # HOME offset is added.  At this point IsaacLab passes absolute joint
        # targets, so clipping them again would incorrectly clip HOME itself.
        target_command = control_action.joint_positions
        uninitialized = torch.nonzero(~self._delay_initialized, as_tuple=False).flatten()
        if uninitialized.numel():
            # Seed each reset environment from its first absolute command. This
            # preserves the actuator's lag without a spurious zero/HOME jump.
            self._delay.reset(uninitialized, target_command[uninitialized])
            self._last_target[uninitialized] = target_command[uninitialized].detach()
            self._delay_initialized[uninitialized] = True
        target = self._delay.push(target_command)
        self._delayed_target.copy_(target.detach())
        self._last_target.copy_(target_command.detach())
        # Effective voltage uses the previous solved motor effort.  PhysX does
        # not expose the same-step external load to this callback; that
        # limitation is recorded as BACKEND_DELTA in the parity ledger.
        previous = self._previous_motor_effort
        supply = effective_supply_voltage(
            self._supply_voltage,
            previous,
            self._vin_drop_gain,
            minimum=self.cfg.vin_min,
        )
        voltage, motor_effort = voltage_torque(
            target,
            joint_pos,
            # BAM's position controller consumes measured dq for back-EMF;
            # velocity targets are not part of its firmware command.
            joint_vel,
            params=self._params,
            vin=supply,
        )
        del voltage  # retained by the math bench; PhysX consumes effort here
        feedforward = control_action.joint_efforts
        if feedforward is None:
            feedforward = torch.zeros_like(motor_effort)
        # Keep actuator-owned state as ordinary writable tensors.  Policy
        # replay is commonly wrapped in ``torch.inference_mode``; rebinding
        # this buffer to the computed inference tensor would make the next
        # episode reset fail when it clears the previous effort in-place.
        with torch.inference_mode(False):
            self._previous_motor_effort.copy_(motor_effort.detach())
            self._applied_effort.copy_(motor_effort + feedforward)
        self.computed_effort = self._applied_effort
        self.applied_effort = self._applied_effort
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

    def physx_friction_coefficients(
        self,
        motor_effort: torch.Tensor,
        external_effort: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map the BAM budget to IsaacLab's static/dynamic/viscous fields."""

        budget = self.friction_budget(motor_effort, external_effort, joint_vel)
        viscous = torch.full_like(budget, self._params.friction_viscous)
        return budget, budget, viscous

    def set_supply_voltage(self, voltage: float | torch.Tensor) -> None:
        """Set the episode voltage used by the dynamic bench or a controlled run."""

        value = torch.as_tensor(voltage, dtype=self._supply_voltage.dtype, device=self._device)
        self._supply_voltage.fill_(float(value))

    def set_friction_scale(
        self,
        scale: float | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set friction multipliers globally or for a reset/DR env subset."""

        value = torch.as_tensor(scale, dtype=self._friction_scale.dtype, device=self._device)
        if env_ids is None:
            if value.numel() == 1:
                self._friction_scale.fill_(float(value))
            elif value.shape == self._friction_scale.shape:
                self._friction_scale.copy_(value)
            else:
                raise ValueError("global friction scale must be scalar or (num_envs, 1)")
            return
        if isinstance(env_ids, slice):
            ids = torch.arange(self._num_envs, device=self._device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        if value.numel() == 1:
            self._friction_scale[ids] = value
        elif value.shape == (ids.numel(), 1):
            self._friction_scale[ids] = value
        else:
            raise ValueError("subset friction scale must be scalar or (len(env_ids), 1)")
