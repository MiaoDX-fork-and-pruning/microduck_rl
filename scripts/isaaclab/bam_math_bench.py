"""Run the simulator-independent BAM numerical parity grid.

The report is deliberately JSON so it can be archived beside later dynamic
bench results.  It compares the Torch port with the reference BAM package over
position error, velocity, supply voltage and friction scale.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from bam.model import load_model
from bam.actuator import TorchBackend

from isaaclab_microduck.actuators.bam_math import XL330_M6, friction_budget, voltage_torque


def run_bench() -> dict[str, float | int]:
    reference = load_model(motor_name="xl330", model="m6")
    reference.actuator.backend = TorchBackend()
    reference.actuator.kp = XL330_M6.kp_fw
    errors_torque: list[float] = []
    errors_voltage: list[float] = []
    errors_friction: list[float] = []
    for error, velocity, supply, scale in itertools.product(
        (-1.0, -0.25, 0.0, 0.25, 1.0),
        (-10.0, -2.0, 0.0, 2.0, 10.0),
        (6.5, 7.5, 8.2),
        (0.7, 1.0, 1.3),
    ):
        reference.actuator.vin = supply
        target = torch.tensor([[error]], dtype=torch.float64)
        position = torch.zeros_like(target)
        speed = torch.tensor([[velocity]], dtype=torch.float64)
        ref_voltage = reference.actuator.compute_control(target, position, speed, 0.005)
        ref_torque = reference.actuator.compute_torque(ref_voltage, True, position, speed)
        voltage, torque = voltage_torque(target, position, speed, params=XL330_M6, vin=supply)
        errors_voltage.append(abs(float(voltage - ref_voltage)))
        errors_torque.append(abs(float(torque - ref_torque)))

        motor = torch.tensor([[float(ref_torque)]], dtype=torch.float64)
        external = torch.tensor([[0.35]], dtype=torch.float64)
        ref_friction = reference.compute_frictions(float(motor), float(external), velocity)[0]
        actual_friction = friction_budget(motor, external, speed, params=XL330_M6, friction_scale=scale)
        # Reference model has no friction-scale knob; compare the nominal slice
        # and report scaled slices separately as an IsaacLab-only operation.
        if scale == 1.0:
            errors_friction.append(abs(float(actual_friction) - ref_friction))

    return {
        "samples": len(errors_torque),
        "voltage_max_abs_error": max(errors_voltage),
        "voltage_mean_abs_error": sum(errors_voltage) / len(errors_voltage),
        "torque_max_abs_error": max(errors_torque),
        "torque_mean_abs_error": sum(errors_torque) / len(errors_torque),
        "friction_nominal_max_abs_error": max(errors_friction),
        "friction_nominal_mean_abs_error": sum(errors_friction) / len(errors_friction),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run_bench(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
