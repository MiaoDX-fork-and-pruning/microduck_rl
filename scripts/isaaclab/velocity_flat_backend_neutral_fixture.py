"""Compare the IsaacLab actuator primitives with the mjlab reference path.

This fixture deliberately avoids an articulation and a policy checkpoint.  It
holds the canonical HOME, fixed raw actions, joint state, reset subset, and RNG
seed constant while comparing target delay, voltage sag, and BAM motor torque.
It is the bounded proof used to classify a trained-policy directional failure
before changing rewards, PPO, or simulator settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from bam.actuator import TorchBackend
from bam.model import load_model
from mjlab.utils.buffers import DelayBuffer

from isaaclab_microduck.actuators.bam_math import voltage_torque
from isaaclab_microduck.policy_abi import HOME_POSITION
from isaaclab_microduck.tasks.parity import ControlStepDelay, effective_supply_voltage, policy_action_to_target


def run_fixture(*, seed: int = 2026, num_envs: int = 4, steps: int = 10) -> dict:
    if num_envs < 2:
        raise ValueError("num_envs must be at least 2 for a subset-reset proof")
    if steps < 2:
        raise ValueError("steps must be at least 2")

    home = torch.as_tensor(HOME_POSITION, dtype=torch.float32).repeat(num_envs, 1)
    actions = torch.stack(
        [
            torch.sin(torch.arange(14, dtype=torch.float32) * 0.17 + step * 0.31)
            for step in range(steps)
        ],
        dim=0,
    )
    actions = actions.unsqueeze(1).expand(steps, num_envs, 14).contiguous()
    # Give each environment a distinct but fixed joint state so broadcast and
    # per-environment voltage sag are both exercised.
    position = home + torch.linspace(-0.08, 0.08, num_envs, dtype=torch.float32).reshape(-1, 1)
    velocity = torch.linspace(-2.0, 2.0, num_envs, dtype=torch.float32).reshape(-1, 1).expand(num_envs, 14).clone()
    nominal_voltage = torch.linspace(6.8, 8.0, num_envs, dtype=torch.float32).reshape(-1, 1)
    drop_gain = torch.linspace(0.05, 0.2, num_envs, dtype=torch.float32).reshape(-1, 1)

    isaac_rng = torch.Generator().manual_seed(seed)
    mjlab_rng = torch.Generator().manual_seed(seed)
    isaac_delay = ControlStepDelay(
        num_envs,
        14,
        min_lag=3,
        max_lag=6,
        generator=isaac_rng,
        sample_lag_each_push=True,
    )
    mjlab_delay = DelayBuffer(
        min_lag=3,
        max_lag=6,
        batch_size=num_envs,
        device="cpu",
        generator=mjlab_rng,
    )

    reference = load_model(motor_name="xl330", model="m6")
    reference.actuator.backend = TorchBackend()
    reference.actuator.kp = 200.0
    previous_isaac = torch.zeros(num_envs, 14, dtype=torch.float64)
    previous_mjlab = torch.zeros_like(previous_isaac)
    rows: list[dict] = []
    max_target_error = 0.0
    max_torque_error = 0.0
    max_voltage_error = 0.0
    for step in range(steps):
        if step == steps // 2:
            reset_ids = torch.tensor([1], dtype=torch.long)
            isaac_delay.reset(reset_ids)
            mjlab_delay.reset(reset_ids)
        target = policy_action_to_target(actions[step], home, scale=1.0)
        delayed_isaac = isaac_delay.push(target)
        mjlab_delay.append(target)
        delayed_mjlab = mjlab_delay.compute()

        supply_isaac = effective_supply_voltage(nominal_voltage, previous_isaac, drop_gain, minimum=6.0)
        supply_mjlab = effective_supply_voltage(nominal_voltage, previous_mjlab, drop_gain, minimum=6.0)
        _, torque_isaac = voltage_torque(delayed_isaac, position, velocity, vin=supply_isaac)
        reference.actuator.vin = supply_mjlab
        control = reference.actuator.compute_control(delayed_mjlab, position, velocity, 0.005)
        torque_mjlab = reference.actuator.compute_torque(control, True, position, velocity)

        target_error = float((delayed_isaac - delayed_mjlab).abs().max())
        voltage_error = float((supply_isaac - supply_mjlab).abs().max())
        torque_error = float((torque_isaac - torque_mjlab).abs().max())
        max_target_error = max(max_target_error, target_error)
        max_voltage_error = max(max_voltage_error, voltage_error)
        max_torque_error = max(max_torque_error, torque_error)
        rows.append(
            {
                "step": step,
                "reset_env_ids": [1] if step == steps // 2 else [],
                "delay_lag_isaac": isaac_delay.delay.tolist(),
                "delay_lag_mjlab": mjlab_delay.current_lags.tolist(),
                "target_error_max_abs": target_error,
                "supply_voltage_error_max_abs": voltage_error,
                "torque_error_max_abs": torque_error,
                "finite": bool(torch.isfinite(delayed_isaac).all() and torch.isfinite(torque_isaac).all()),
            }
        )
        previous_isaac = torque_isaac
        previous_mjlab = torque_mjlab

    return {
        "seed": seed,
        "num_envs": num_envs,
        "steps": steps,
        "reset_step": steps // 2,
        "action_contract": "HOME + raw_action (scale=1.0, no clip)",
        "delay_contract": "mjlab DelayBuffer default: lag sampled every push, 3..6",
        "max_target_error": max_target_error,
        "max_voltage_error": max_voltage_error,
        "max_torque_error": max_torque_error,
        "passed": bool(max_target_error <= 1e-6 and max_voltage_error <= 5e-7 and max_torque_error <= 1e-6),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_fixture(seed=args.seed, num_envs=args.num_envs, steps=args.steps)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
