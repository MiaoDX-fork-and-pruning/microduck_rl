from __future__ import annotations

from pathlib import Path

import torch

from isaaclab_microduck.tasks.parity import (
    ControlStepDelay,
    clip_policy_action,
    effective_supply_voltage,
    force_turn_in_place,
    gaussian_tracking,
    policy_action_to_target,
    sample_uniform_with_zero,
)


def test_action_clip_and_home_transform_are_mjlab_semantics() -> None:
    action = torch.tensor([[-2.0, -0.25, 0.5, 2.0]])
    home = torch.ones(1, 4)
    assert torch.equal(clip_policy_action(action), torch.tensor([[-1.0, -0.25, 0.5, 1.0]]))
    assert torch.equal(policy_action_to_target(action, home), torch.tensor([[0.0, 0.75, 1.5, 2.0]]))


def test_bam_config_keeps_reference_delay_sag_contract() -> None:
    source = (Path(__file__).parents[1] / "src/isaaclab_microduck/actuators/bam_actuator.py").read_text()
    assert "delay_min_lag: int = 3" in source
    assert "delay_max_lag: int = 6" in source
    assert "vin_drop_gain_range: tuple[float, float] = (0.0, 0.2)" in source
    assert "vin_min: float = 6.0" in source


def test_bam_reset_keeps_startup_voltage_samples() -> None:
    """mjlab samples vin/drop gain once at actuator initialization."""
    source = (Path(__file__).parents[1] / "src/isaaclab_microduck/actuators/bam_actuator.py").read_text()
    reset_body = source.split("    def reset(", 1)[1].split("    def compute(", 1)[0]
    assert "uniform_" not in reset_body
    assert "_applied_effort[ids] = 0.0" in reset_body
    assert "self._delay.reset(ids, self._last_target[ids])" in reset_body


def test_bam_delay_initialization_is_per_environment() -> None:
    source = (Path(__file__).parents[1] / "src/isaaclab_microduck/actuators/bam_actuator.py").read_text()
    assert "torch.zeros(\n            (self._num_envs,), dtype=torch.bool" in source
    assert "uninitialized = torch.nonzero(~self._delay_initialized" in source


def test_bam_delay_is_bounded_and_resettable() -> None:
    delay = ControlStepDelay(1, 1, min_lag=3, max_lag=6)
    delay.set_delays(torch.tensor([3]))
    out = [float(delay.push(torch.tensor([[float(i)]]))[0, 0]) for i in range(8)]
    assert out[0] == 0.0
    assert out[3] == 0.0
    assert out[4] == 1.0
    delay.reset(torch.tensor([0]), torch.tensor([[9.0]]))
    assert float(delay.push(torch.tensor([[10.0]]))[0, 0]) == 9.0


def test_bam_delay_reset_accepts_compact_nonzero_env_subset() -> None:
    delay = ControlStepDelay(4, 1, min_lag=3, max_lag=6)
    delay.set_delays(torch.tensor([3, 3, 3, 3]))
    delay.reset(torch.tensor([3]), torch.tensor([[9.0]]))
    target = torch.zeros(4, 1)
    target[3] = 10.0
    out = delay.push(target)
    assert float(out[3, 0]) == 9.0


def test_voltage_sag_has_floor_and_depends_on_previous_load() -> None:
    nominal = torch.tensor([[7.5], [7.5]])
    effort = torch.tensor([[1.0, 2.0], [20.0, 20.0]])
    out = effective_supply_voltage(nominal, effort, 0.2, minimum=6.0)
    assert torch.allclose(out, torch.tensor([[6.9], [6.0]]))


def test_seeded_turn_bucket_is_reproducible_and_zeroes_linear() -> None:
    command = torch.ones(32, 3)
    a = force_turn_in_place(command, 1.0, generator=torch.Generator().manual_seed(5))
    b = force_turn_in_place(command, 1.0, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    assert torch.all(a[:, :2] == 0)
    assert torch.all(a[:, 2].abs() >= 0.4)


def test_zero_bucket_and_reward_kernel() -> None:
    out = sample_uniform_with_zero((128, 3), -1.0, 1.0, zero_probability=1.0,
                                   generator=torch.Generator().manual_seed(2))
    assert torch.count_nonzero(out) == 0
    assert torch.allclose(gaussian_tracking(torch.zeros(4, 3), 0.5), torch.ones(4))
