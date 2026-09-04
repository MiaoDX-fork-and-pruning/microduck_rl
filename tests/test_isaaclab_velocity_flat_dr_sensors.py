from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

from isaaclab_microduck.tasks.velocity_flat_dr import (
    randomize_mass_inertia,
    sample_uniform,
)
from isaaclab_microduck.tasks.velocity_flat_sensors import (
    misaligned_imu,
    reset_imu_mounting,
    reset_actor_sensor_state,
)


def test_velocity_flat_wires_reset_and_dr_events() -> None:
    source = (Path(__file__).parents[1] / "src/isaaclab_microduck/tasks/velocity_flat.py").read_text()
    for name in (
        "reset_velocity_flat_state",
        "randomize_com_offsets",
        "randomize_mass_inertia",
        "randomize_armature",
        "push_velocity",
        "randomize_foot_material",
        "reset_actor_sensor_state_event",
    ):
        assert name in source


def test_seeded_uniform_reference_is_repeatable() -> None:
    torch.manual_seed(23)
    first = sample_uniform((16, 3), -0.003, 0.003)
    torch.manual_seed(23)
    second = sample_uniform((16, 3), -0.003, 0.003)
    assert torch.equal(first, second)
    assert bool((first >= -0.003).all())
    assert bool((first <= 0.003).all())


def test_mass_inertia_scale_is_coupled_and_restored_before_new_sample() -> None:
    class Asset:
        num_bodies = 1
        device = torch.device("cpu")

        def __init__(self):
            self.data = SimpleNamespace(
                body_mass=torch.ones(2, 1),
                body_inertia=torch.ones(2, 1, 9),
            )
            self.mass_writes = []
            self.inertia_writes = []

        def set_masses_index(self, **kwargs):
            self.mass_writes.append(kwargs["masses"].clone())

        def set_inertias_index(self, **kwargs):
            self.inertia_writes.append(kwargs["inertias"].clone())

    class Env:
        num_envs = 2
        device = torch.device("cpu")

        def __init__(self):
            self.scene = {"robot": Asset()}

    env = Env()
    cfg = SimpleNamespace(name="robot", body_ids=[0])
    torch.manual_seed(1)
    randomize_mass_inertia(env, torch.tensor([0, 1]), (0.0, 0.0), cfg)
    assert torch.allclose(env.scene["robot"].mass_writes[-1], torch.ones(2, 1))
    assert torch.allclose(env.scene["robot"].inertia_writes[-1], torch.ones(2, 1, 9))
    # The second call must use cached defaults, not the previous write.
    torch.manual_seed(2)
    randomize_mass_inertia(env, torch.tensor([0, 1]), (0.0, 0.0), cfg)
    assert torch.allclose(env.scene["robot"].mass_writes[-1], torch.ones(2, 1))
    assert torch.allclose(env.scene["robot"].inertia_writes[-1], torch.ones(2, 1, 9))


def test_imu_mounting_is_episode_stable_and_resettable() -> None:
    env = SimpleNamespace(num_envs=4, device=torch.device("cpu"))
    torch.manual_seed(7)
    reset_imu_mounting(env, torch.tensor([0, 1, 2, 3]), max_angle_deg=6.0)
    q = env._imu_mount_quat.clone()
    vector = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
    observed = misaligned_imu(vector, env)
    assert torch.isfinite(observed).all()
    assert torch.allclose(q, env._imu_mount_quat)
    assert bool((observed[:, 2] > 0.98).all())
    torch.manual_seed(7)
    reset_imu_mounting(env, torch.tensor([0, 1, 2, 3]), max_angle_deg=6.0)
    assert torch.equal(q, env._imu_mount_quat)


def test_sensor_state_created_in_inference_mode_resets_in_normal_mode() -> None:
    class Env:
        num_envs = 2
        device = torch.device("cpu")

    env = Env()
    env._gyro_history = torch.zeros(2, 2, 3)
    env._encoder_bias = torch.zeros(2, 14)
    with torch.inference_mode():
        env._gyro_history = torch.ones(2, 2, 3)
        env._encoder_bias = torch.ones(2, 14)
    reset_actor_sensor_state(env, torch.tensor([1]))
    assert torch.all(env._gyro_history[:, 1] == 0)
    assert torch.all(env._gyro_history[:, 0] == 1)
    assert torch.all(env._encoder_bias[1].abs() <= 0.015)
