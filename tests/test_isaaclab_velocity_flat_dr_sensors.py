from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

from isaaclab_microduck.tasks.velocity_flat_dr import (
    randomize_mass_inertia,
    reset_velocity_flat_state,
    sample_uniform,
)
from isaaclab_microduck.tasks.velocity_flat_sensors import (
    misaligned_imu,
    reset_imu_mounting,
    reset_actor_sensor_state,
    sensor_corruption,
)
from isaaclab_microduck.tasks.parity import observation_noise


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


def test_reset_matches_mjlab_root_randomization_and_exact_home_joints() -> None:
    class Asset:
        device = torch.device("cpu")
        num_joints = 3

        def __init__(self):
            self.data = SimpleNamespace(
                default_root_pose=torch.tensor(
                    [[0.2, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0],
                     [0.3, 0.1, 0.5, 0.0, 0.0, 0.0, 1.0]]
                ),
                default_root_vel=torch.zeros(2, 6),
                default_joint_pos=torch.tensor([[0.2, -0.4, 0.7], [0.2, -0.4, 0.7]]),
                default_joint_vel=torch.zeros(2, 3),
            )
            self.root_writes = []
            self.joint_writes = []

        def write_root_pose_to_sim_index(self, **kwargs):
            self.root_writes.append(kwargs["root_pose"].clone())

        def write_root_velocity_to_sim_index(self, **kwargs):
            pass

        def write_joint_position_to_sim_index(self, **kwargs):
            self.joint_writes.append(kwargs["position"].clone())

        def write_joint_velocity_to_sim_index(self, **kwargs):
            pass

    class Scene(dict):
        env_origins = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    class Env:
        num_envs = 2
        device = torch.device("cpu")

        def __init__(self):
            self.scene = Scene(robot=Asset())

    env = Env()
    cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0, 1, 2]))
    reset_velocity_flat_state(
        env,
        torch.tensor([0, 1]),
        z_range=(0.12, 0.12),
        xy_range=(0.0, 0.0),
        yaw_range=(0.0, 0.0),
        asset_cfg=cfg,
    )
    roots = env.scene["robot"].root_writes[-1]
    assert torch.allclose(roots[:, :3], torch.tensor([[1.2, 1.9, 3.12], [4.3, 5.1, 6.12]]))
    assert torch.allclose(roots[:, 3:], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1))
    # mjlab position_range=(0, 0): all actuated joints return exactly to HOME.
    assert torch.equal(env.scene["robot"].joint_writes[-1], env.scene["robot"].data.default_joint_pos)
    # The second call must use cached defaults, not the previous write.


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


def test_imu_mounting_subset_reset_preserves_other_envs() -> None:
    env = SimpleNamespace(num_envs=4, device=torch.device("cpu"))
    torch.manual_seed(4)
    reset_imu_mounting(env, None, max_angle_deg=6.0)
    before = env._imu_mount_quat.clone()
    torch.manual_seed(99)
    reset_actor_sensor_state(env, torch.tensor([1, 3]))
    assert torch.equal(before[[0, 2]], env._imu_mount_quat[[0, 2]])
    assert torch.equal(before[[1, 3]], env._imu_mount_quat[[1, 3]])


def test_periodic_sensor_delay_is_control_step_based_and_bounded() -> None:
    env = SimpleNamespace(num_envs=1, device=torch.device("cpu"))
    outputs = []
    for step in range(66):
        value = torch.tensor([[float(step)]])
        outputs.append(
            sensor_corruption(
                env,
                "gyro",
                value,
                noise=0.0,
                delay=1,
                delay_update_period=64,
            ).item()
        )
    assert all(torch.isfinite(torch.tensor(outputs)))
    assert outputs[0] in (0.0, 1.0)
    # No lag resampling occurs merely because a PhysX substep would run; the
    # only state transition is the explicit control-step call above.
    assert env._gyro_delay_step.item() == 66
    assert 0 <= env._gyro_lag.item() <= 1


def test_sensor_delay_subset_reset_does_not_rewind_other_envs() -> None:
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))
    for step in range(4):
        value = torch.full((2, 1), float(step))
        sensor_corruption(env, "gyro", value, noise=0.0, delay=1, delay_update_period=64)
    before = env._gyro_delay_step.clone()
    reset_actor_sensor_state(env, torch.tensor([0]))
    assert env._gyro_delay_step[0].item() == 0
    assert env._gyro_delay_step[1].item() == before[1].item()


def test_joint_position_actor_noise_matches_mjlab_bound() -> None:
    value = torch.zeros(128, 14)
    torch.manual_seed(123)
    noisy = observation_noise(value, 0.001)
    delta = noisy - value
    assert bool((delta >= -0.001).all())
    assert bool((delta <= 0.001).all())
    assert bool((delta != 0).any())


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
