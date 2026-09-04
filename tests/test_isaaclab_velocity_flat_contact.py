from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from isaaclab_microduck.tasks import velocity_flat_contact as c


def test_velocity_flat_config_has_no_contact_zero_placeholders():
    source = (Path(__file__).parents[1] / "src/isaaclab_microduck/tasks/velocity_flat.py").read_text()
    assert "func=zero_reward" not in source
    assert "func=contact_mdp.feet_clearance" in source
    assert "func=contact_mdp.feet_swing_height" in source
    assert "func=contact_mdp.feet_slip" in source
    assert "func=contact_mdp.self_collision_cost" in source
    assert "func=contact_mdp.foot_contact_forces" in source


class _Proxy:
    def __init__(self, value):
        self.torch = value


class _Sensor:
    def __init__(self, force, air, first):
        self.data = SimpleNamespace(
            net_forces_w=_Proxy(force),
            current_air_time=_Proxy(air),
            last_air_time=_Proxy(air),
        )
        self._first = first

    def compute_first_contact(self, _dt):
        return _Proxy(self._first)


class _Scene:
    def __init__(self, sensors, robot):
        self.sensors = sensors
        self.env_origins = torch.zeros(1, 3)
        self._robot = robot

    def __getitem__(self, name):
        if name == "robot":
            return self._robot
        raise KeyError(name)


class _Env:
    def __init__(self):
        force = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]])
        air = torch.tensor([[0.2, 0.05]])
        first = torch.tensor([[True, False]])
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.step_dt = 0.02
        self.episode_length_buf = torch.tensor([10])
        self.command_manager = SimpleNamespace(get_command=lambda _: torch.tensor([[0.2, 0.0, 0.0]]))
        sensors = {
            "feet_ground_contact": _Sensor(force, air, first),
            "self_collision": _Sensor(force, air, first),
        }
        self.scene = _Scene(sensors, SimpleNamespace(
                data=SimpleNamespace(
                    body_pos_w=_Proxy(torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.08]]])),
                    body_lin_vel_w=_Proxy(torch.tensor([[[0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]])),
                    joint_pos=_Proxy(torch.zeros(1, 2)),
                    joint_vel=_Proxy(torch.zeros(1, 2)),
                    root_pos_w=_Proxy(torch.tensor([[0.0, 0.0, 0.1]])),
                    root_quat_w=_Proxy(torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
                )
            ))


def _feet_cfg(name="feet_ground_contact"):
    return c.SceneEntityCfg(name, body_names=["left", "right"], body_ids=[0, 1], preserve_order=True)


def _robot_cfg():
    return c.SceneEntityCfg("robot", body_names=["left", "right"], body_ids=[0, 1], preserve_order=True)


def test_force_observation_is_log_compressed_and_nonzero():
    env = _Env()
    out = c.foot_contact_forces(env, _feet_cfg())
    assert out.shape == (1, 6)
    assert torch.allclose(out[0, 2], torch.log1p(torch.tensor(2.0)))
    assert float(out.abs().sum()) > 0.0


def test_air_time_and_clearance_use_sensor_state_and_command_gate():
    env = _Env()
    assert torch.equal(c.foot_air_time(env, _feet_cfg()), torch.tensor([[0.2, 0.05]]))
    reward = c.feet_air_time(env, _feet_cfg(), threshold_min=0.1, threshold_max=0.3, command_name="cmd")
    assert torch.equal(reward, torch.tensor([1.0]))
    cost = c.feet_clearance(env, 0.02, "cmd", asset_cfg=_robot_cfg())
    assert torch.allclose(cost, torch.tensor([0.022]))


def test_slip_self_collision_nan_and_flat_bounds_are_real_terms():
    env = _Env()
    slip = c.feet_slip(env, _feet_cfg(), "cmd", asset_cfg=_robot_cfg())
    assert torch.allclose(slip, torch.tensor([0.05]))
    assert torch.equal(c.self_collision_cost(env, _feet_cfg(), force_threshold=3.0), torch.tensor([1.0]))
    env.scene.sensors["feet_ground_contact"].data.net_forces_w = _Proxy(torch.tensor([[[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]]))
    assert bool(c.nan_state(env, sensor_names=("feet_ground_contact",))[0])
    assert not bool(c.terrain_out_of_bounds(env)[0])
