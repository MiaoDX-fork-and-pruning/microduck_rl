from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import torch

from isaaclab_microduck.tasks import velocity_flat_contact as c


ROOT = Path(__file__).parents[1]
WALK_MJCF = ROOT / "src/mjlab_microduck/robot/microduck/robot_walk.xml"


def test_velocity_flat_config_has_no_contact_zero_placeholders():
    source = (ROOT / "src/isaaclab_microduck/tasks/velocity_flat.py").read_text()
    assert "func=zero_reward" not in source
    assert "func=contact_mdp.feet_clearance" in source
    assert "func=contact_mdp.feet_swing_height" in source
    assert "func=contact_mdp.feet_slip" in source
    assert "func=contact_mdp.self_collision_cost" in source
    assert "func=contact_mdp.foot_contact_forces" in source
    assert "self_collision_trunk = ContactSensorCfg(" in source
    assert "self_collision_legs = ContactSensorCfg(" in source
    assert source.count("track_contact_points=True") == 2
    assert 'params={"sensor_names": ("self_collision_trunk", "self_collision_legs")}' in source


def test_contact_constants_match_mjlab_walk_mjcf():
    root = ElementTree.parse(WALK_MJCF).getroot()
    sites = {
        site.attrib["name"]: tuple(float(value) for value in site.attrib["pos"].split())
        for site in root.iter("site")
        if site.attrib.get("name") in {"left_foot", "right_foot"}
    }
    assert sites == {
        "left_foot": c.FOOT_SITE_OFFSETS_B[0],
        "right_foot": c.FOOT_SITE_OFFSETS_B[1],
    }

    self_collision_bodies = {
        body.attrib["name"]
        for body in root.iter("body")
        if any(geom.attrib.get("class") == "self_collision_only" for geom in body.findall("geom"))
    }
    assert self_collision_bodies == {"trunk_base", "leg", "leg_2"}


def test_xyzw_quaternion_rotates_foot_site_offset():
    half_sqrt_two = 2.0**-0.5
    quat = torch.tensor([[[0.0, 0.0, half_sqrt_two, half_sqrt_two]]])
    vector = torch.tensor([[[1.0, 0.0, 0.0]]])
    assert torch.allclose(
        c._quat_apply_xyzw(quat, vector),
        torch.tensor([[[0.0, 1.0, 0.0]]]),
        atol=1.0e-6,
    )


class _Proxy:
    def __init__(self, value):
        self.torch = value


class _Sensor:
    def __init__(self, force, air, first, force_matrix=None, contact_counts=None):
        self.data = SimpleNamespace(
            net_forces_w=_Proxy(force),
            current_air_time=_Proxy(air),
            last_air_time=_Proxy(air),
            force_matrix_w=None if force_matrix is None else _Proxy(force_matrix),
        )
        self._first = first
        self._contact_counts = contact_counts

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
        if name in self.sensors:
            return self.sensors[name]
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
            "self_collision_trunk": _Sensor(
                force,
                air,
                first,
                torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]]]),
                torch.tensor([[0, 2]], dtype=torch.int32),
            ),
            "self_collision_legs": _Sensor(
                force,
                air,
                first,
                torch.zeros(1, 1, 1, 3),
                torch.zeros(1, 1, dtype=torch.int32),
            ),
        }
        self.scene = _Scene(sensors, SimpleNamespace(
                data=SimpleNamespace(
                    body_pos_w=_Proxy(torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.08]]])),
                    body_quat_w=_Proxy(
                        torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
                    ),
                    body_lin_vel_w=_Proxy(torch.tensor([[[0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]])),
                    body_ang_vel_w=_Proxy(torch.zeros(1, 2, 3)),
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
    assert torch.allclose(cost, torch.tensor([0.01777444]))


def test_slip_self_collision_nan_and_flat_bounds_are_real_terms():
    env = _Env()
    slip = c.feet_slip(env, _feet_cfg(), "cmd", asset_cfg=_robot_cfg())
    assert torch.allclose(slip, torch.tensor([0.05]))
    assert torch.equal(
        c.self_collision_cost(env, ("self_collision_trunk", "self_collision_legs")),
        torch.tensor([2.0]),
    )
    env.scene.sensors["feet_ground_contact"].data.net_forces_w = _Proxy(torch.tensor([[[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]]))
    assert bool(c.nan_state(env, sensor_names=("feet_ground_contact",))[0])
    assert not bool(c.terrain_out_of_bounds(env)[0])


def test_self_collision_ignores_unfiltered_ground_contact_force():
    env = _Env()
    env.scene.sensors["self_collision_trunk"].data.net_forces_w = _Proxy(
        torch.full((1, 1, 3), 100.0)
    )
    env.scene.sensors["self_collision_trunk"].data.force_matrix_w = _Proxy(
        torch.zeros(1, 1, 2, 3)
    )
    env.scene.sensors["self_collision_trunk"]._contact_counts.zero_()
    assert torch.equal(
        c.self_collision_cost(env, ("self_collision_trunk", "self_collision_legs")),
        torch.tensor([0.0]),
    )


def test_self_collision_count_matches_mjlab_reference_kernel():
    from mjlab.tasks.velocity.mdp.rewards import self_collision_cost as mjlab_self_collision_cost

    env = _Env()
    isaac_cost = c.self_collision_cost(env, ("self_collision_trunk", "self_collision_legs"))
    reference_sensor = env.scene.sensors["self_collision_trunk"]
    reference_sensor.data.force_history = None
    reference_sensor.data.found = torch.tensor([[2.0]])
    mjlab_cost = mjlab_self_collision_cost(env, "self_collision_trunk")
    assert torch.equal(isaac_cost, mjlab_cost)


def test_stateful_swing_term_uses_isaaclab_manager_contract():
    source = (ROOT / "src/isaaclab_microduck/tasks/velocity_flat_contact.py").read_text()
    assert "from isaaclab.managers import ManagerTermBase, SceneEntityCfg" in source
    assert "class feet_swing_height(ManagerTermBase):" in source
    assert "super().__init__(cfg, env)" in source


def test_first_contact_is_normalized_to_boolean_for_stateful_masks():
    source = (ROOT / "src/isaaclab_microduck/tasks/velocity_flat_contact.py").read_text()
    assert "compute_first_contact(env.step_dt)).to(dtype=torch.bool)" in source
