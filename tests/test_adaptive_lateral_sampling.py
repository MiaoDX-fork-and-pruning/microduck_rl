"""Exercise command sampling without allocating a physics environment."""
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    make_microduck_adaptive_velocity_env_cfg,
)


def _command(n=128):
    cfg = make_microduck_adaptive_velocity_env_cfg(diagnostic_mode="lateral").commands["twist"]
    term = object.__new__(mdp.VelocityCommandCommandOnly)
    term.cfg = cfg
    term._env = SimpleNamespace(device="cpu", num_envs=n)
    term.vel_command_b = torch.full((n, 3), 9.)
    term.vel_command_w = term.vel_command_b.clone()
    for name in ("is_standing_env", "is_world_env", "is_heading_env", "is_forward_env"):
        setattr(term, name, torch.zeros(n, dtype=torch.bool))
    term.cfg.heading_command = False
    term.cfg.init_velocity_prob = 0.
    return term


def test_lateral_survives_update_without_turns_and_respects_reset_subset():
    term = _command()
    term.cfg.rel_turn_in_place_envs = 0.
    term.cfg.rel_lateral_envs = 1.
    term.cfg.rel_standing_envs = 0.
    ids = torch.arange(0, 128, 2)
    torch.manual_seed(17)
    term._resample_command(ids)
    term._update_command()
    command = term.command[ids]
    lateral = (command[:, 0] == 0.) & (command[:, 2] == 0.)
    assert torch.all((command[lateral, 1].abs() >= .12) & (command[lateral, 1].abs() <= .30))
    assert torch.any(command[lateral, 1] < 0.) and torch.any(command[lateral, 1] > 0.)
    assert not term.is_standing_env[ids].any()
    assert torch.equal(term.vel_command_w[ids][lateral], command[lateral])
    assert torch.all(term.command[1::2] == 9.)


def test_disabled_lateral_does_not_change_commands_or_rng():
    term = _command()
    term.cfg.rel_lateral_envs = 0.
    ids = torch.arange(128)
    torch.manual_seed(123)
    term._resample_command(ids)
    command = term.command.clone()
    rng = torch.get_rng_state().clone()
    # Older cfg objects do not have this field; both paths must remain identical.
    term.cfg = SimpleNamespace(**{k: v for k, v in vars(term.cfg).items() if k != "rel_lateral_envs"})
    torch.manual_seed(123)
    term._resample_command(ids)
    assert torch.equal(term.command, command)
    assert torch.equal(torch.get_rng_state(), rng)


def test_lateral_bucket_preserves_standing_and_turn_flags():
    term = _command()
    term.cfg.rel_turn_in_place_envs = 0.25
    term.cfg.rel_lateral_envs = 0.20
    term.cfg.rel_standing_envs = 0.20
    ids = torch.arange(128)
    torch.manual_seed(23)
    term._resample_command(ids)
    standing = term.is_standing_env.clone()
    turning = (term.command[:, 0] == 0) & (term.command[:, 1] == 0) & (term.command[:, 2].abs() >= 0.4)
    lateral = (term.command[:, 0] == 0) & (term.command[:, 1].abs() >= 0.12) & (term.command[:, 2] == 0)
    assert not torch.any(lateral & standing)
    assert not torch.any(lateral & turning)
    assert int(standing.sum()) > 0
    assert int(turning.sum()) > 0
    assert int(lateral.sum()) > 0
