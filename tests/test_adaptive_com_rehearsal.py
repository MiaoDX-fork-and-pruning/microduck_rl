"""Final-CoM rehearsal uses real stock DR and keeps trainer/eval boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp import dr
from mjlab.managers import SceneEntityCfg

from mjlab_microduck.evaluation.capability import BUCKETS
from mjlab_microduck.tasks.adaptive_curriculum import (
    ADAPTIVE_AXIS_CONFIGS,
    CapabilityGate,
    apply_stage_to_env,
)
from mjlab_microduck.tasks.adaptive_runner import AdaptiveMicroduckOnPolicyRunner
from mjlab_microduck.tasks.mdp import randomize_com_with_rehearsal
from mjlab_microduck.tasks.microduck_adaptive_velocity_env_cfg import (
    make_microduck_adaptive_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_real_stock_dr_bounds_partial_reset_and_no_accumulation():
    defaults = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    field = defaults[None].repeat(10, 1, 1)
    env = SimpleNamespace(
        num_envs=10, device="cpu",
        scene={"robot": SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.arange(2, dtype=torch.int)))},
        sim=SimpleNamespace(model=SimpleNamespace(body_ipos=field), get_default_field=lambda _: defaults),
    )
    params = {"ranges": (-0.003, 0.003), "final_ranges": (-0.015, 0.015),
                  "final_fraction": 0.2, "asset_cfg": SceneEntityCfg("robot", body_ids=[1])}
    torch.manual_seed(17)
    randomize_com_with_rehearsal(env, None, **params)
    first = field.clone()
    offset = field[:, 1] - defaults[1]
    assert offset[:2].abs().max() > 0.003
    assert offset[:2].abs().max() <= 0.015
    assert offset[2:].abs().max() <= 0.00301
    torch.testing.assert_close(field[:, 0], defaults[0].expand(10, -1))
    # Same seed after prior randomization reproduces exactly: stock DR reads
    # defaults, not the already perturbed model. No mocking the DR operation.
    torch.manual_seed(17)
    randomize_com_with_rehearsal(env, None, **params)
    torch.testing.assert_close(field, first, rtol=0, atol=0)
    params.update(ranges=(0.002, 0.002), final_ranges=(0.010, 0.010))
    for _ in range(3):
        randomize_com_with_rehearsal(env, torch.tensor([8, 1]), **params)
    torch.testing.assert_close(field[1, 1], defaults[1] + 0.010)
    torch.testing.assert_close(field[8, 1], defaults[1] + 0.002)
    untouched = [0, 2, 3, 4, 5, 6, 7, 9]
    torch.testing.assert_close(field[untouched], first[untouched], rtol=0, atol=0)


def test_rehearsal_retains_full_stock_recompute_contract():
    assert randomize_com_with_rehearsal.model_fields == dr.body_ipos.model_fields
    assert randomize_com_with_rehearsal.recompute == dr.body_ipos.recompute


@pytest.mark.parametrize("mode,names", [
    ("com", ("com_range",)), ("head_com", ("head_com_range",)),
    ("composed", ("com_range", "head_com_range")),
])
def test_only_owned_axes_get_rehearsal_and_stage_ranges_remain_live(mode, names):
    cfg = make_microduck_adaptive_velocity_env_cfg(axis_mode=mode)
    live = deepcopy(cfg.events)
    runner = object.__new__(AdaptiveMicroduckOnPolicyRunner)
    runner.env = SimpleNamespace(cfg=cfg, event_manager=SimpleNamespace(get_term_cfg=live.__getitem__))
    runner.capability_gate = CapabilityGate(
        tuple(a for a in ADAPTIVE_AXIS_CONFIGS if a.name in names), axis_mode=mode,
        critical_buckets=BUCKETS,
    )
    runner._set_final_com_fraction(0.2)
    for axis in ADAPTIVE_AXIS_CONFIGS:
        event = {"com_range": "randomize_com", "head_com_range": "randomize_head_com"}[axis.name]
        if axis.name in names:
            apply_stage_to_env(runner.env, axis.name, 0.005)
            assert live[event].func is randomize_com_with_rehearsal
            assert live[event].params["ranges"] == (-0.005, 0.005)
            assert live[event].params["final_ranges"] == (-axis.stages[-1], axis.stages[-1])
        else:
            assert live[event] == cfg.events[event]
        assert cfg.events[event].func is dr.body_ipos  # manager owns the change
    runner._set_final_com_fraction(0.0)
    assert all(live[event].func is dr.body_ipos for event in ("randomize_com", "randomize_head_com"))


@pytest.mark.parametrize("fraction", [-0.1, 0.21, float("nan"), float("inf")])
def test_launch_rejects_unbounded_rehearsal(monkeypatch, fraction):
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION", str(fraction))
    with pytest.raises(ValueError, match="fraction must be"):
        make_microduck_adaptive_velocity_env_cfg()


def test_opt_in_does_not_change_canonical_events(monkeypatch):
    canonical = make_microduck_velocity_env_cfg()
    monkeypatch.setenv("MICRODUCK_ADAPTIVE_FINAL_COM_FRACTION", "0.2")
    adaptive = make_microduck_adaptive_velocity_env_cfg()
    assert adaptive.adaptive_final_com_fraction == 0.2
    assert make_microduck_velocity_env_cfg().events == canonical.events
