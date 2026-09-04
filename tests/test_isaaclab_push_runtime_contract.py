from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/isaaclab/push_runtime_probe.py"


def test_push_probe_uses_production_task_and_event_manager_term() -> None:
    source = SCRIPT.read_text()
    assert 'gym.make("IsaacLab-Velocity-Flat-MicroDuck"' in source
    assert 'get_term_cfg("push_robot")' in source
    assert "event_cfg.func(" in source
    assert "write_root_velocity_to_sim_index" not in source
    assert '"physics_manager"' in source


def test_push_probe_proves_subset_and_reset_non_accumulation() -> None:
    source = SCRIPT.read_text()
    assert "torch.tensor([1, 3]" in source
    assert '"selected_first_exact"' in source
    assert '"unselected_first_unchanged"' in source
    assert '"reset_restores_default"' in source
    assert '"selected_second_exact"' in source
    assert '"unselected_second_unchanged"' in source
    assert '"status": "PROVEN"' in source
