from pathlib import Path


def test_sim_smoke_script_has_bounded_step_contract() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/sim_smoke.py").read_text()

    assert "SimulationContext" in source
    assert "for _ in range(args.steps)" in source
    assert "simulation_app.close()" in source
