from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_one_update_probe_declares_version_and_update_outputs() -> None:
    source = (ROOT / "scripts/rsl_rl_one_update_probe.py").read_text()
    for field in (
        '"rsl_rl_version"',
        '"initial_actor_sha256"',
        '"final_actor_sha256"',
        '"metrics"',
        "PPO.construct_algorithm",
    ):
        assert field in source


def test_probe_executes_one_host_update(tmp_path: Path) -> None:
    from scripts.rsl_rl_one_update_probe import run

    result = run(tmp_path / "probe.json", seed=2026)
    assert result["num_envs"] == 8
    assert result["num_steps_per_env"] == 4
    assert result["initial_actor_sha256"] != result["final_actor_sha256"]
    assert set(result["metrics"]) == {"entropy", "surrogate", "value"}
