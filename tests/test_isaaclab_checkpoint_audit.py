from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.isaaclab.velocity_flat_checkpoint_audit import audit


def _write_checkpoint(path: Path, *, mean: float) -> None:
    actor = {
        "mlp.0.weight": torch.zeros(2, 3),
        "distribution.std_param": torch.ones(2),
        "obs_normalizer._mean": torch.full((1, 3), mean),
        "obs_normalizer._std": torch.ones(1, 3),
        "obs_normalizer._var": torch.ones(1, 3),
        "obs_normalizer.count": torch.tensor(12),
    }
    critic = {"mlp.0.weight": torch.zeros(2, 3)}
    torch.save({"iter": int(mean), "actor_state_dict": actor, "critic_state_dict": critic}, path)


def test_checkpoint_audit_requires_matching_architecture_and_reports_normalizer_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    strict = tmp_path / "strict.pt"
    _write_checkpoint(reference, mean=0)
    _write_checkpoint(strict, mean=1)

    report = audit([strict], reference)

    assert report["classification"].startswith("checkpoint_schema_and_normalizer_present")
    comparison = report["comparisons"][0]
    assert comparison["normalizer_mean_linf"] == 1.0
    assert comparison["normalizer_count_equal"]
    assert comparison["actor_parameter_count_equal"]

