import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("velstand_probe", ROOT / "scripts/run_velstand_causal_probe.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_case_list_is_frozen_and_covers_nine_buckets():
    mod = _runner(); cases = mod.case_list()
    assert len(cases) == 32
    assert {row["bucket"] for row in cases} == set(mod.BUCKETS)
    assert {row["command_bucket"] for row in cases} == {"zero", "nominal_hold"}
    assert {row["horizon_ticks"] for row in cases} == {1000}


def test_teacher_initialization_is_action_exact():
    from mjlab_microduck.generalist_model import VelstandProbeActor, initialize_velstand_probe_from_teacher
    from mjlab_microduck.generalist_schema import make_conditioned_observation
    from mjlab_microduck.generalist_teachers import FrozenG0Teachers
    import numpy as np
    actor = VelstandProbeActor()
    mapping = initialize_velstand_probe_from_teacher(actor, str(ROOT / "artifacts/specialists/velstand_flat/checkpoint.pt"))
    rng = np.random.default_rng(3)
    legacy = rng.normal(size=(4, 61)).astype(np.float32)
    conditioned = make_conditioned_observation(legacy, legacy[:, 48:61], "stand")
    with torch.inference_mode():
        expected = FrozenG0Teachers()(torch.from_numpy(conditioned))
        actual = actor(torch.from_numpy(conditioned))
    assert torch.max(torch.abs(actual - expected)) <= 1e-5
    assert mapping["normalizer"] == "folded_into_first_linear_std_plus_0.01"
