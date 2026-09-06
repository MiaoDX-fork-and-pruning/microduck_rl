import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_specialist_reset_contract import EvaluatorTrace, compare_evaluator_traces

_SPEC = importlib.util.spec_from_file_location(
    "replay_specialist_recovery_trace",
    ROOT / "scripts" / "replay_specialist_recovery_trace.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_legacy_recovery_trace_is_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, tilt_rad=np.array([0.9], dtype=np.float32))
    with pytest.raises(ValueError, match="integration state"):
        _MODULE.replay(path)


def test_native_capture_is_a_pre_step_copy_and_distinguishes_clipping(tmp_path):
    observation = {"actor": torch.ones((1, 61))}
    action = torch.full((1, 14), 2.0)
    raw = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(
            qpos=torch.ones((1, 21)), qvel=torch.zeros((1, 20)), ctrl=torch.zeros((1, 14)),
        )),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_pose_w=torch.zeros((1, 7))))},
        action_manager=SimpleNamespace(action=torch.full((1, 14), 0.5)),
        episode_length_buf=torch.tensor([17]),
        termination_manager=SimpleNamespace(active_terms=["time_out"], get_term=lambda _: torch.tensor([True])),
    )
    env = SimpleNamespace(unwrapped=raw, clip_actions=1.0)
    active = torch.tensor([True])
    trace = EvaluatorTrace({"schema": "specialist-evaluator-trace", "version": 1})
    trace.before_step(env, observation, action, active)
    # env.step may mutate observations and auto-reset the simulator in place.
    observation["actor"].zero_()
    raw.sim.data.qpos.zero_()
    active.zero_()
    trace.after_step(raw, torch.tensor([3.0]), torch.tensor([1]))
    output = tmp_path / "native.npz"
    trace.write(output)
    with np.load(output) as captured:
        np.testing.assert_array_equal(captured["qpos"], 1.0)
        np.testing.assert_array_equal(captured["observation"], 1.0)
        np.testing.assert_array_equal(captured["raw_action"], 2.0)
        np.testing.assert_array_equal(captured["applied_action"], 1.0)
        np.testing.assert_array_equal(captured["previous_action"], 0.5)
        assert captured["first_episode_active"].item()
        assert captured["termination"].item()
        assert captured["episode_step"].item() == 17
        assert not observation["actor"].any()


def test_native_prefix_comparison_rejects_truncation_state_and_contract_drift(tmp_path):
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    arrays = {"metadata": '{"seed":42}'}
    arrays.update({
        field: np.zeros((3, 2), dtype=bool if field in {"done", "first_episode_active"} else np.float32)
        for field in {
            "observation", "qpos", "qvel", "ctrl", "root_pose", "previous_action",
            "raw_action", "applied_action", "first_episode_active", "episode_step",
            "reward", "done", "termination",
        }
    })
    np.savez(reference, **arrays)
    np.savez(candidate, **arrays)
    assert compare_evaluator_traces(reference, candidate)["passed"]
    np.savez(candidate, **{**arrays, "qpos": arrays["qpos"][:-1]})
    assert not compare_evaluator_traces(reference, candidate)["passed"]
    np.savez(candidate, **{**arrays, "qpos": arrays["qpos"] + 0.01})
    assert not compare_evaluator_traces(reference, candidate)["passed"]
    np.savez(candidate, **{**arrays, "metadata": '{"seed":43}'})
    assert not compare_evaluator_traces(reference, candidate)["passed"]

    np.savez(candidate, metadata=arrays["metadata"], qpos=arrays["qpos"])
    result = compare_evaluator_traces(reference, candidate)
    assert not result["fields_match"]
    assert "termination" in result["required_fields"]
