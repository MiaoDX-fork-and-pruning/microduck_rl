from pathlib import Path

from mjlab_microduck.evaluation.capability import build_capability_report, BUCKETS
from mjlab_microduck.evaluation.protocol import FrozenCapabilityEvaluator


def test_protocol_is_runtime_checkable_by_structural_usage() -> None:
    class Fake:
        def evaluate(self, **kwargs):
            return build_capability_report(
                {name: {"survival_fraction": 1.0, "tilt_p95_rad": 0.1,
                        ("zero_drift_m" if name == "zero" else
                         "angular_tracking_error_rad_s" if name in ("yaw", "turn-left", "turn-right")
                         else "tracking_error_m_s"): 0.0} for name in BUCKETS},
                metadata={"task_id": "x", "source_sha": "s", "evaluator_config_sha256": "c",
                          "checkpoint": str(Path("p")), "checkpoint_sha256": "h",
                          "policy_format": "onnx", "seed_set_id": "seed", "generated_at": "now"},
            )
    result = Fake().evaluate(checkpoint_path=Path("p"), task_id="x", axis_mode="composed",
                             curriculum_state={}, iteration=1, seed_set_id="seed")
    assert result.payload["schema_version"] == 2
    assert FrozenCapabilityEvaluator is not None
