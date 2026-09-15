"""Runner adapter for feeding frozen capability-battery results to the gate."""

from __future__ import annotations

from typing import Mapping

from .adaptive_curriculum import (
    ADAPTIVE_AXIS_CONFIGS,
    CapabilityGate,
    apply_stage_to_env,
)
from . import MicroduckOnPolicyRunner


class AdaptiveMicroduckOnPolicyRunner(MicroduckOnPolicyRunner):
    """Canonical PPO runner plus explicit, checkpointed curriculum state."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        mode = getattr(env.cfg, "adaptive_axis_mode", "composed")
        axis_names = {
            "all_static": (),
            "com": ("com_range",),
            "head_com": ("head_com_range",),
            "composed": ("com_range", "head_com_range"),
        }.get(mode)
        if axis_names is None:
            raise ValueError(f"unsupported adaptive axis mode: {mode}")
        axis_configs = tuple(axis for axis in ADAPTIVE_AXIS_CONFIGS if axis.name in axis_names)
        if not axis_configs:
            self.capability_gate = None
            return
        self.capability_gate = CapabilityGate(
            axis_configs,
            critical_buckets=("zero", "forward", "lateral", "yaw", "turn-left", "turn-right"),
            ema_alpha=0.25,
            preservation_tolerance=0.05,
        )

    def record_capability_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        checkpoint: str | None = None,
        seed: int = 0,
    ):
        """Consume one frozen battery window and apply at most one transition."""
        if self.capability_gate is None:
            return None
        transition = self.capability_gate.update(
            self.current_learning_iteration if step is None else step,
            metrics,
            checkpoint=checkpoint,
            seed=seed,
        )
        if transition is not None:
            apply_stage_to_env(
                self.env,
                transition.axis,
                self.capability_gate.stage_value(transition.axis),
            )
        return transition

    def save(self, path: str, infos: dict | None = None) -> None:
        payload = {} if infos is None else dict(infos)
        payload["adaptive_curriculum"] = (
            None if self.capability_gate is None else self.capability_gate.state_dict()
        )
        super().save(path, payload)

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and infos.get("adaptive_curriculum") and self.capability_gate is not None:
            self.capability_gate.load_state_dict(infos["adaptive_curriculum"])
            for axis_name, state in self.capability_gate.states.items():
                apply_stage_to_env(
                    self.env,
                    axis_name,
                    self.capability_gate.stage_value(axis_name),
                )
        return infos
