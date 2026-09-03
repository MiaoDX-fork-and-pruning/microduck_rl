"""IsaacLab task registrations.

Framework registration is deferred until an IsaacLab runtime explicitly asks
for it; importing this module must remain safe for mjlab-only workflows.
"""

from __future__ import annotations

from ..registry import available_tasks, require_isaaclab


def register_tasks() -> None:
    """Register IsaacLab tasks after validating the simulator runtime."""

    require_isaaclab()
    import gymnasium as gym

    task_id = "IsaacLab-Velocity-Flat-MicroDuck"
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:"
                    "MicroduckVelocityFlatPPORunnerCfg"
                ),
            },
        )


__all__ = ["available_tasks", "register_tasks"]
