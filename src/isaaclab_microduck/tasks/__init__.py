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

    registrations = {
        "IsaacLab-Velocity-Flat-MicroDuck": (
            "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatEnvCfg",
            "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:MicroduckVelocityFlatPPORunnerCfg",
        ),
        "IsaacLab-Velocity-Flat-MicroDuck-Adapted": (
            "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatAdaptedEnvCfg",
            "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:MicroduckVelocityFlatAdaptedPPORunnerCfg",
        ),
        "IsaacLab-Velocity-Flat-MicroDuck-ActionRateFlat": (
            "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatActionRateFlatEnvCfg",
            "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:MicroduckVelocityFlatActionRateFlatPPORunnerCfg",
        ),
        "IsaacLab-Velocity-Flat-MicroDuck-Symmetry": (
            "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatSymmetryEnvCfg",
            "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:MicroduckVelocityFlatSymmetryPPORunnerCfg",
        ),
        "IsaacLab-Velocity-Flat-MicroDuck-CommandBuckets": (
            "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatCommandBucketsEnvCfg",
            "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:MicroduckVelocityFlatCommandBucketsPPORunnerCfg",
        ),
        "IsaacLab-Velocity-Flat-MicroDuck-CommandBucketsSymmetry": (
            "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatCommandBucketsSymmetryEnvCfg",
            "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:MicroduckVelocityFlatCommandBucketsSymmetryPPORunnerCfg",
        ),
        "IsaacLab-Velocity-Flat-MicroDuck-Strictification": (
            "isaaclab_microduck.tasks.velocity_flat:IsaacLabVelocityFlatStrictificationEnvCfg",
            "isaaclab_microduck.tasks.agents.rsl_rl_ppo_cfg:MicroduckVelocityFlatStrictificationPPORunnerCfg",
        ),
    }
    for task_id, (env_cfg_entry_point, rsl_rl_cfg_entry_point) in registrations.items():
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": env_cfg_entry_point,
                    "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
                },
            )


__all__ = ["available_tasks", "register_tasks"]
