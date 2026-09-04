"""RSL-RL configuration used by the IsaacLab Velocity-Flat task.

Observation normalization is part of the shared deployment contract. IsaacLab's
RSL-RL runner owns the running statistics and serializes them with the policy
checkpoint, so playback and export use the same transform as training.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MicroduckVelocityFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # Match the accepted mjlab Velocity-Flat PPO recipe. CLI smoke tests still
    # override max_iterations to 5 explicitly.
    max_iterations = 6000
    save_interval = 250
    experiment_name = "microduck_isaaclab_velocity_flat_mjlab_match"
    # Keep the critic capacity identical to the accepted mjlab recipe.  The
    # critic currently consumes the shared policy observation group; once the
    # privileged critic group is ported, this remains the same architecture.
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = ["MicroduckVelocityFlatPPORunnerCfg"]
