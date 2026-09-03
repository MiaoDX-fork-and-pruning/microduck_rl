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
    max_iterations = 5
    save_interval = 5
    experiment_name = "microduck_isaaclab_velocity_flat_normalized"
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Locomotion acceptance requires bounded deterministic action means;
        # entropy pressure otherwise inflated the Gaussian scale to ~22 in the
        # first normalized run despite the environment-side action clip.
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = ["MicroduckVelocityFlatPPORunnerCfg"]
