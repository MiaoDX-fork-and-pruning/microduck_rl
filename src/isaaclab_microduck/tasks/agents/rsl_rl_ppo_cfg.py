"""RSL-RL configuration used by the IsaacLab Velocity-Flat task.

Observation normalization is part of the shared deployment contract. IsaacLab's
RSL-RL runner owns the running statistics and serializes them with the policy
checkpoint, so playback and export use the same transform as training.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from isaaclab_microduck.tasks.symmetry import microduck_velocity_symmetry


@configclass
class MicroduckVelocityFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # Match the production mjlab Velocity-Flat runner exactly: its inherited
    # default is None, so raw policy actions are not clipped at the VecEnv
    # boundary. BAM torque saturation and the command-side limit penalty own
    # the physical/action constraints.
    clip_actions = None
    # Match the accepted mjlab Velocity-Flat PPO recipe. CLI smoke tests still
    # override max_iterations to 5 explicitly.
    max_iterations = 6000
    save_interval = 250
    experiment_name = "microduck_isaaclab_velocity_flat_mjlab_match"
    # Keep the critic capacity identical to the accepted mjlab recipe.  The
    # critic currently consumes the shared policy observation group; once the
    # privileged critic group is ported, this remains the same architecture.
    # Actor stays the deployable 61D ABI; critic receives privileged velocity
    # and contact data from the task's separate group.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
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


@configclass
class MicroduckVelocityFlatAdaptedPPORunnerCfg(MicroduckVelocityFlatPPORunnerCfg):
    """Runner for the historical IsaacLab-specific adapted profile."""

    experiment_name = "microduck_isaaclab_velocity_flat_adapted"


@configclass
class MicroduckVelocityFlatActionRateFlatPPORunnerCfg(MicroduckVelocityFlatPPORunnerCfg):
    """Runner for the isolated strict action-rate curriculum diagnostic."""

    experiment_name = "microduck_isaaclab_velocity_flat_action_rate_flat"


@configclass
class MicroduckVelocityFlatSymmetryPPORunnerCfg(MicroduckVelocityFlatPPORunnerCfg):
    """Diagnostic runner with left-right data augmentation enabled."""

    experiment_name = "microduck_isaaclab_velocity_flat_symmetry"
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=microduck_velocity_symmetry,
        ),
    )


@configclass
class MicroduckVelocityFlatCommandBucketsPPORunnerCfg(MicroduckVelocityFlatPPORunnerCfg):
    """Strict diagnostic runner for the adapted command-bucket hypothesis."""

    experiment_name = "microduck_isaaclab_velocity_flat_command_buckets"


@configclass
class MicroduckVelocityFlatCommandBucketsSymmetryPPORunnerCfg(MicroduckVelocityFlatSymmetryPPORunnerCfg):
    """T21 runner: adapted command buckets plus left-right augmentation."""

    experiment_name = "microduck_isaaclab_velocity_flat_command_buckets_symmetry"


@configclass
class MicroduckVelocityFlatStrictificationPPORunnerCfg(MicroduckVelocityFlatPPORunnerCfg):
    """T22 runner for adapted bootstrap followed by strictification."""

    experiment_name = "microduck_isaaclab_velocity_flat_strictification"


__all__ = [
    "MicroduckVelocityFlatPPORunnerCfg",
    "MicroduckVelocityFlatAdaptedPPORunnerCfg",
    "MicroduckVelocityFlatActionRateFlatPPORunnerCfg",
    "MicroduckVelocityFlatSymmetryPPORunnerCfg",
    "MicroduckVelocityFlatCommandBucketsPPORunnerCfg",
    "MicroduckVelocityFlatCommandBucketsSymmetryPPORunnerCfg",
    "MicroduckVelocityFlatStrictificationPPORunnerCfg",
]
