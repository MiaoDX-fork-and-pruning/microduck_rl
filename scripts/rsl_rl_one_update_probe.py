"""Compare one deterministic synthetic RSL-RL rollout/update.

The probe intentionally avoids either simulator. It exercises the installed
runner's model construction, rollout storage, return calculation, and one PPO
update on identical tensors so package-version differences are isolated from
task and physics differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
from tensordict import TensorDict


class _SyntheticEnv:
    num_envs = 8
    num_actions = 3


def _parameter_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(output: Path, seed: int) -> dict[str, object]:
    from rsl_rl.algorithms import PPO

    torch.manual_seed(seed)
    env = _SyntheticEnv()
    obs = TensorDict(
        {
            "policy": torch.randn(env.num_envs, 5),
            "critic": torch.randn(env.num_envs, 7),
        },
        batch_size=[env.num_envs],
    )
    cfg: dict[str, object] = {
        "num_steps_per_env": 4,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [16, 8],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [16, 8],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "multi_gpu": None,
    }
    algorithm = PPO.construct_algorithm(obs, env, cfg, "cpu")
    initial_actor = _parameter_digest(algorithm.actor)
    initial_critic = _parameter_digest(algorithm.critic)
    for step in range(cfg["num_steps_per_env"]):  # type: ignore[arg-type]
        next_obs = TensorDict(
            {
                "policy": torch.randn(env.num_envs, 5),
                "critic": torch.randn(env.num_envs, 7),
            },
            batch_size=[env.num_envs],
        )
        algorithm.act(obs)
        rewards = (next_obs["policy"][:, 0] * 0.25 + next_obs["critic"][:, 0] * 0.1).to(torch.float32)
        dones = torch.zeros(env.num_envs, dtype=torch.bool)
        algorithm.process_env_step(next_obs, rewards, dones, {})
        obs = next_obs
    algorithm.compute_returns(obs)
    metrics = algorithm.update()
    result = {
        "schema_version": 1,
        "seed": seed,
        "rsl_rl_version": version("rsl-rl-lib"),
        "torch_version": torch.__version__,
        "num_envs": env.num_envs,
        "num_steps_per_env": 4,
        "initial_actor_sha256": initial_actor,
        "initial_critic_sha256": initial_critic,
        "final_actor_sha256": _parameter_digest(algorithm.actor),
        "final_critic_sha256": _parameter_digest(algorithm.critic),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
