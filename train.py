import os
import numpy as np
import gymnasium as gym
from gym_env import SatelliteEnv  # Ensure your environment is importable
from gymnasium.envs.registration import register
import ray
from ray import tune
from ray.tune.registry import register_env
from ray import air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
from ray.rllib.env import PettingZooEnv


env_config={
    "num_targets": 10,
    "num_observers": 10,
    "simulator_type": 'everyone',
    "time_step": 1,
    "duration": 24 * 60 * 60
}

'''
def env_creator(env_config):
    return SatelliteEnv(
        num_targets=env_config['num_targets'],
        num_observers=env_config['num_observers'],
        simulator_type=env_config['simulator_type'],
        time_step=env_config['time_step'],
        duration=env_config['duration']
    )
'''

def env_creator(env_config):
    env = SatelliteEnv(**env_config)
    return PettingZooEnv(env)  # Wrap your environment

register_env("SatelliteEnv", lambda config: PettingZooEnv(env_creator(config)))

def get_config(algorithm_name):
    if algorithm_name == "PPO":
        config = PPOConfig()
        config = config.training(
            lr=tune.loguniform(1e-4, 1e-2),
            gamma=tune.uniform(0.9, 0.99),
            use_gae=True,
            lambda_=tune.uniform(0.9, 1.0),  # Note: 'lambda_' not 'lambda'
            model={"fcnet_hiddens": tune.choice([[256, 256], [128, 128], [64, 64]]), "fcnet_activation": "relu"}
        )
    elif algorithm_name == "DQN":
        config = DQNConfig()
        config = config.training(
            lr=tune.loguniform(1e-4, 1e-2),
            gamma=tune.uniform(0.9, 0.99),
            learning_starts=1000,
            buffer_size=50000,
            model={"fcnet_hiddens": tune.choice([[256, 256], [128, 128], [64, 64]]), "fcnet_activation": "relu"}
        )
    
    config = config.environment(
        env="SatelliteEnv",
        env_config=env_config,
        disable_env_checking=True,
    )
    config = config.rollouts(
        num_rollout_workers=4,
        num_envs_per_worker=4,
    )
    config = config.resources(
        num_gpus=1 if ray.is_initialized() and ray.available_resources().get("GPU", 0) > 0 else 0,
    )
    return config

def run_experiments_with_tuning():
    algo_names = ["PPO", "DQN"]  # Extend this list with other algorithms
    for algo in algo_names:
        config = get_config(algo)
        search_alg = OptunaSearch(metric="episode_reward_mean", mode="max")
        scheduler = ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)
        tuner = tune.Tuner(
            algo,
            run_config=air.RunConfig(
                stop={"training_iteration": 100},
                name="HyperparamSearch_" + algo,
            ),
            param_space=config.to_dict(),
            tune_config=tune.TuneConfig(
                num_samples=10,
                metric="episode_reward_mean", 
                mode="max",
                search_alg=search_alg,
                scheduler=scheduler,
            ),
        )
        tuner.fit()

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    run_experiments_with_tuning()
    ray.shutdown()
