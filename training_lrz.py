import os
from typing import Dict
import argparse
import torch
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune import TuneConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from FSS_env import FSS_env
import json
import pandas as pd
from ray.util.accelerators import NVIDIA_A100
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.tune.search.bayesopt import BayesOptSearch
import matplotlib.pyplot as plt

# Enable mixed precision training in PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True # Ensure mixed precision in the training loop or algorithm configuration

os.environ["RAY_verbose_spill_logs"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
parser.add_argument("--stop-iters", type=int, default=200, help="Number of iterations to train.")
parser.add_argument("--policy", choices=["ppo", "dqn", "sac"], default="ppo", required=True, help="Policy to train.")
parser.add_argument("--checkpoint-dir", type=str, default="/mnt/checkpoints", help="Directory to save checkpoints.")
parser.add_argument("--resume", action="store_true", help="Whether to resume training from a checkpoint.")
parser.add_argument("--tune", action="store_true", help="Whether to perform hyperparameter tuning.")

args = parser.parse_args()
experiment_name = "FSS_env_train_" + args.policy

print("Starting training script")
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
ray.init(ignore_reinit_error=True, num_gpus=gpu_count)

###### EDIT HERE ##########################################################
# Environment configurations
env_config = {
    "num_targets": 3, 
    "num_observers": 3,
    "simulator_type": 'everyone',
    "time_step": 1, 
    "duration": 24*60*60
}

metric = "episode_reward_mean" # "mean_reward" # "episode_reward_mean" # "info/learner/default_policy/learner_stats/total_loss"
mode = "max"
batch_mode="complete_episodes" # "truncate_episodes" "complete_episodes"
rollout_fragment_length = "auto" # 256 # if batch_mode is “complete_episodes”, rollout_fragment_length is ignored. Data is given in chunks of 10 workers * 5 envs * 1000 steps_per_rollout = 50,000 steps 
train_batch_size = 2**10
sgd_minibatch_size = 32
num_sgd_iter = 1
num_learner_workers = 1 # parallel GPUs


# Resource allocation settings
# GPUs are automatically detected and used if available
resources = {
    "num_rollout_workers": 1, # Number of rollout workers (parallel actors for simulating environment interactions)
    "num_envs_per_worker": 1, # Number of environments per worker
    "num_cpus_per_worker": 5, # Number of CPUs per worker
    "num_gpus_per_worker": 0, # Number of GPUs per worker - can be 0 for CPU simulations
    "num_learner_workers": num_learner_workers, # For multi-gpu training change num_gpus_per_learner_worker
    "num_cpus_per_learner_worker": 1, # Number of CPUs per local worker (trainer) =1!!!!!
    "num_gpus_per_learner_worker": 0, # Number of GPUs per local worker (trainer)
}

# Serch space configurations
if args.policy == "ppo":
    search_space = {
        "lr": tune.uniform(1e-7, 1e-4),
        "gamma": tune.uniform(0.9, 0.99),
        "lambda": tune.uniform(0.9, 1.0),
    }
elif args.policy == "dqn":
    search_space = {
        "target_network_update_freq": tune.choice([500, 1000, 2000, 3600]),
        "lr_schedule": tune.choice([[[0, 1e-4], [1000000, 1e-5]], [[0, 1e-3], [1000000, 1e-4]]]),
    }
elif args.policy == "sac":
    search_space = {
        "lr": tune.loguniform(1e-7, 1e-4),
        "gamma": tune.uniform(0.9, 0.99),
        "lambda": tune.uniform(0.9, 1.0),
    }

search_alg = BayesOptSearch(
        metric=metric,
        mode=mode
    )

# Hyperparameter search
num_samples_per_policy = 1 # random combinations of search space
max_concurrent_trials = num_learner_workers # number of trials to run concurrently
checkpoint_frequency = 10

# Scheduler - Jetson ~1M steps a day
scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=metric,
        mode=mode, # maximize the reward
        max_t=30, # maximum number of training iterations (complete episodes * workers) - Exploration ~10-20, Exploitation ~30-50
        grace_period=20, # * iter_for_complete_episodes, minimum number of training iterations
        reduction_factor=2, # factor to reduce the number of trials
    )


###########################################################################

def env_creator(env_config):
    env = FSS_env(**env_config)
    return env
    
run_config = train.RunConfig(
                verbose=1,
                name=experiment_name,
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute=metric,
                    checkpoint_score_order=mode
                ),
                stop={
                    'training_iteration': args.stop_iters
                },
            )
            
scaling_config = ScalingConfig(
                num_workers=resources["num_learner_workers"],
                use_gpu=True,
                resources_per_worker={
                    "CPU": resources["num_cpus_per_learner_worker"],
                    "GPU": resources["num_gpus_per_learner_worker"],
                }
            )
            
tune_config = tune.TuneConfig(
        max_concurrent_trials=max_concurrent_trials,
        num_samples=num_samples_per_policy,
        search_alg=search_alg,
        scheduler=scheduler,
    )
            


# @ray.remote(accelerator_type=NVIDIA_A100)
def train_func(config: Dict):
    if args.policy == "ppo":
        algo = "PPO"
    elif args.policy == "sac":
        algo = "SAC"
    elif args.policy == "dqn":
        algo = "DQN"
        
    algorithm_checkpoint_path = os.path.join(args.checkpoint_dir, algo)
    
    # Ensure the checkpoint directory exists
    if not os.path.exists(algorithm_checkpoint_path):
        try:
            os.makedirs(algorithm_checkpoint_path)
            print(f"Created directory: {algorithm_checkpoint_path}")
        except Exception as e:
            print(f"Failed to create directory: {algorithm_checkpoint_path} with error {e}")

    # print(f"Starting training function for {algo}")
    
    # Apply hyperparameters to algo_config
    algo_config = setup_config(algo)

        

    # Initialize the algorithm trainer
    algorithm = algo_config.build()
    # print(f"Trainer built for {algo}")

    # Load checkpoint if available
    checkpoint = train.get_checkpoint()
    print(f"Using checkpoint: {checkpoint}")
    if checkpoint:
        print("Restoring from checkpoint")
        algorithm.from_checkpoint(checkpoint.path)

    # Training loop
    for i in range(args.stop_iters):
        # print(f"Starting iteration {i+1}/{args.stop_iters}")

        result = algorithm.train()
        train.report(
            metrics={"episode_reward_mean": result[metric]},
        )
        if i % checkpoint_frequency == 0:
            try:
                checkpoint = algorithm.save(algorithm_checkpoint_path)
                print(f"Saving checkpoint at {algorithm_checkpoint_path}")
                print(f"Checkpoint saved at {checkpoint}")
                # List the files in the directory to verify
                print(f"Checkpoint directory contents: {os.listdir(algorithm_checkpoint_path)}")
            except Exception as e:
                print(f"Failed to save checkpoint at {algorithm_checkpoint_path} with error {e}")

    print(f"Training function for {algo} completed")
    

def run_experiment(algo):
    algo_config = setup_config(algo)
    
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
    )

    if args.tune:
        if args.policy == "sac":
            algo_config = algo_config.training(
                lr=search_space["lr"],
                gamma=search_space["gamma"],
            )
            results = tune.run(
            "SAC",
            config=algo_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="sac_tune",
            checkpoint_at_end=False
        )
        elif args.policy == "ppo":
            algo_config = algo_config.training(
                lr=search_space["lr"],
                gamma=search_space["gamma"],
                lambda_=search_space["lambda"],
                # train_batch_size_per_learner=train_batch_size,
                # sgd_minibatch_size=sgd_minibatch_size,
                # num_sgd_iter=num_sgd_iter,
            )
            results = tune.run(
            "PPO",
            config=algo_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="ppo_tune",
            checkpoint_at_end=False
        )
        elif args.policy == "dqn":
            algo_config = algo_config.training(
                target_network_update_freq=search_space["target_network_update_freq"],
                lr_schedule=search_space["lr_schedule"],
            )
            results = tune.run(
            "DQN",
            config=algo_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="dqn_tune",
            checkpoint_at_end=False
        )
    elif args.resume:
        if args.policy == "sac":
            algo_config = algo_config.training(
                lr=9.69543e-06, # SAC_FSS_env-v0_0f4fd_00003
                gamma= 0.946126,
            )
        elif args.policy == "ppo":
            algo_config = algo_config.training(
                lr=6.38965e-05, # 
                gamma=0.9745,
                lambda_=0.92,
                train_batch_size_per_learner=train_batch_size,
                sgd_minibatch_size=sgd_minibatch_size,
                num_sgd_iter=num_sgd_iter,
            )
        elif args.policy == "dqn":
            algo_config = algo_config.training(
                target_network_update_freq=1000, # DQN_FSS_env-v0_7c8f4_00003
                lr_schedule=[[0, 1e-4], [1000000, 1e-5]],
            )
        results = trainer.fit()
    return results


def setup_config(algo):
    if algo == "PPO":
        algo_config = PPOConfig()
        algo_config = algo_config.experimental(_enable_new_api_stack=True)
    elif algo == "DQN":
        algo_config = DQNConfig()
        algo_config = algo_config.experimental(_enable_new_api_stack = False)
    elif algo == "SAC":
        algo_config = SACConfig()
        algo_config = algo_config.experimental(_enable_new_api_stack=False)
        
        
    algo_config = algo_config.environment(env="FSS_env-v0", env_config=env_config, disable_env_checking=True)
    algo_config = algo_config.framework(args.framework)
    algo_config = algo_config.rollouts(
        num_rollout_workers=resources["num_rollout_workers"],
        num_envs_per_worker=resources["num_envs_per_worker"],
        rollout_fragment_length=rollout_fragment_length,
        batch_mode=batch_mode,
    )
    algo_config = algo_config.resources(
        num_gpus=gpu_count,
        num_learner_workers=num_learner_workers,  # <- in most cases, set this value to the number of GPUs
        num_gpus_per_learner_worker=resources["num_gpus_per_learner_worker"],  # <- set this to 1, if you have at least 1 GPU
        num_cpus_for_local_worker=resources["num_cpus_per_learner_worker"],
    )
    return algo_config


register_env("FSS_env-v0", lambda config: env_creator(env_config))
print("Registered environment")


if args.policy == "ppo":
    results = run_experiment("PPO")
    # pretty_print(results)
if args.policy == "sac":
    results = run_experiment("SAC")
    # pretty_print(results)
if args.policy == "dqn":
    results = run_experiment("DQN")
    # pretty_print(results)
else:
    print("Invalid policy specified. Please choose either 'ppo', 'sac' or 'dqn'.")
