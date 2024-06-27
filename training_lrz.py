import os
import time
import tempfile
import argparse
import torch
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune import TuneConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from FSS_env import FSS_env
import json
import pandas as pd
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
from torch.profiler import profile, record_function, ProfilerActivity
from ray.util.accelerators import NVIDIA_A100
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device, get_devices
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.tune.search.bayesopt import BayesOptSearch
import matplotlib.pyplot as plt

# Enable mixed precision training in PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True # Ensure mixed precision in the training loop or algorithm configuration



print("Starting training script")
# ray.init(num_gpus=gpu_count)

# Jetson (10 rw, 1 env, auto length, complete episodes, 11 cpu, 1 gpu)
# More than 162140 observations in 16214 env steps for episode 839967284702637949 are buffered in the sampler. 
# If this is more than you expected, check that that you set a horizon on your environment correctly and that it terminates at some point. 
# Note: In multi-agent environments, `rollout_fragment_length` sets the batch size based on (across-agents) environment steps, not the steps of individual agents, which can result in unexpectedly large batches.
# Also, you may be waiting for your Env to terminate (batch_mode=`complete_episodes`). Make sure it does at some point.

###### EDIT HERE ##########################################################

# Environment configurations
env_config = {
    "num_targets": 50, 
    "num_observers": 50,
    "simulator_type": 'everyone',
    "time_step": 1, 
    "duration": 20000 # 24*60*60
}

metric = "episode_reward_mean" # "info/learner/default_policy/learner_stats/total_loss"
mode = "max"
batch_mode="complete_episodes" # "truncate_episodes" "complete_episodes"
rollout_fragment_length = "auto" # if batch_mode is “complete_episodes”, rollout_fragment_length is ignored. Data is given in chunks of 10 workers * 5 envs * 1000 steps_per_rollout = 50,000 steps 
num_learner_workers = 4 # parallel GPUs

# Resource allocation settings
# GPUs are automatically detected and used if available
resources = {
    "num_rollout_workers": 1, # Number of rollout workers (parallel actors for simulating environment interactions)
    "num_envs_per_worker": 1, # Number of environments per worker
    "num_cpus_per_worker": 1, # Number of CPUs per worker
    "num_gpus_per_worker": 0, # Number of GPUs per worker - can be 0 for CPU simulations
    "num_learner_workers": num_learner_workers, # For multi-gpu training change num_gpus_per_learner_worker
    "num_cpus_per_learner_worker": 1, # Number of CPUs per local worker (trainer) =1!!!!!
    "num_gpus_per_learner_worker": 1, # Number of GPUs per local worker (trainer)
}

# Serch space configurations
search_space = {
    "lr": tune.loguniform(1e-7, 1e-4),
    "gamma": tune.uniform(0.9, 0.99),
    "lambda": tune.uniform(0.9, 1.0),
}

# Hyperparameter search
num_samples_per_policy = 20 # random combinations of search space
max_concurrent_trials = num_learner_workers # number of trials to run concurrently
checkpoint_frequency = 5

# Scheduler - Jetson ~1M steps a day
scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=metric,
        mode=mode, # maximize the reward
        max_t=20, # maximum number of training iterations (complete episodes * workers) - Exploration ~10-20, Exploitation ~30-50
        grace_period=10, # * iter_for_complete_episodes, minimum number of training iterations
        reduction_factor=2, # factor to reduce the number of trials
    )

search_alg = BayesOptSearch(
    # search_space,
    metric=metric,
    mode=mode
    )

###########################################################################
os.environ["RAY_verbose_spill_logs"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
parser.add_argument("--stop-iters", type=int, default=10, help="Number of iterations to train.")
# parser.add_argument("--stop-reward", type=float, default=1000000, help="Reward at which we stop training.")
parser.add_argument("--policy", choices=["ppo", "dqn", "a2c", "a3c", "impala"], default="ppo", required=True, help="Policy to train.")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
parser.add_argument("--resume", action="store_true", help="Whether to resume training from a checkpoint.")
parser.add_argument("--tune", action="store_true", help="Whether to perform hyperparameter tuning.")
# parser.add_argument("--fine-tuning", action="store_true", help="Whether to fine-tune and train the best policy after hyperparameter tuning.")

args = parser.parse_args()

def env_creator(env_config):
    env = FSS_env(**env_config)
    return env
    
def setup_config(config, algo):
    if algo == "PPO":
        algo_config = PPOConfig()
    elif algo == "A3C":
        algo_config = A3CConfig()
    elif algo == "DQN":
        algo_config = DQNConfig()
    algo_config = algo_config.environment(env="FSS_env-v0", env_config=env_config, disable_env_checking=True)
    algo_config = algo_config.framework(args.framework)
    algo_config = algo_config.experimental(_enable_new_api_stack=True)
    algo_config = algo_config.training(
        lr=config.get("lr", 1e-6),  # Set default if not tuning
        gamma=config.get("gamma", 0.99),
        use_gae=True, 
        lambda_=config.get("lambda", 0.95),
        model={
            "fcnet_hiddens": [64, 64, 64],  # 128x3 crashes in LRZ AI
            "fcnet_activation": "relu",
        }
    )
    return algo_config
    
run_config = train.RunConfig(
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute=metric,
                    checkpoint_score_order=mode
                )
            )
            
scaling_config = ScalingConfig(
                num_workers=resources["num_learner_workers"],
                use_gpu=True,
                resources_per_worker={"CPU": resources["num_cpus_per_worker"], "GPU": resources["num_gpus_per_worker"]},
                trainer_resources={"CPU": resources["num_cpus_per_learner_worker"], "GPU": resources["num_gpus_per_learner_worker"]}
            )
            
tune_config = tune.TuneConfig(
        max_concurrent_trials=max_concurrent_trials,
        num_samples=num_samples_per_policy,
        search_alg=search_alg,
        scheduler=scheduler,
    )
            


# @ray.remote(accelerator_type=NVIDIA_A100)
def train_func(config):
    if args.policy == "ppo":
        algo = "PPO"
    elif args.policy == "a3c":
        algo = "A3C"
    elif args.policy == "dqn":
        algo = "DQN"

    # print(f"Starting training function for {algo}")
    
    algo_config = setup_config(config, algo)

    # Initialize the algorithm trainer
    trainer = algo_config.build()
    # print(f"Trainer built for {algo}")

    # Load checkpoint if available
    checkpoint = train.get_checkpoint()
    if checkpoint:
        print("Restoring from checkpoint")
        trainer.restore(checkpoint.path)

    # Training loop
    for i in range(args.stop_iters):
        # print(f"Starting iteration {i+1}/{args.stop_iters}")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_training"):
                result = trainer.train()
                tune.report(mean_reward=result[metric])
                # print(f"Iteration {i+1} completed, mean reward: {result[metric]}")

                if (i + 1) % checkpoint_frequency == 0:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        trainer.save_checkpoint(temp_checkpoint_dir)
                        train.report(
                            metrics={"mean_reward": result[metric]},
                            checkpoint=train.Checkpoint.from_directory(temp_checkpoint_dir)
                        )
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(os.path.join(args.checkpoint_dir, f"profile_trace_{i}.json"))

    print(f"Training function for {algo} completed")

def run_experiment(algo):
    """ trainer = DataParallelTrainer(
            train_loop_per_worker=lambda config: train_func(config, algo),
            scaling_config=scaling_config,
    ) """

    tuner = tune.Tuner(
        tune.with_resources(train_func, resources=tune.PlacementGroupFactory(
            [{'CPU': resources["num_cpus_per_learner_worker"]}, {'GPU': resources["num_gpus_per_learner_worker"]}] 
             + [{'CPU': resources["num_cpus_per_worker"]}] * resources["num_rollout_workers"]
            )),
        tune_config=tune_config,
        run_config=run_config,
        param_space=search_space,
    )

    results = tuner.fit()

    return results


def evaluate_best_model(results, algo):
    best_result = results.get_best_result(metric=metric, mode=mode)
    print(f"Best hyperparameters for {algo} found were: {best_result.config}")

    # You can restore the best checkpoint and continue training or evaluation
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        if algo == "PPO":
            trainer = PPOConfig().build()
        elif algo == "A3C":
            trainer = A3CConfig().build()
        elif algo == "DQN":
            trainer = DQNConfig().build()

        trainer.restore(checkpoint_dir)
        # Evaluate or continue training

def print_results(results):
    dfs = {result.path: result.metrics_dataframe for result in results}
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d[metric].plot(ax=ax, legend=False)
    plt.show()
    pretty_print(results)

register_env("FSS_env-v0", lambda config: env_creator(env_config))
print("Registered environment")

if args.tune:
    if args.policy == "ppo":
        results = run_experiment("PPO")
        print_results(results)
    if args.policy == "a3c":
        results = run_experiment("A3C")
        print_results(results)
    if args.policy == "dqn":
        results = run_experiment("DQN")
        print_results(results)
    else:
        print("Invalid policy specified. Please choose either 'ppo', 'a3c' or 'dqn'.")

if args.resume:
    # Resume training from a checkpoint
    if args.policy == "ppo":
        results = run_experiment("PPO")
        evaluate_best_model(results, "PPO")
    elif args.policy == "a3c":
        results = run_experiment("A3C")
        evaluate_best_model(results, "A3C")
    elif args.policy == "dqn":
        results = run_experiment("DQN")
        evaluate_best_model(results, "DQN")
    else:
        print("Invalid policy specified. Please choose either 'ppo', 'a3c' or 'dqn'.")