import os
import time
import argparse
import ray
import torch
from ray import air, tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from FSS_env import FSS_env
import json
import pandas as pd
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0


##### EDIT HERE ##########################################################
# Environment configurations
env_config = {
    "num_targets": 10, 
    "num_observers": 10, 
    "simulator_type": 'everyone', 
    "time_step": 1, 
    "duration": 24*60*60
}

epochs = 40 # per training iteration - increase gradually when training
metric = "info/learner/default_policy/learner_stats/total_loss" # "episode_reward_mean"
mode = "min"
num_parallel_trainers = 1
rollout_fragment_length = "auto"

# Resource allocation settings
# GPUs are automatically detected and used if available
resources = {
    "num_rollout_workers" : 10, # Number of rollout workers (parallel actors for simulating environment interactions)
    "num_envs_per_worker" : 4, # Number of environments per worker
    "num_cpus_per_worker" : 1, # Number of CPUs per worker
    "num_gpus_per_worker" : 0, # Number of GPUs per worker - only CPU simulations
    "num_learner_workers" : num_parallel_trainers, # Number of learner workers (parallel actors for training)
    "num_cpus_per_learner_worker" : 1, # Number of CPUs per local worker (trainer) =1!!!!! 
    "num_gpus_per_learner_worker" : gpu_count/num_parallel_trainers, # Number of GPUs per local worker (trainer)
}

# Serch space configurations
search_space = {
    "fcnet_hiddens": tune.choice([[128, 128], [256, 256], [64, 64, 64]]),
    "num_sgd_iter": tune.choice(10, 30, 50),
    "lr": tune.loguniform(1e-5, 1e-3),
    "gamma": tune.uniform(0.9, 0.99),
    "lambda": tune.uniform(0.9, 1.0),
    "train_batch_size": tune.choice([10000, 20000, 40000]),
    "sgd_minibatch_size": tune.choice([128, 512, 2048]),
}

# Hyperparameter search
num_samples_per_policy = 30
max_concurrent_trials = 5

# Scheduler - Jetson ~16 iterations per day - 90 mins per iteration (complete episodes)
scheduler = ASHAScheduler(
        metric=metric,
        mode=mode, # maximize the reward
        max_t=20, # maximum number of training iterations - Exploration ~10-20, Exploitation ~30-50
        grace_period=10, # minimum number of training iterations
        reduction_factor=2, # factor to reduce the number of trials
    )

###########################################################################






os.environ["RAY_verbose_spill_logs"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
parser.add_argument("--stop-iters", type=int, default=50, help="Number of iterations to train.")
parser.add_argument("--stop-reward", type=float, default=1000000, help="Reward at which we stop training.")
parser.add_argument("--policy", choices=["ppo", "dqn", "a2c", "a3c", "impala"], default="ppo", required=True, help="Policy to train.")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
parser.add_argument("--resume", action="store_true", help="Whether to resume training from a checkpoint.")
parser.add_argument("--tune", action="store_true", help="Whether to perform hyperparameter tuning.")
parser.add_argument("--fine-tuning", action="store_true", help="Whether to fine-tune and train the best policy after hyperparameter tuning.")

args = parser.parse_args()

def env_creator(env_config):
    env = FSS_env(**env_config)
    return env

def setup_config(config):
    config.environment(env=env_name, env_config=env_config, disable_env_checking=True)
    config.framework(args.framework)
    config.rollouts(num_rollout_workers=resources["num_rollout_workers"],
                     num_envs_per_worker=resources["num_envs_per_worker"],
                     batch_mode="truncate_episodes",
                     rollout_fragment_length=rollout_fragment_length)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    config.resources(num_gpus=gpu_count,
                     num_cpus_per_worker=resources["num_cpus_per_worker"], 
                     num_gpus_per_worker=resources["num_gpus_per_worker"],
                     num_learner_workers=resources["num_learner_workers"],
                     num_cpus_per_learner_worker=resources["num_cpus_per_learner_worker"], 
                     num_gpus_per_learner_worker=resources["num_gpus_per_learner_worker"]
                     )
    print(f"Using {gpu_count} GPU(s) for training.")
    return config

# Register environment
env_name = "FSS_env-v0"

register_env(env_name, lambda config: env_creator(env_config))

ray.init(num_gpus=gpu_count)

# Configuration for PPO - https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md#
ppo_config = setup_config(PPOConfig())
ppo_config.training(
    vf_loss_coeff=0.01, 
    num_sgd_iter=search_space["num_sgd_iter"] if args.tune else epochs, # number of epochs to execute per iteration
    train_batch_size=search_space["train_batch_size"] if args.tune else 10000, 
    lr=search_space["lr"] if args.tune else 1e-3,  # Set a default value if not tuning
    gamma=search_space["gamma"] if args.tune else 0.99,
    use_gae=True, 
    lambda_=search_space["lambda"] if args.tune else 0.95,
    clip_param=0.2, 
    entropy_coeff=0.01, 
    sgd_minibatch_size=search_space["sgd_minibatch_size"] if args.tune else 128,
    model={
        "fcnet_hiddens": search_space["fcnet_hiddens"] if args.tune else [64, 64],
        "fcnet_activation": "relu",
    }
)

# Configuration for DQN
dqn_config = setup_config(DQNConfig())
dqn_config.training(
    n_step=3,
    lr=tune.loguniform(1e-4, 1e-2) if args.tune else 1e-3,
    gamma=tune.uniform(0.9, 0.99) if args.tune else 0.99,
)

# Configuration for A2C
a2c_config = setup_config(A2CConfig())
a2c_config.training(
    lr=tune.loguniform(1e-4, 1e-2) if args.tune else 1e-3,
    gamma=tune.uniform(0.9, 0.99) if args.tune else 0.99,
    sample_async=False,
)

# Configuration for A3C
a3c_config = setup_config(A3CConfig())
a3c_config.training(
    lr=tune.loguniform(1e-4, 1e-2) if args.tune else 1e-3,
    gamma=tune.uniform(0.9, 0.99) if args.tune else 0.99,
    sample_async=False,
)

# Configuration for IMPALA
impala_config = setup_config(ImpalaConfig())
impala_config.training(
    lr=tune.loguniform(1e-4, 1e-2) if args.tune else 1e-3,
    gamma=tune.uniform(0.9, 0.99) if args.tune else 0.99,
)

# Function to train a policy
def train_policy(config, policy_name, checkpoint_dir):
    algorithm = config.build()

    # Set paths
    algorithm_checkpoint_path = os.path.join(checkpoint_dir, policy_name)
    os.makedirs(algorithm_checkpoint_path, exist_ok=True)

    for i in range(args.stop_iters):
        print(f"== {policy_name.upper()} Iteration {i} ==")
        start_time = time.time()
        result = algorithm.train()

        print(pretty_print(result))
        print(f"Time taken for training {policy_name.upper()}: ", time.time() - start_time)
        
        if i>0 and i % 5 == 0:
            checkpoint = algorithm.save(algorithm_checkpoint_path)
            print(f"Checkpoint saved at {checkpoint}")

        if result["episode_reward_mean"] >= args.stop_reward:
            print(f"Stopping {policy_name.upper()} training as it reached the reward threshold.")
            break

def train_policy_from_checkpoint(config, policy_name, checkpoint_dir, algorithm_checkpoint_path):
    # Initialize the algorithm
    algorithm = config.build()

    # Restore the algorithm from the last checkpoint
    algorithm.restore(algorithm_checkpoint_path)
    print(f"Restored algorithm from checkpoint: {algorithm_checkpoint_path}")

    # Validate the restoration by checking the state of the algorithm
    restored_policy = algorithm.get_policy()
    print("Restored policy configuration: ", restored_policy.config)

    for i in range(args.stop_iters):
        print(f"== {policy_name.upper()} Iteration {i} ==")
        start_time = time.time()
        result = algorithm.train()

        # Log the loss function
        if 'learner' in result['info']:
            print(f"Loss: {result['info']['learner']['default_policy']['learner_stats']['loss']}")

        print(pretty_print(result))
        print(f"Time taken for training {policy_name.upper()}: ", time.time() - start_time)
        
        if i > 0 and i % 5 == 0:
            checkpoint = algorithm.save(checkpoint_dir)
            print(f"Checkpoint saved at {checkpoint}")

        if result["episode_reward_mean"] >= args.stop_reward:
            print(f"Stopping {policy_name.upper()} training as it reached the reward threshold.")
            break

def save_best_config(best_config, config_dir):
    # Ensure the directory exists
    os.makedirs(config_dir, exist_ok=True)
    
    # Filter out non-serializable parts of the configuration
    def make_serializable(config):
        if isinstance(config, dict):
            return {k: make_serializable(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [make_serializable(v) for v in config]
        elif isinstance(config, set):
            return list(make_serializable(v) for v in config)
        elif hasattr(config, '__dict__'):
            return make_serializable(vars(config))
        else:
            return config
    
    serializable_config = make_serializable(best_config)

    best_config_path = os.path.join(config_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(serializable_config, f, indent=4)
    print(f"Best configuration saved to {best_config_path}")

def save_hyperparameter_results(analysis, config_dir):
    trials_data = []
    for trial in analysis.trials:
        trial_result = {
            "episode_reward_mean": trial.metric_analysis["episode_reward_mean"],
            "loss": trial.metric_analysis.get("learner/default_policy/learner_stats/loss", None)
        }
        trials_data.append(trial_result)

    df = pd.DataFrame(trials_data)

    # Ensure the directory exists
    os.makedirs(config_dir, exist_ok=True)
    hyperparam_results_path = os.path.join(config_dir, "hyperparameter_results.csv")
    df.to_csv(hyperparam_results_path, index=False)
    print(f"Hyperparameter search results saved to {hyperparam_results_path}")


if args.tune:
    if args.policy == "ppo":
        analysis = tune.run(
            "PPO",
            config=ppo_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="ppo_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "dqn":
        analysis = tune.run(
            "DQN",
            config=dqn_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="dqn_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "impala":
        analysis = tune.run(
            "IMPALA",
            config=impala_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="impala_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "a2c":
        analysis = tune.run(
            "A2C",
            config=a2c_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="a2c_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "a3c":
        analysis = tune.run(
            "A3C",
            config=a3c_config.to_dict(),
            num_samples=num_samples_per_policy,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            local_dir=args.checkpoint_dir,
            name="a3c_experiment",
            checkpoint_at_end=False
        )

    # Get the best hyperparameters
    best_result = analysis.get_best_config(metric, mode)
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final average reward: {}".format(
        best_result.metrics["episode_reward_mean"]))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))


    # Train the policy with the best hyperparameters after exploration search
    if args.fine-tuning:
        if args.policy == "ppo":
            train_policy(best_result, "ppo_policy", args.checkpoint_dir)
        elif args.policy == "dqn":
            train_policy(best_result, "dqn_policy", args.checkpoint_dir)
        elif args.policy == "impala":
            train_policy(best_result, "impala_policy", args.checkpoint_dir)
        elif args.policy == "a2c":
            train_policy(best_result, "a2c_policy", args.checkpoint_dir)
        elif args.policy == "a3c":
            train_policy(best_result, "a3c_policy", args.checkpoint_dir)
elif args.resume:
    if args.policy == "ppo":
        algorithm_path = os.path.join(args.checkpoint_dir, "ppo_policy")
        policy_checkpoint_path = os.path.join(algorithm_path, "policies", "default_policy")
        train_policy_from_checkpoint(ppo_config,"ppo_policy", args.checkpoint_dir, algorithm_path)
    elif args.policy == "dqn":
        algorithm_path = os.path.join(args.checkpoint_dir, "dqn_policy")
        policy_checkpoint_path = os.path.join(algorithm_path, "policies", "default_policy")
        train_policy_from_checkpoint(dqn_config, "dqn_policy", args.checkpoint_dir, algorithm_path)
    elif args.policy == "impala":
        algorithm_path = os.path.join(args.checkpoint_dir, "impala_policy")
        policy_checkpoint_path = os.path.join(algorithm_path, "policies", "default_policy")
        train_policy_from_checkpoint(impala_config, "impala_policy", args.checkpoint_dir, algorithm_path)
    elif args.policy == "a2c":
        algorithm_path = os.path.join(args.checkpoint_dir, "a2c_policy")
        policy_checkpoint_path = os.path.join(algorithm_path, "policies", "default_policy")
        train_policy_from_checkpoint(a2c_config, "a2c_policy", args.checkpoint_dir, algorithm_path)
    elif args.policy == "a3c":
        algorithm_path = os.path.join(args.checkpoint_dir, "a3c_policy")
        policy_checkpoint_path = os.path.join(algorithm_path, "policies", "default_policy")
        train_policy_from_checkpoint(a3c_config, "a3c_policy", args.checkpoint_dir, algorithm_path)
else:
    if args.policy == "ppo":
        # inspect_policy(ppo_config, "ppo_policy", args.checkpoint_dir)
        train_policy(ppo_config, "ppo_policy", args.checkpoint_dir)
    elif args.policy == "dqn":
        # inspect_policy(dqn_config, "dqn_policy", args.checkpoint_dir)
        train_policy(dqn_config, "dqn_policy", args.checkpoint_dir)
    elif args.policy == "impala":
        # inspect_policy(impala_config, "impala_policy", args.checkpoint_dir)
        train_policy(impala_config, "impala_policy", args.checkpoint_dir)
    elif args.policy == "a2c":
        # inspect_policy(a2c_config, "a2c_policy", args.checkpoint_dir)
        train_policy(a2c_config, "a2c_policy", args.checkpoint_dir)
    elif args.policy == "a3c":
        # inspect_policy(a3c_config, "a3c_policy", args.checkpoint_dir)
        train_policy(a3c_config, "a3c_policy", args.checkpoint_dir)


""" def inspect_policy(config, policy, checkpoint_dir):
    algorithm = config.build()

    algorithm_checkpoint_path = os.path.join(checkpoint_dir, policy_name)

    # Restore the algorithm from the last checkpoint
    algorithm.restore(algorithm_checkpoint_path)
    print(f"Restored algorithm from checkpoint: {algorithm_checkpoint_path}")

    # Validate the restoration by checking the state of the algorithm
    restored_policy = algorithm.get_policy()
    print("Restored policy configuration: ", restored_policy.config)
    
    if latest_checkpoint:
        algorithm.restore(latest_checkpoint)
    
    # Get the policy object
    policy = algorithm.get_policy()
    
    # Access the model
    model = policy.model
    
    # Print the model's structure
    # print(model)

    # Print details of each layer and parameters
    print("\nModel's named children (layers):")
    for name, child in model.named_children():
        print(f"Layer name: {name}, Layer details: {child}")
    
    print("\nModel's named parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Parameter details: {param.size()}") """
