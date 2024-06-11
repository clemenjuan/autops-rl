import os
import time
import argparse
import ray
import torch
from ray import air, tune
from ray.tune import CLIReporter, Tuner, TuneConfig
import tempfile
from ray import train
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


##### EDIT THIS FUNCTION #####
# Common configuration setup
# Resource allocation settings
def setup_config(config):
    num_rollout_workers = 10 # Number of rollout workers (parallel actors for simulating environment interactions)
    num_envs_per_worker = 1 # Number of environments per worker
    num_cpus_per_worker = 1 # Number of CPUs per worker
    num_cpus_per_learner_worker = 1 # Number of CPUs per local worker (trainer) - just 1 and use GPU

    config.environment(env=env_name, env_config=env_config, disable_env_checking=True)
    config.framework(args.framework)
    config.rollouts(num_rollout_workers=num_rollout_workers, num_envs_per_worker=num_envs_per_worker, batch_mode="complete_episodes") #, rollout_fragment_length="auto")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    config.resources(num_gpus=gpu_count,
                     num_cpus_per_worker=num_cpus_per_worker, 
                     num_gpus_per_worker=0,
                     num_cpus_per_learner_worker=num_cpus_per_learner_worker, 
                     num_gpus_per_learner_worker=gpu_count)
    print(f"Using {gpu_count} GPU(s) for training.")
    return config

env_config = {
    "num_targets": 10, 
    "num_observers": 10, 
    "simulator_type": 'everyone', 
    "time_step": 1, 
    "duration": 24*60*60
}
###############################

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
parser.add_argument("--tune", action="store_true", help="Whether to perform hyperparameter tuning.")
parser.add_argument("--resume", action="store_true", help="Whether to resume training from a checkpoint.")
args = parser.parse_args()

def env_creator(env_config):
    env = FSS_env(**env_config)
    return env

# Register environment
env_name = "FSS_env-v0"

register_env(env_name, lambda config: env_creator(env_config))

ray.init(num_cpus=12, num_gpus=1)

# Serch space configurations
search_space = {
    "fcnet_hiddens": tune.choice([[64, 64], [128, 128], [256, 256]]),
    "lr": tune.loguniform(1e-5, 1e-3),
    "gamma": tune.uniform(0.9, 0.99),
    "lambda": tune.uniform(0.9, 1.0),
    "train_batch_size": tune.choice([512, 1024, 2048]),
}

# Configuration for PPO - https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md#
ppo_config = setup_config(PPOConfig())
ppo_config.training(
    vf_loss_coeff=0.01, 
    num_sgd_iter=10, # num_sgd_iter: Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
    train_batch_size=search_space["train_batch_size"] if args.tune else 512, 
    lr=search_space["lr"] if args.tune else 1e-3,  # Set a default value if not tuning
    gamma=search_space["gamma"] if args.tune else 0.99,
    use_gae=True, 
    lambda_=search_space["lambda"] if args.tune else 0.95,
    clip_param=0.2, 
    entropy_coeff=0.01, 
    sgd_minibatch_size=64,
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
        print(pretty_print(result))
        print(f"Time taken for training {policy_name.upper()}: ", time.time() - start_time)
        
        if i > 0 and i % 5 == 0:
            checkpoint = algorithm.save(checkpoint_dir)
            print(f"Checkpoint saved at {checkpoint}")

        if result["episode_reward_mean"] >= args.stop_reward:
            print(f"Stopping {policy_name.upper()} training as it reached the reward threshold.")
            break

def save_best_config(best_config, config_dir):
    best_config_path = os.path.join(config_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"Best configuration saved to {best_config_path}")

def save_hyperparameter_results(analysis, config_dir):
    trials_data = []
    for trial in analysis.trials:
        trials_data.append(trial.metric_analysis["episode_reward_mean"])

    df = pd.DataFrame(trials_data)
    hyperparam_results_path = os.path.join(config_dir, "hyperparameter_results.csv")
    df.to_csv(hyperparam_results_path, index=False)
    print(f"Hyperparameter search results saved to {hyperparam_results_path}")



def inspect_policy(config, policy, checkpoint_dir):
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
        print(f"Parameter name: {name}, Parameter details: {param.size()}")


def train_rl(config, algo_config_cls):
    algo_config = algo_config_cls().update_from_dict(config)
    algo = algo_config.build()
    
    # Load existing checkpoint if available
    if train.get_checkpoint():
        checkpoint = train.get_checkpoint()
        with checkpoint.as_directory() as checkpoint_dir:
            algo.restore_checkpoint(checkpoint_dir)
    
    for i in range(config["iterations"]):
        result = algo.train()
        tune.report(mean_reward=result["episode_reward_mean"])
        
        if i % 10 == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                algo.save_checkpoint(temp_checkpoint_dir)
                checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
                tune.report(mean_reward=result["episode_reward_mean"], checkpoint=checkpoint)
    
    algo.stop()


if args.tune:
    # Hyperparameter tuning setup
    # Jetson ~ 1M steps per day (1057531)
    reporter = CLIReporter(
        metric_columns=["episode_reward_mean", "episode_len_mean", "timesteps_total"]
    )
    scheduler = ASHAScheduler(
        max_t=200000,  # maximum timesteps per trial
        grace_period=10000,  # minimum timesteps before a trial can be stopped
        reduction_factor=2  # halving the number of trials
    )

    algorithm_config_cls = eval(f"{args.policy.upper()}Config")
    param_space = eval(f"{args.policy}_config").to_dict()
    param_space["iterations"] = args.stop_iters  # Number of iterations for each trial

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_rl, algo_config_cls=algorithm_config_cls),
            resources={"cpu": 1, "gpu": 1}  # adjust resources based on your setup
        ),
        tune_config=tune.TuneConfig(
            metric="mean_reward",
            mode="max",
            scheduler=scheduler,
            num_samples=30,
        ),
        param_space=param_space,
        run_config=train.RunConfig(
            local_dir=args.checkpoint_dir,
            name=f"{args.policy}_experiment",
            progress_reporter=reporter,
            # time_budget_s=3600 * 24 * 7  # total time budget in seconds (7 days)
        )
        resources_per_trial=PlacementGroupFactory(
            [{'CPU': 1.0, 'GPU': 1.0}] + [{'CPU': 1.0}] * (10)  # 10 rollout workers + 1 trainer
        )
    )

    results = tuner.fit()

    # Get the best hyperparameters
    best_result = results.get_best_result("mean_reward", "max")
    best_config = best_result.config
    print(f"Best config: {best_config}")

    # Save the best configuration and hyperparameter search results
    save_best_config(best_config, args.checkpoint_dir)
    save_hyperparameter_results(results, args.checkpoint_dir)

    # Train the policy with the best hyperparameters
    best_policy_config = eval(f"{args.policy}_config").update_from_dict(best_config)
    train_policy(best_policy_config, f"{args.policy}_policy", args.checkpoint_dir)
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
