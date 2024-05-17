import os
import time
import argparse
import ray
import torch
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from FSS_env import FSS_env


'''
Edit the common configuration setup function to include gpu resources or add more parallelism to the training process. 
#### Usage ##############################
To perform hyperparameter tuning and training for different policies, run the following commands:
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy ppo --checkpoint-dir ppo_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a2c --checkpoint-dir a2c_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a3c --checkpoint-dir a3c_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy impala --checkpoint-dir impala_checkpoints --tune

To train (or keep training from last checkpoint) the policies without tuning, omit the --tune argument:
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy ppo --checkpoint-dir ppo_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a2c --checkpoint-dir a2c_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a3c --checkpoint-dir a3c_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy impala --checkpoint-dir impala_checkpoints
'''

##### EDIT THIS FUNCTION #####
# Common configuration setup
def setup_config(config):
    config.environment(env=env_name, env_config=env_config, disable_env_checking=True)
    config.framework(args.framework)
    config.rollouts(num_rollout_workers=4, num_envs_per_worker=2, rollout_fragment_length="auto", batch_mode="complete_episodes")
    config.resources(num_gpus=1 if torch.cuda.is_available() else 0)
    print(f"Using {config.resources['num_gpus']} GPU(s) for training.")
    return config

env_config = {
    "num_targets": 5, 
    "num_observers": 5, 
    "simulator_type": 'everyone', 
    "time_step": 1, 
    "duration": 24*60*60
}
###############################

os.environ["RAY_verbose_spill_logs"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
parser.add_argument("--stop-iters", type=int, default=20, help="Number of iterations to train.")
parser.add_argument("--stop-reward", type=float, default=500000.0, help="Reward at which we stop training.")
parser.add_argument("--policy", choices=["ppo", "dqn", "a2c", "a3c", "impala"], required=True, help="Policy to train.")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
parser.add_argument("--tune", action="store_true", help="Whether to perform hyperparameter tuning.")
args = parser.parse_args()

def env_creator(env_config):
    env = FSS_env(**env_config)
    return env

# Register environment
env_name = "FSS_env-v0"

register_env(env_name, lambda config: env_creator(env_config))


# Configuration for PPO
ppo_config = setup_config(PPOConfig())
ppo_config.training(
    vf_loss_coeff=0.01, num_sgd_iter=6, train_batch_size=env_config["duration"],
    lr=tune.loguniform(1e-4, 1e-2) if args.tune else 1e-3,  # Set a default value if not tuning
    gamma=tune.uniform(0.9, 0.99) if args.tune else 0.99,
    use_gae=True, lambda_=tune.uniform(0.9, 1.0) if args.tune else 0.95,
    clip_param=0.2, entropy_coeff=0.01, sgd_minibatch_size=64,
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

# Function to get the latest checkpoint path
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir)]
    checkpoints = [path for path in checkpoints if os.path.isdir(path)]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

# Function to train a policy
def train_policy(config, policy_name, checkpoint_dir):
    algorithm = config.build()

    checkpoint_path = os.path.join(checkpoint_dir, policy_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    latest_checkpoint = get_latest_checkpoint(checkpoint_path)

    if latest_checkpoint:
        algorithm.restore(latest_checkpoint)

    for i in range(args.stop_iters):
        print(f"== {policy_name.upper()} Iteration {i} ==")
        start_time = time.time()
        result = algorithm.train()
        print(pretty_print(result))
        print(f"Time taken for training {policy_name.upper()}: ", time.time() - start_time)
        
        checkpoint = algorithm.save(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint}")

        if result["episode_reward_mean"] >= args.stop_reward:
            print(f"Stopping {policy_name.upper()} training as it reached the reward threshold.")
            break

if args.tune:
    if args.policy == "ppo":
        analysis = tune.run(
            "PPO",
            config=ppo_config.to_dict(),
            num_samples=10,
            metric="episode_reward_mean",
            mode="max",
            local_dir=args.checkpoint_dir,
            name="ppo_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "dqn":
        analysis = tune.run(
            "DQN",
            config=dqn_config.to_dict(),
            num_samples=10,
            metric="episode_reward_mean",
            mode="max",
            local_dir=args.checkpoint_dir,
            name="dqn_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "a2c":
        analysis = tune.run(
            "A2C",
            config=a2c_config.to_dict(),
            num_samples=10,
            metric="episode_reward_mean",
            mode="max",
            local_dir=args.checkpoint_dir,
            name="a2c_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "a3c":
        analysis = tune.run(
            "A3C",
            config=a3c_config.to_dict(),
            num_samples=10,
            metric="episode_reward_mean",
            mode="max",
            local_dir=args.checkpoint_dir,
            name="a3c_experiment",
            checkpoint_at_end=False
        )
    elif args.policy == "impala":
        analysis = tune.run(
            "IMPALA",
            config=impala_config.to_dict(),
            num_samples=10,
            metric="episode_reward_mean",
            mode="max",
            local_dir=args.checkpoint_dir,
            name="impala_experiment",
            checkpoint_at_end=False
        )

    # Get the best hyperparameters
    best_config = analysis.best_config
    print(f"Best config: {best_config}")

    # Train the policy with the best hyperparameters
    if args.policy == "ppo":
        ppo_config.update_from_dict(best_config)
        train_policy(ppo_config, "ppo_policy", args.checkpoint_dir)
    elif args.policy == "dqn":
        dqn_config.update_from_dict(best_config)
        train_policy(dqn_config, "dqn_policy", args.checkpoint_dir)
    elif args.policy == "a2c":
        a2c_config.update_from_dict(best_config)
        train_policy(a2c_config, "a2c_policy", args.checkpoint_dir)
    elif args.policy == "a3c":
        a3c_config.update_from_dict(best_config)
        train_policy(a3c_config, "a3c_policy", args.checkpoint_dir)
    elif args.policy == "impala":
        impala_config.update_from_dict(best_config)
        train_policy(impala_config, "impala_policy", args.checkpoint_dir)
else:
    if args.policy == "ppo":
        train_policy(ppo_config, "ppo_policy", args.checkpoint_dir)
    elif args.policy == "dqn":
        train_policy(dqn_config, "dqn_policy", args.checkpoint_dir)
    elif args.policy == "a2c":
        train_policy(a2c_config, "a2c_policy", args.checkpoint_dir)
    elif args.policy == "a3c":
        train_policy(a3c_config, "a3c_policy", args.checkpoint_dir)
    elif args.policy == "impala":
        train_policy(impala_config, "impala_policy", args.checkpoint_dir)
