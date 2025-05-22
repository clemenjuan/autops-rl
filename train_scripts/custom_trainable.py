import os
import argparse
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.schedulers import ASHAScheduler
import torch
from ray.tune.stopper import (
             CombinedStopper,
             MaximumIterationStopper,
        )
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule # Deprecated
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    TorchRLModule,
    DefaultPPOTorchRLModule
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
import wandb
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import json

# Import your environment
from src.envs.FSS_env_v1 import FSS_env  # Make sure you're importing the correct version

# Get Wandb API key from environment variable
WANDB_API_KEY = "4b5c9c4ae3ffb150f67942dec8cc7d9f6fbcd558"     # fail fast if missing. define with export WANDB_API_KEY=...
WANDB_PROJECT = "autops-rl"
WANDB_ENTITY  = "sps-tum"      # team slug only (sps-tum)

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Detected {gpu_count} GPUs")

ray.init(
    num_gpus=gpu_count,
    runtime_env={"env_vars": {
    "WANDB_API_KEY": WANDB_API_KEY}})

# Get SLURM job ID if available
slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
metric = "episode_return_mean" # "episode_reward_mean" in old API

# Create environment
def env_creator(env_config):
    return FSS_env(env_config)


def _env_to_module_pipeline(env):
    return FlattenObservations(
        input_observation_space=env.observation_space,
        input_action_space=env.action_space,
        multi_agent=True
    )

def main(args):
    # Parse seeds
    seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
    
    # Parse simulator types
    if args.simulator_types:
        simulator_types = [sim_type.strip() for sim_type in args.simulator_types.split(",")]
    else:
        simulator_types = [args.simulator_type]
    
    # At the beginning of the main function, add:
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    
    for simulator_type in simulator_types:
        for seed in seeds:
            print("\n\n================================================================================")
            print(f"Running with simulator_type: {simulator_type}, seed: {seed}")
            print("================================================================================\n")
            
            # Environment configuration
            env_config = {
                "num_targets": args.num_targets,
                "num_observers": args.num_observers,
                "time_step": args.time_step,
                "duration": args.duration,
                "simulator_type": simulator_type,
                "seed": seed,
                "reward_type": args.reward_type,
            }
            
            # If reward_config is provided, parse it as JSON
            if args.reward_config:
                env_config["reward_config"] = json.loads(args.reward_config)
            
            # Create experiment name
            experiment_name = f"{args.policy}_{simulator_type}"
            if slurm_job_id != "local":
                experiment_name += f"_job{slurm_job_id}"
            else:
                experiment_name += "_local"
            
            # Create checkpoint directory
            checkpoint_dir = os.path.join(args.checkpoint_dir, simulator_type)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Configure algorithm
            if args.policy == "PPO":
                algo_config = configure_ppo(args, env_config)
            else:
                raise ValueError(f"Unsupported policy: {args.policy}")
            
            # Configure WandbLoggerCallback
            wandb_config = {
                "project": WANDB_PROJECT,
                "entity": WANDB_ENTITY
            }
            
            # Run experiment based on mode
            if args.mode == "tune":
                # Only do hyperparameter tuning
                print("Running hyperparameter tuning...")
                run_hyperparameter_tuning(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config, simulator_type)
                
            elif args.mode == "train":
                # Only do training
                print("Running training...")
                run_training(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config, simulator_type)
                
            elif args.mode == "tune_then_train":
                # First tune, then train with the best config
                print("Running hyperparameter tuning followed by training...")
                best_config = run_hyperparameter_tuning(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config, simulator_type)
                
                # Update algo_config with the best parameters from tuning
                if best_config:
                    print("Using best hyperparameters from tuning for training")
                    # Convert best_config to algo_config
                    if args.policy == "PPO":
                        algo_config = configure_ppo(args, env_config)
                        # Update with best parameters (handling nested dicts)
                        if "training" in best_config:
                            for key, value in best_config["training"].items():
                                algo_config.training(**{key: value})
                
                # Run training with the tuned config
                run_training(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config, simulator_type)
            else:
                raise ValueError(f"Unknown mode: {args.mode}. Choose from 'tune', 'train', or 'tune_then_train'")

def configure_ppo(args, env_config):
    env = env_creator(env_config)

    # Create PPO config
    algo_config = (
        PPOConfig()
        .environment("FSS_env", env_config=env_config)
        .framework("torch",
                   torch_compile_learner=True)
        # Configure learners (replaces num_gpus)
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner,
            num_cpus_per_learner=args.num_cpus_per_learner
        )
        # Configure environment runners (replaces num_workers)
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_runner,
            num_cpus_per_env_runner=args.num_cpus_per_runner,
            num_gpus_per_env_runner=args.num_gpus_per_runner,
            explore=True,
            env_to_module_connector=_env_to_module_pipeline,
            rollout_fragment_length=args.rollout_fragment_length,
            batch_mode=args.batch_mode,
            sample_timeout_s=None
        )
        .rl_module(
            # We need to explicitly specify here RLModule to use
                rl_module_spec=RLModuleSpec(
                module_class=PPOTorchRLModule,
                model_config={
                    "head_fcnet_hiddens": [256,256],
                    "head_fcnet_activation": "relu",
                },
            ),
        )
        .multi_agent(
            policies={"autops-rl_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "autops-rl_policy",
        )
        .training(
            # Learning rate can be a schedule or a fixed value
            lr=args.lr * (args.num_learners ** 0.5),
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            gamma=args.gamma,
            lambda_=args.lambda_val,
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
    )
    
    return algo_config



def run_hyperparameter_tuning(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config, simulator_type):
    # Configure hyperparameter search space
    if args.policy == "PPO":
        param_space = {
            "training": {
                "lr": tune.loguniform(1e-5, 1e-3),
                "gamma": tune.uniform(0.9, 0.999),
                "lambda_": tune.uniform(0.9, 1.0),
                "train_batch_size_per_learner": tune.choice([4096, 8192, 16384]),
                "minibatch_size": tune.choice([256, 512, 1024])
            },
            "env_runners": {
                "rollout_fragment_length": tune.choice([128, 256, 512])
            }
        }
    # Configure ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        max_t=args.max_iterations_hyperparameter_tuning,
        grace_period=args.grace_period_hyperparameter_tuning,
        reduction_factor=2
    )
    
    # Configure WandbLoggerCallback with comprehensive logging
    wandb_logger = WandbLoggerCallback(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        api_key=WANDB_API_KEY,
        log_config=True,  # Log configuration parameters
    )
    
    # Run hyperparameter tuning
    tuner = tune.Tuner(
        algo_config.algo_class,
        param_space=algo_config.to_dict() | param_space,
        run_config=ray.tune.RunConfig(
            name=f"{experiment_name}_tune",
            storage_path=checkpoint_dir,
            callbacks=[wandb_logger],
            stop={"training_iteration": args.max_iterations_hyperparameter_tuning},
            checkpoint_config=ray.tune.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq if hasattr(args, "checkpoint_freq") else 10,
                checkpoint_at_end=True,
                num_to_keep=5,  # Keep the last 5 checkpoints
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.num_samples_hyperparameter_tuning,
            metric="env_runners/episode_return_mean",
            mode="max",
        ),
    )
    
    results = tuner.fit()
    
    # Get the best result and directly report metrics
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max",
    )
    best_score = best_result.metrics["env_runners"]["episode_return_mean"]
    best_ckpt = best_result.checkpoint
    print("Best score:", best_score, "\nBest checkpoint:", best_ckpt)

    # Save checkpoint with reward case and seed info
    reward_type = algo_config.env_config.get("reward_type", "unknown")
    seed = algo_config.env_config.get("seed", "unknown")
    simulator_type = algo_config.env_config.get("simulator_type", "unknown")
    best_ckpt_path = os.path.join(checkpoint_dir, f"best_{reward_type}_seed{seed}_sim_{simulator_type}_tune.ckpt")
    best_ckpt.to_directory(best_ckpt_path)
    print(f"✓  Saved best tuned checkpoint to {best_ckpt_path} (reward_type: {reward_type}, seed: {seed}, simulator_type: {simulator_type})")

    return best_result.config


def run_training(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config, simulator_type):
    # Configure training
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=args.iterations),
    )
    
    # Configure WandbLoggerCallback with comprehensive logging
    wandb_logger = WandbLoggerCallback(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        api_key=WANDB_API_KEY,
        log_config=True,  # Log configuration parameters
    )

    # Run training
    tuner = tune.Tuner(
        algo_config.algo_class,
        param_space=algo_config.to_dict(),
        run_config=ray.tune.RunConfig(
            name=experiment_name,
            storage_path=checkpoint_dir,
            callbacks=[wandb_logger],
            stop=stopper,
            checkpoint_config=ray.tune.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq if hasattr(args, "checkpoint_freq") else 10,
                checkpoint_at_end=True,
                num_to_keep=5,  # Keep the last 5 checkpoints
            ),
        ),
    )
    
    results = tuner.fit()
    
    # Get the best result and directly report metrics
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max",
    )
    best_score = best_result.metrics["env_runners"]["episode_return_mean"]
    best_ckpt = best_result.checkpoint
    print("Best score:", best_score, "\nBest checkpoint:", best_ckpt)
    
    # Save checkpoint with reward case and seed info
    reward_type = algo_config.env_config.get("reward_type", "unknown")
    seed = algo_config.env_config.get("seed", "unknown")
    simulator_type = algo_config.env_config.get("simulator_type", "unknown")
    best_ckpt_path = os.path.join(checkpoint_dir, f"best_{reward_type}_seed{seed}_sim_{simulator_type}.ckpt")
    best_ckpt.to_directory(best_ckpt_path)
    print(f"✓  Saved best checkpoint to {best_ckpt_path} (reward_type: {reward_type}, seed: {seed}, simulator_type: {simulator_type})")
    
    return results

register_env("FSS_env", lambda config: env_creator(config))
print("Registered environment")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Training configuration arguments
    parser.add_argument("--policy", type=str, default="PPO", help="RL algorithm to use")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--mode", type=str, default="train", choices=["tune", "train", "tune_then_train"],
                       help="Operation mode: 'tune' for hyperparameter tuning only, 'train' for training only, or 'tune_then_train' to do both")
    parser.add_argument("--tune", action="store_true", help="DEPRECATED: Use --mode=tune instead")
    parser.add_argument("--num-samples-hyperparameter-tuning", type=int, default=20, help="Number of hyperparameter samples to run")
    parser.add_argument("--max-iterations-hyperparameter-tuning", type=int, default=25, help="Maximum number of training iterations")
    parser.add_argument("--grace-period-hyperparameter-tuning", type=int, default=10, help="Grace period for early stopping")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Frequency of checkpoints during training")
    parser.add_argument("--resume", action="store_true", help="Resume tuning from the specified checkpoint directory")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file for continuing training")
    parser.add_argument("--best-config", type=str, help="Path to best config JSON file for continuing training")
    parser.add_argument("--simulator-types", type=str, help="Comma-separated list of simulator types to run (e.g., 'everyone,centralized,decentralized')")
    parser.add_argument("--simulator-type", type=str, default="everyone", help="Default simulator type")
    parser.add_argument("--seeds", type=str, default="42, 43, 44, 45, 46", help="Comma-separated list of seeds to run (e.g., '0,1,2,3,4')")
    
    # Environment configuration arguments
    parser.add_argument("--num-targets", type=int, default=20, help="Number of targets")
    parser.add_argument("--num-observers", type=int, default=20, help="Number of observers")
    parser.add_argument("--time-step", type=int, default=1, help="Time step for simulation")
    parser.add_argument("--duration", type=int, default=86400, help="Duration of simulation in seconds")
    
    # Runner and learner configuration
    parser.add_argument("--num-env-runners", type=int, default=32, help="Number of environment runners")
    parser.add_argument("--num-envs-per-runner", type=int, default=1, help="Number of environments per runner")
    parser.add_argument("--num-cpus-per-runner", type=int, default=2, help="Number of CPUs per runner")
    parser.add_argument("--num-gpus-per-runner", type=float, default=0, help="Number of GPUs per runner")
    parser.add_argument("--num-learners", type=int, default=1, help="Number of learners")
    parser.add_argument("--num-gpus-per-learner", type=float, default=1, help="Number of GPUs per learner")
    parser.add_argument("--num-cpus-per-learner", type=int, default=1, help="Number of CPUs per learner")
    
    # Batch configuration arguments
    parser.add_argument("--train-batch-size", type=int, default=8192, help="Total size of batches for policy updates per learner")
    parser.add_argument("--minibatch-size", type=int, default=512, help="Size of minibatches for SGD updates")
    parser.add_argument("--rollout-fragment-length", type=int, default=256, help="Steps collected per worker before sending")
    parser.add_argument("--batch-mode", type=str, default="truncate_episodes", choices=["truncate_episodes", "complete_episodes"],
                        help="How to build batches: 'truncate_episodes' for partial episodes, 'complete_episodes' for full episodes only")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda_val", type=float, default=0.95, help="GAE lambda parameter")
    
    # Reward configuration arguments
    parser.add_argument("--reward-type", type=str, default="case1", 
                       help="Type of reward function to use (case1, case2, case3, or case4)")
    parser.add_argument("--reward-config", type=str, default=None,
                       help="JSON string with reward function parameters")
    
    args = parser.parse_args()
    
    # Handle backward compatibility with --tune flag
    if args.tune and args.mode == "train":
        print("Warning: --tune flag is deprecated. Using --mode=tune instead")
        args.mode = "tune"

    # Start training
    main(args)