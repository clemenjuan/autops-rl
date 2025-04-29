import os
import argparse
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
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

# Import your environment
from src.envs.FSS_env import FSS_env

# Get Wandb API key from environment variable
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "4b5c9c4ae3ffb150f67942dec8cc7d9f6fbcd558")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "autops-rl")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "TUM")

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Detected {gpu_count} GPUs")

# Explicitly login to wandb using the API key
try:
    wandb.login(key=WANDB_API_KEY)
except Exception as e:
    print(f"Warning: Failed to log in to wandb: {e}")

ray.init(
    num_gpus=gpu_count,
    runtime_env={"env_vars": {"WANDB_API_KEY": WANDB_API_KEY}}
)

# Get SLURM job ID if available
slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

# Create environment
def env_creator(env_config):
    return FSS_env(**env_config)

# Define env-to-module-connector pipeline for the new stack.
# def _env_to_module_pipeline(env):
#     return FlattenObservations(obs_space=env.observation_space, act_space=env.action_space)

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
            }
            
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
                "entity": WANDB_ENTITY,
                "name": f"{experiment_name}_seed{seed}",
                "group": experiment_name,
            }
            
            # Run experiment
            if args.tune:
                run_hyperparameter_tuning(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config)
            else:
                run_training(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config)

def configure_ppo(args, env_config):
    env = env_creator(env_config)

    # Create PPO config
    algo_config = (
        PPOConfig()
        .environment("FSS_env", env_config=env_config)
        .framework("torch")
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
            # env_to_module_connector=_env_to_module_pipeline,
        )
        .rl_module(
            # We need to explicitly specify here RLModule to use
                rl_module_spec=RLModuleSpec(
                module_class=PPOTorchRLModule,
                model_config={
                    "head_fcnet_hiddens": [64, 64],
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
            lr=1e-5 * (args.num_learners ** 0.5),
        )
    )
    
    return algo_config


def run_hyperparameter_tuning(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config):
    # Configure hyperparameter search space
    if args.policy == "PPO":
        param_space = {
            "training": {
                "lr": tune.loguniform(1e-5, 1e-3),
                "gamma": tune.uniform(0.9, 0.999),
                "lambda_": tune.uniform(0.9, 1.0),
            }
        }
    # Configure ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        max_t=args.max_iterations_hyperparameter_tuning,
        grace_period=args.grace_period_hyperparameter_tuning,
        reduction_factor=2
    )
    
    # Configure WandbLoggerCallback
    wandb_logger = WandbLoggerCallback(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        group=f"{wandb_config['group']}_tune",
        name=f"{wandb_config['name']}_tune",
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
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.num_samples_hyperparameter_tuning,
            metric="episode_reward_mean",
            mode="max",
        ),
    )
    
    results = tuner.fit()
    
    # Get best trial
    best_trial = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final reward: {best_trial.metrics['episode_reward_mean']}")
    
    # Save best config
    best_config_path = os.path.join(checkpoint_dir, f"{experiment_name}_best_config_seed{seed}.json")
    with open(best_config_path, "w") as f:
        import json
        json.dump(best_trial.config, f, indent=2)
    
    print(f"Best config saved to: {best_config_path}")

def run_training(args, algo_config, checkpoint_dir, experiment_name, seed, wandb_config):
    # Configure WandbLoggerCallback
    wandb_logger = WandbLoggerCallback(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        name=wandb_config["name"],
        group=wandb_config["group"],
    )

    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=args.iterations),
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
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    )
    
    results = tuner.fit()
    
    # Get best checkpoint
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best result: {best_result.metrics}")
    print(f"Best checkpoint: {best_result.checkpoint}")

register_env("FSS_env", lambda config: env_creator(config))
print("Registered environment")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Training configuration arguments
    parser.add_argument("--policy", type=str, default="PPO", help="RL algorithm to use")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--num-samples-hyperparameter-tuning", type=int, default=20, help="Number of hyperparameter samples to run")
    parser.add_argument("--max-iterations-hyperparameter-tuning", type=int, default=25, help="Maximum number of training iterations")
    parser.add_argument("--grace-period-hyperparameter-tuning", type=int, default=10, help="Grace period for early stopping")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
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
    parser.add_argument("--num-env-runners", type=int, default=10, help="Number of environment runners")
    parser.add_argument("--num-envs-per-runner", type=int, default=1, help="Number of environments per runner")
    parser.add_argument("--num-cpus-per-runner", type=int, default=1, help="Number of CPUs per runner")
    parser.add_argument("--num-gpus-per-runner", type=float, default=0, help="Number of GPUs per runner")
    parser.add_argument("--num-learners", type=int, default=1, help="Number of learners")
    parser.add_argument("--num-gpus-per-learner", type=float, default=1, help="Number of GPUs per learner")
    parser.add_argument("--num-cpus-per-learner", type=int, default=1, help="Number of CPUs per learner")
    
    args = parser.parse_args()

    # Start training
    main(args) 