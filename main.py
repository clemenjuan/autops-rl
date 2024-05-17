import os
import time
import numpy as np
import argparse
import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from FSS_env import FSS_env
from data_logging import log_full_matrices, log_summary_results, compute_statistics_from_npy
from plotting import plot

'''
#### Usage ##############################
python3 main.py --framework torch --policy ppo --checkpoint-dir ppo_checkpoints/ppo_policy
'''
os.environ["RAY_verbose_spill_logs"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

def setup_config(config):
    config.environment(env=env_name, env_config=env_config, disable_env_checking=True)
    config.framework(args.framework)
    config.rollouts(num_rollout_workers=4, num_envs_per_worker=2, rollout_fragment_length="auto", batch_mode="complete_episodes")
    config.resources(num_gpus=1 if torch.cuda.is_available() else 0)
    return config

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
parser.add_argument("--policy", choices=["ppo"], required=True, help="Policy to test.")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to load checkpoints.")
args = parser.parse_args()

def env_creator(env_config):
    env = FSS_env(**env_config)
    return env

# Register environment
env_name = "FSS_env-v0"
env_config = {
    "num_targets": 10, 
    "num_observers": 10, 
    "simulator_type": 'everyone', 
    "time_step": 1, 
    "duration": 24*60*60
}
register_env(env_name, lambda config: env_creator(env_config))

# Configuration for PPO
ppo_config = setup_config(PPOConfig())

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir)]
    checkpoints = [path for path in checkpoints if os.path.isdir(path)]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def test_policy(config, policy_name, checkpoint_dir, num_simulations):
    algorithm = config.build()

    checkpoint_path = os.path.join(checkpoint_dir, policy_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    latest_checkpoint = get_latest_checkpoint(checkpoint_path)

    if latest_checkpoint:
        algorithm.restore(latest_checkpoint)

    results_folder = os.path.join("Results", "PPO")  # Updated folder name
    results_folder_plots = os.path.join(results_folder, "plots")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(results_folder_plots, exist_ok=True)

    write_to_csv_file_flag = True  # Write the output to a file
    plot_flag = True  # Plot the output
    relevant_attributes = [
        'adjacency_matrix', 'data_matrix', 'contacts_matrix',
        'global_observation_counts', 'max_pointing_accuracy_avg',
        'global_observation_status_matrix', 'batteries', 'storage', 'total_reward', 
        'total_duration', 'total_steps',
    ]

    for i in range(num_simulations):
        print()
        print(f"######## Starting simulation {i+1} ########")
        # Run the simulation until timeout or agent failure
        env = env_creator(env_config)
        total_reward = 0
        observation, infos = env.reset()

        action_counts = {}
        start_time = time.time()

        while env.agents:
            step_start_time = time.time()
            actions = {agent: algorithm.compute_single_action(observation[agent]) for agent in env.agents}
            for agent, action in actions.items():
                action_counts.setdefault(agent, []).append(action)

            observation, rewards, terminated, truncated, infos = env.step(actions)
            total_reward += sum(rewards.values())
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            if any(terminated.values()) or any(truncated.values()):
                print("Episode finished")
                break

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Total steps: {env.simulator.time_step_number}")
        print(f"Total duration of episode: {total_duration:.3f} seconds")
        print(f"Total reward: {total_reward}")

        # Prepare the data for logging
        matrices = {
            'adjacency_matrix': env.simulator.adjacency_matrix_acc,
            'data_matrix': env.simulator.data_matrix_acc,
            'contacts_matrix': env.simulator.contacts_matrix_acc,
            'global_observation_counts': np.sum(env.simulator.global_observation_counts, axis=0),
            'max_pointing_accuracy_avg': env.simulator.max_pointing_accuracy_avg,
            'global_observation_status_matrix': env.simulator.global_observation_status_matrix,
            'batteries': env.simulator.batteries,
            'storage': env.simulator.storage,
            'total_reward': total_reward,
            'total_duration': total_duration,
            'total_steps': env.simulator.time_step_number,
        }

        if write_to_csv_file_flag:
            log_full_matrices(matrices, results_folder)
            data_summary = {
                'Total Reward': total_reward,
                'Total Duration': total_duration,
            }
            log_summary_results(data_summary, results_folder)

        if plot_flag:
            plot(matrices, results_folder_plots, total_duration, total_reward)
    
    if write_to_csv_file_flag:
        compute_statistics_from_npy(results_folder, relevant_attributes)
        print("Averages written to averages.csv")

if __name__ == "__main__":
    test_policy(ppo_config, "ppo_policy", args.checkpoint_dir, num_simulations=10)
