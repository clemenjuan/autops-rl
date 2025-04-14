import os
import time
import numpy as np
import argparse
import ray
import torch
from ray import tune
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from src.envs.FSS_env import FSS_env
from utils.data_logging import log_full_matrices, log_summary_results, compute_statistics_from_npy
from utils.plotting import plot

'''
#### Usage ##############################
python3 main.py --checkpoint-dir mnt/checkpoints/ppo --policy ppo
'''
os.environ["RAY_verbose_spill_logs"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
parser.add_argument("--policy", choices=["ppo","dqn","sac"], required=True, help="Policy to test.")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to load checkpoints.")

args = parser.parse_args()


def env_creator(env_config):
    env = FSS_env(**env_config)
    return env

def test_policy(checkpoint_dir, num_simulations, simulator_type):
    # Set paths
    policy_checkpoint_path = os.path.join(checkpoint_dir, "policies", "default_policy")

    my_restored_policy = Policy.from_checkpoint(policy_checkpoint_path)
    print("Using policy: ", my_restored_policy)

    results_folder = os.path.join("/mnt/Results", simulator_type, args.policy)  # Updated folder name
    results_folder_plots = os.path.join(results_folder, "plots")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(results_folder_plots, exist_ok=True)

    write_to_csv_file_flag = False  # Write the output to a file
    plot_flag = False  # Plot the output
    relevant_attributes = [
        'adjacency_matrix', 'data_matrix', 'contacts_matrix',
        'global_observation_counts', 'max_pointing_accuracy_avg',
        'global_observation_status_matrix', 'batteries', 'storage', 'total_reward', 
        'total_duration', 'total_steps', 'global_communication_counts',
    ]

    total_compute_times = 0.0  # Variable to accumulate the total compute time
    total_actions = 0  # Variable to count the number of actions computed

    for i in range(num_simulations):
        print()
        print(f"######## Starting simulation {i+1} ########")
        # Run the simulation until timeout or agent failure
        env = env_creator(env_config)
        total_reward = 0
        observation, infos = env.reset()

        start_time = time.time()

        while env.agents:
            step_start_time = time.time()

            # Compute actions for all agents and record the time taken
            actions = {agent: my_restored_policy.compute_single_action(observation[agent])[0] for agent in env.agents}

            compute_action_time = time.time() - step_start_time
            total_compute_times += compute_action_time  # Accumulate compute time
            total_actions += len(env.agents)  # Accumulate the number of actions

            observation, rewards, terminated, truncated, infos = env.step(actions)
            total_reward += sum(rewards.values())

            # if env.simulator.time_step_number % 1000 == 0:
                # print(f"Computed actions in {compute_action_time * 1e3:.2f} ms for {len(env.agents)} agents")
                # print(f"Actions: {actions}")

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
            'global_communication_counts': env.simulator.global_communication_counts,
            'global_observation_counts': env.simulator.global_observation_counts,
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
            plot(matrices, results_folder_plots, total_duration, total_reward, env.simulator.time_step_number)
    
    if write_to_csv_file_flag:
        compute_statistics_from_npy(results_folder, relevant_attributes)
        print("Averages written to averages.csv")

    # Calculate and print average compute time per action per agent
    if total_actions > 0:
        avg_compute_time_per_action = (total_compute_times / total_actions) * 1e3  # Convert to milliseconds
        print(f"Average compute time per action per agent: {avg_compute_time_per_action:.2f} ms")

if __name__ == "__main__":
    # Register environment
    env_name = "FSS_env-v0"
    simulator_types = ["everyone", "centralized", "decentralized"]
    for simulator_type in simulator_types:
        env_config = {
            "num_targets": 20, 
            "num_observers": 20, 
            "simulator_type": simulator_type, 
            "time_step": 1, 
            "duration": 24*60*60
        }
        register_env("FSS_env-v0", lambda config: env_creator(env_config))
        test_policy(args.checkpoint_dir, num_simulations=1, simulator_type=simulator_type)