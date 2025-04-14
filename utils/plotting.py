import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from mpl_toolkits.mplot3d import Axes3D


def plot(matrices, results_folder, total_duration, total_reward, total_steps):
    # Call the plotting function
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    i = get_next_simulation_number_plot(results_folder)

    plot_matrices(matrices, results_folder, f'simulation_{i}', total_duration, total_reward, total_steps)



def plot_matrices(matrix_dict, plot_dir, file_identifier, total_time, total_reward, total_steps):
    # Use a 3x2 grid layout
    fig, axs = plt.subplots(3, 2, figsize=(12, 18), gridspec_kw={'hspace': 0.3, 'top': 0.85})

    # Plotting the binary grid plots
    binary_matrices = ['adjacency_matrix']
    for i, title in enumerate(binary_matrices):
        ax = axs[i // 2, i % 2]
        matrix = matrix_dict[title]
        c = ax.matshow(matrix, cmap='Greys', aspect='auto')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Adjacency Matrix", fontsize=14) # Set the title
        ax.set_xlabel('Observer', fontsize=12)
        ax.set_ylabel('Observer', fontsize=12)

    # Plotting the global_observation_status_matrix with a discrete colormap
    observation_status_matrix = matrix_dict['global_observation_status_matrix']
    # Create a discrete colormap
    cmap = colors.ListedColormap(['white', 'yellow', 'orange', 'green'])
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    ax = axs[0, 1]
    c = ax.matshow(observation_status_matrix, cmap=cmap, norm=norm, aspect='auto')
    fig.colorbar(c, ax=ax, boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ticks=[0, 1, 2, 3])
    ax.set_title('Observation Status', fontsize=14)
    ax.set_xlabel('Target', fontsize=12)
    ax.set_ylabel('Observer', fontsize=12)

    # Plotting the bar charts for 1D arrays 2nd row
    array_labels = ['global_observation_counts', 'max_pointing_accuracy_avg']
    for i, title in enumerate(array_labels):
        ax = axs[1, i % 2] 
        data = matrix_dict[title]
        ax.bar(np.arange(len(data)), data, color='skyblue')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Target', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

    array_labels = ['batteries', 'storage']
    for i, title in enumerate(array_labels):
        ax = axs[2, i % 2] 
        data = matrix_dict[title]
        ax.bar(np.arange(len(data)), data, color='skyblue')
        ax.set_title(title, fontsize=14)
        # Setting the y-axis limit to 1 for normalization
        ax.set_ylim(0, 1)
        ax.set_xlabel('Observer', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

    # Annotate the total time and reward
    plt.figtext(0.5, 0.92, f"Total Time: {total_time:.3f} seconds\nTotal Reward: {total_reward:.3f}\nTotal steps: {total_steps}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Adjust the layout to ensure everything fits
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # rect parameter is [left, bottom, right, top]

    # Save the figure
    plt.savefig(os.path.join(plot_dir, f'plot_{file_identifier}.png'), bbox_inches='tight')
    plt.close(fig)


    
def get_next_simulation_number_plot(results_folder):
    """
    Get the next simulation number based on existing files in the results folder.
    """
    existing_files = [file for file in os.listdir(results_folder) if file.startswith("plot_simulation_")]

    if existing_files:
        latest_simulation_number = max(int(file.split("_")[2].split(".")[0]) for file in existing_files)
        return latest_simulation_number + 1
    else:
        return 1
    
def find_global_min_max(csv_files):
    global_min, global_max = float('inf'), float('-inf')
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        min_val = df['episode_reward_mean'].min()
        max_val = df['episode_reward_mean'].max()
        if min_val < global_min:
            global_min = min_val
        if max_val > global_max:
            global_max = max_val
    return global_min, global_max

def normalize_series(series, global_min, global_max):
    return (series - global_min) / (global_max - global_min)


def plot_normalized(csv_file, plot_dir, algorithm_name, global_min, global_max):
    df = pd.read_csv(csv_file)
    normalized_reward_mean = normalize_series(df['episode_reward_mean'], global_min, global_max)
    
    plt.figure(figsize=(14, 8))
    plt.plot(df['training_iteration'], normalized_reward_mean, marker='o', linestyle='-', label=f'{algorithm_name}')
    plt.xlabel('Training Iteration')
    plt.ylabel('Normalized Episode Reward Mean')
    plt.title(f'Normalized Episode Reward Mean for {algorithm_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'normalized_episode_reward_mean_{algorithm_name}.png'))
    plt.close()

    # Normalize the timestamps to start from zero
    initial_timestamp = df['timestamp'].min()
    normalized_timestamp = df['timestamp'] - initial_timestamp

    # Debug: Print the first few values to verify normalization
    # print(f"{algorithm_name} - Initial Timestamp: {initial_timestamp}")
    # print(f"{algorithm_name} - Normalized Timestamps: {normalized_timestamp.head()}")
    
    plt.figure(figsize=(14, 8))
    plt.plot(normalized_timestamp, normalized_reward_mean, linestyle='-', label=f'{algorithm_name}', alpha=0.7)
    plt.xlabel('Training time (seconds)')
    plt.ylabel('Normalized Episode Reward')
    plt.title(f'Normalized Mean Episode Rewards Over Time for {algorithm_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'normalized_mean_rewards_over_training_time_{algorithm_name}.png'))
    plt.close()


def plot_combined_normalized(csv_files, plot_dir, algorithm_names):
    global_min, global_max = find_global_min_max(csv_files)

    plt.figure(figsize=(14, 8))
    for csv_file, algorithm_name in zip(csv_files, algorithm_names):
        df = pd.read_csv(csv_file)
        normalized_reward_mean = normalize_series(df['episode_reward_mean'], global_min, global_max)
        plt.plot(df['training_iteration'], normalized_reward_mean, marker='o', linestyle='-', label=f'{algorithm_name}')
    plt.xlabel('Training Iteration')
    plt.ylabel('Normalized Episode Reward Mean')
    plt.title('Normalized Episode Reward Mean for All Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'normalized_episode_reward_mean_all_algorithms.png'))
    plt.close()

    plt.figure(figsize=(14, 8))
    for csv_file, algorithm_name in zip(csv_files, algorithm_names):
        df = pd.read_csv(csv_file)
        normalized_reward_mean = normalize_series(df['episode_reward_mean'], global_min, global_max)
        initial_timestamp = df['timestamp'].min()
        normalized_timestamp = df['timestamp'] - initial_timestamp

        # Debug: Print the first few values to verify normalization
        # print(f"{algorithm_name} - Initial Timestamp: {initial_timestamp}")
        # print(f"{algorithm_name} - Normalized Timestamps: {normalized_timestamp.head()}")

        plt.plot(normalized_timestamp, normalized_reward_mean, linestyle='-', label=f'{algorithm_name}', alpha=0.7)
    plt.xlabel('Training time (seconds)')
    plt.ylabel('Normalized Episode Reward')
    plt.title('Normalized Mean Episode Rewards Over Training Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'normalized_mean_rewards_all_algorithms.png'))
    plt.close()


if __name__ == "__main__":
    # Define the CSV files and algorithm names
    csv_files = [
        'checkpoints_new/dqn/FSS_env_dqn/TorchTrainer_62445_00000_0_2024-08-12_20-47-54/progress.csv',
        'checkpoints_new/sac/FSS_env_sac/TorchTrainer_62234_00000_0_2024-08-12_20-47-53/progress.csv',
        'checkpoints_new/ppo/FSS_env_ppo/TorchTrainer_50672_00000_0_2024-08-12_20-47-24/progress.csv'
    ]
    plot_dir = 'Results'
    algorithm_names = ["DQN", "SAC", "PPO"]
    
    # Plot combined normalized statistics
    plot_combined_normalized(csv_files, plot_dir, algorithm_names)

    # Plot separate normalized statistics for each algorithm
    global_min, global_max = find_global_min_max(csv_files)
    for csv_file, algorithm_name in zip(csv_files, algorithm_names):
        plot_normalized(csv_file, plot_dir, algorithm_name, global_min, global_max)