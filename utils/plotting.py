import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from mpl_toolkits.mplot3d import Axes3D


def plot(matrices, results_folder, total_duration, total_reward, total_steps):
    """Main plotting function that handles all edge cases"""
    # Ensure scalar values for annotations
    try:
        total_duration = float(total_duration) if hasattr(total_duration, 'item') else float(total_duration)
        total_reward = float(total_reward) if hasattr(total_reward, 'item') else float(total_reward)
        total_steps = int(total_steps) if hasattr(total_steps, 'item') else int(total_steps)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert metrics to scalar values: {total_duration}, {total_reward}, {total_steps}")
        total_duration, total_reward, total_steps = 0.0, 0.0, 0
    
    # Check if matrices is empty or None
    if not matrices:
        print("Warning: No matrices to plot")
        return
        
    # Create the output directory if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Get the next simulation number for the filename
    i = get_next_simulation_number_plot(results_folder)
    
    try:
        # Call the plotting function with validated parameters
        plot_matrices(matrices, results_folder, f'simulation_{i}', total_duration, total_reward, total_steps)
        print(f"✓ Successfully generated plot: simulation_{i}")
    except Exception as e:
        print(f"Error generating plot: {e}")
        # Try a simplified backup plot
        try:
            create_simple_summary_plot(matrices, results_folder, f'summary_{i}', total_duration, total_reward, total_steps)
            print(f"✓ Generated simplified summary plot: summary_{i}")
        except Exception as e2:
            print(f"Error generating simplified plot: {e2}")


def create_simple_summary_plot(matrices, plot_dir, file_identifier, total_time, total_reward, total_steps):
    """Create a simplified text-based summary when regular plotting fails"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Prepare text summary
    summary_text = [
        f"Run Summary (ID: {file_identifier})",
        f"------------------------",
        f"Total Time: {float(total_time):.2f} seconds",
        f"Total Reward: {float(total_reward):.2f}",
        f"Total Steps: {int(total_steps)}",
        f"------------------------",
        f"Available Data:"
    ]
    
    # Add list of available matrices
    for key in matrices:
        if key in matrices and matrices[key] is not None:
            if isinstance(matrices[key], np.ndarray):
                shape_info = matrices[key].shape if hasattr(matrices[key], 'shape') else "Unknown shape"
                summary_text.append(f"• {key}: {shape_info}")
            else:
                summary_text.append(f"• {key}: {type(matrices[key])}")
    
    # Display the text summary
    ax.text(0.1, 0.9, '\n'.join(summary_text), 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', family='monospace')
    
    # Save the figure
    plt.savefig(os.path.join(plot_dir, f'summary_{file_identifier}.png'), bbox_inches='tight')
    plt.close(fig)


def plot_matrices(matrix_dict, plot_dir, file_identifier, total_time, total_reward, total_steps):
    """Plot matrices with robust error handling"""
    # Convert annotation values to scalar
    total_time = float(total_time) if hasattr(total_time, 'item') else float(total_time)
    total_reward = float(total_reward) if hasattr(total_reward, 'item') else float(total_reward)
    total_steps = int(total_steps) if hasattr(total_steps, 'item') else int(total_steps)
    
    # Use a 3x2 grid layout
    fig, axs = plt.subplots(3, 2, figsize=(12, 18), gridspec_kw={'hspace': 0.3, 'top': 0.85})
    
    # Plotting the binary grid plots - check if data exists
    binary_matrices = ['adjacency_matrix']
    for i, title in enumerate(binary_matrices):
        ax = axs[i // 2, i % 2]
        try:
            if title in matrix_dict and matrix_dict[title] is not None and hasattr(matrix_dict[title], 'size') and matrix_dict[title].size > 0:
                matrix = matrix_dict[title]
                c = ax.matshow(matrix, cmap='Greys', aspect='auto')
                fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title("Adjacency Matrix", fontsize=14)
                ax.set_xlabel('Observer', fontsize=12)
                ax.set_ylabel('Observer', fontsize=12)
            else:
                ax.text(0.5, 0.5, f"No {title} data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                ax.set_title(f"Missing: {title}", fontsize=14)
        except Exception as e:
            print(f"Error plotting {title}: {e}")
            ax.text(0.5, 0.5, f"Error plotting {title}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(f"Error: {title}", fontsize=14)

    # Plotting the global_observation_status_matrix with a discrete colormap
    ax = axs[0, 1]
    try:
        if ('global_observation_status_matrix' in matrix_dict and 
            matrix_dict['global_observation_status_matrix'] is not None and 
            hasattr(matrix_dict['global_observation_status_matrix'], 'size') and 
            matrix_dict['global_observation_status_matrix'].size > 0):
            
            observation_status_matrix = matrix_dict['global_observation_status_matrix']
            # Create a discrete colormap
            cmap = colors.ListedColormap(['white', 'yellow', 'orange', 'green'])
            norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
            c = ax.matshow(observation_status_matrix, cmap=cmap, norm=norm, aspect='auto')
            fig.colorbar(c, ax=ax, boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ticks=[0, 1, 2, 3])
            ax.set_title('Observation Status', fontsize=14)
            ax.set_xlabel('Target', fontsize=12)
            ax.set_ylabel('Observer', fontsize=12)
        else:
            ax.text(0.5, 0.5, "No observation status data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_title("Missing: Observation Status", fontsize=14)
    except Exception as e:
        print(f"Error plotting observation status: {e}")
        ax.text(0.5, 0.5, "Error plotting observation status", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes)
        ax.set_title("Error: Observation Status", fontsize=14)

    # Safe conversion for annotation values (again, to be super sure)
    try:
        total_time_str = f"{float(total_time):.3f}"
        total_reward_str = f"{float(total_reward):.3f}" 
        total_steps_str = f"{int(total_steps)}"
    except (ValueError, TypeError):
        total_time_str = "0.000"
        total_reward_str = "0.000"
        total_steps_str = "0"

    # Annotate the total time and reward
    plt.figtext(0.5, 0.92, 
        f"Total Time: {total_time_str} seconds\nTotal Reward: {total_reward_str}\nTotal steps: {total_steps_str}", 
        ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

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