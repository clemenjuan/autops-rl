import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

def plot(matrices, results_folder, total_duration, total_reward):
    # Call the plotting function
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    i = get_next_simulation_number_plot(results_folder)

    plot_matrices(matrices, results_folder, f'simulation_{i}', total_duration, total_reward)



def plot_matrices(matrix_dict, plot_dir, file_identifier, total_time, total_reward):
    # Use a 3x2 grid layout
    fig, axs = plt.subplots(3, 2, figsize=(12, 18), gridspec_kw={'hspace': 0.3, 'top': 0.85})

    # Plotting the binary grid plots
    binary_matrices = ['adjacency_matrix']
    for i, title in enumerate(binary_matrices):
        ax = axs[i // 2, i % 2]
        matrix = matrix_dict[title]
        c = ax.matshow(matrix, cmap='Greys', aspect='auto')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=14) # Set the title
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
    array_labels = ['global_observation_count_matrix', 'maximum_pointing_accuracy_average_matrix']
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
    plt.figtext(0.5, 0.92, f"Total Time: {total_time:.3f} seconds\nTotal Reward: {total_reward:.3f}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

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