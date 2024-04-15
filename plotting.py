import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import re

def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def parse_additional_info(content):
    # Regex to extract one-dimensional arrays and single-value metrics
    global_obs_count = re.search(r'Global observation count matrix:\s*\[(.*?)\]', content)
    max_pointing_accuracy = re.search(r'Maximum pointing accuracy average matrix:\s*\[(.*?)\]', content)
    total_time = re.search(r'Total time of episode:\s*(\d+\.\d+) seconds', content)
    total_reward = re.search(r'Total reward:\s*([-\d\.]+)', content)

    # Convert extracted strings to appropriate data types
    global_obs_count = np.fromstring(global_obs_count.group(1), sep=' ') if global_obs_count else np.array([])
    max_pointing_accuracy = np.fromstring(max_pointing_accuracy.group(1), sep=' ') if max_pointing_accuracy else np.array([])
    total_time = float(total_time.group(1)) if total_time else 0
    total_reward = float(total_reward.group(1)) if total_reward else 0

    return global_obs_count, max_pointing_accuracy, total_time, total_reward

def parse_matrix_from_string(matrix_str):
    # Trim the outer double brackets and split into rows
    rows = matrix_str.strip()[2:-2].split('], [')
    matrix = []
    for row in rows:
        # Remove potential trailing characters and split by spaces
        clean_row = row.replace(']', '').replace('[', '').strip()
        numbers = clean_row.split()
        # Convert each number to float or int after cleaning trailing commas
        matrix.append([float(num.strip(',')) if '.' in num else int(num.strip(',')) for num in numbers])
    return np.array(matrix)

def parse_matrices_from_file(file_path, save_dir):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regex to extract matrices with consideration for spacing and formatting
    matrix_strs = re.findall(r'\[\[\s*.*?\s*\]\]', content, flags=re.DOTALL)
    matrix_dict = {}
    matrix_labels = ['adjacency_matrix', 'data_matrix', 'contacts_matrix',
                     'global_observation_status_matrix', 'global_observation_count_matrix',
                     'maximum_pointing_accuracy_average_matrix']

    for label, matrix_str in zip(matrix_labels, matrix_strs):
        try:
            matrix = parse_matrix_from_string(matrix_str)
        except Exception as e:
            print(f"Error parsing matrix for {label}: {e}")
            continue

        matrix_dict[label] = matrix
        np.save(os.path.join(save_dir, f'{label}.npy'), matrix)  # Saving as .npy

    # Parse additional info
    global_obs_count, max_pointing_accuracy, total_time, total_reward = parse_additional_info(content)

    # Save the additional data as npy files
    np.save(os.path.join(save_dir, 'global_observation_count_matrix.npy'), global_obs_count)
    np.save(os.path.join(save_dir, 'maximum_pointing_accuracy_average_matrix.npy'), max_pointing_accuracy)
    np.save(os.path.join(save_dir, 'total_time.npy'), np.array([total_time]))  # Wrapped in an array to save as .npy
    np.save(os.path.join(save_dir, 'total_reward.npy'), np.array([total_reward]))  # Wrapped in an array to save as .npy

    # Include the additional info in the dictionary
    matrix_dict['global_observation_count_matrix'] = global_obs_count
    matrix_dict['maximum_pointing_accuracy_average_matrix'] = max_pointing_accuracy

    return matrix_dict, total_time, total_reward



def plot_matrices(matrix_dict, plot_dir, file_identifier, total_time, total_reward):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust for a 2x2 subplot grid if fewer plots are needed

    # Plotting the binary grid plots
    binary_matrices = ['adjacency_matrix']
    for i, title in enumerate(binary_matrices):
        ax = axs[i // 2, i % 2]
        matrix = matrix_dict[title]
        c = ax.matshow(matrix, cmap='Greys', aspect='auto')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Index', fontsize=12)

    # Plotting the global_observation_status_matrix with a discrete colormap
    observation_status_matrix = matrix_dict['global_observation_status_matrix']
    # Create a discrete colormap
    cmap = colors.ListedColormap(['white', 'yellow', 'orange', 'green'])
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
    ax = axs[0, 1]
    c = ax.matshow(observation_status_matrix, cmap=cmap, norm=norm, aspect='auto')
    fig.colorbar(c, ax=ax, boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ticks=[0, 1, 2, 3])
    ax.set_title('global_observation_status_matrix', fontsize=14)
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Index', fontsize=12)

    # Plotting the bar charts for 1D arrays
    array_labels = ['global_observation_count_matrix', 'maximum_pointing_accuracy_average_matrix']
    for i, title in enumerate(array_labels):
        ax = axs[1, i % 2]  # Assuming you are using a 2x2 grid
        data = matrix_dict[title]
        ax.bar(np.arange(len(data)), data, color='skyblue')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)

    # Adjusting the layout
    plt.tight_layout()

    # Annotating the total time and reward
    plt.figtext(0.5, 0.01, f"Total Time: {total_time:.3f} seconds\nTotal Reward: {total_reward:.3f}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Save the figure
    plt.savefig(os.path.join(plot_dir, f'plot_{file_identifier}.png'), bbox_inches='tight')
    plt.close(fig)



# Main setup
folder_path = 'Results/v0'
npy_dir = 'Results/v0/npy_matrices'
plot_dir = 'Results/v0/plots'
# ensure_directory_exists(npy_dir)
ensure_directory_exists(plot_dir)


files = os.listdir(folder_path)
all_matrices = {}

for file in files:
    if file.startswith('simulation_output') and file.endswith('.txt'):
        file_path = os.path.join(folder_path, file)
        file_identifier = os.path.splitext(file)[0]  # Use filename without extension for identifiers
        matrices, total_time, total_reward = parse_matrices_from_file(file_path, npy_dir)  # Unpack the returned tuple
        all_matrices[file] = matrices
        plot_matrices(matrices, plot_dir, file_identifier, total_time, total_reward)  # Pass all required arguments
