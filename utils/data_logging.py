import os
import numpy as np
import pandas as pd
import csv


def log_full_matrices(matrices, folder):
    simulation_number = get_next_simulation_number_write(folder)
    matrix_folder = os.path.join(folder, f"simulation_{simulation_number}_matrices")
    os.makedirs(matrix_folder, exist_ok=True)

    for key, matrix in matrices.items():
        # Ensure matrix is a numpy array
        matrix = np.array(matrix)
        # Define file path for the matrix
        file_path = os.path.join(matrix_folder, f"{key}.npy")
        # Save the matrix to file in .npy format
        np.save(file_path, matrix)

def log_summary_results(data_summary, folder):
    simulation_number = get_next_simulation_number_write(folder)
    summary_file = os.path.join(folder, f"simulation_{simulation_number - 1}_summaries.csv")
    df = pd.DataFrame([data_summary])
    if not os.path.exists(summary_file):
        df.to_csv(summary_file, index=False)
    else:
        df.to_csv(summary_file, mode='a', header=False, index=False)


def log_simulation_results(data_summary, folder):
    simulation_number = get_next_simulation_number_write(folder)
    df = pd.DataFrame([data_summary])
    csv_file = os.path.join(folder, f'simulation_results_{simulation_number}.csv')
    
    df.to_csv(csv_file, index=False)  # Always write as a new file

def get_next_simulation_number_write(results_folder):
    """
    Get the next simulation number based on existing folders in the results folder.
    """
    # List all items in the results folder
    existing_items = os.listdir(results_folder)
    
    # Filter to only include directories that match the "simulation_x_matrices" pattern
    existing_simulation_dirs = [
        item for item in existing_items 
        if os.path.isdir(os.path.join(results_folder, item)) and item.startswith("simulation_") and item.endswith("_matrices")
    ]
    
    # Extract the simulation numbers
    simulation_numbers = [
        int(item.split("_")[1]) for item in existing_simulation_dirs
    ]
    
    if simulation_numbers:
        # Get the maximum simulation number and add 1
        latest_simulation_number = max(simulation_numbers)
        return latest_simulation_number + 1
    else:
        # If no simulation directories are found, start with 1
        return 1
    
def aggregate_simulation_results(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('simulation_')]
    if not files:
        print("No simulation data found.")
        return

    all_data = [pd.read_csv(f) for f in files]
    full_data = pd.concat(all_data)
    average_data = full_data.mean()
    
    print("Averaged Results Across Simulations:")
    print(average_data)

    # Optionally save or further process the averaged results
    average_data.to_csv(os.path.join(folder, 'averaged_simulation_results.csv'), header=True)


def compute_statistics_from_npy(folder, relevant_attributes):
    statistics = {}
    simulation_dirs = [os.path.join(folder, d) for d in os.listdir(folder) if 'simulation_' in d and os.path.isdir(os.path.join(folder, d))]
    
    for attr in relevant_attributes:
        all_values = []  # List to hold all values for the attribute across simulations
        for sim_dir in simulation_dirs:
            file_path = os.path.join(sim_dir, f"{attr}.npy")
            if os.path.isfile(file_path):
                data = np.load(file_path)
                if attr in ['adjacency_matrix', 'contacts_matrix', 'global_observation_status_matrix', 'data_matrix']:
                    # Average over all elements in the matrix
                    mean_value = np.mean(data)
                else:
                    # If data is multi-dimensional, take the mean across all dimensions
                    mean_value = np.mean(data) if data.ndim > 0 else data.item()
                all_values.append(mean_value)
                # print(f"Simulation: {sim_dir}, Attribute: {attr}, Mean Value: {mean_value}")
        
        # Compute the mean across all simulations for this attribute
        overall_mean = np.mean(all_values)
        statistics[attr] = overall_mean
        # print(f"Attribute: {attr}, All Values: {all_values}, Overall Mean: {overall_mean}")
    
    # Write the overall means to a single CSV
    csv_file = os.path.join(folder, "averages.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Attribute", "Average"])
        for attr, avg_value in statistics.items():
            writer.writerow([attr, f"{avg_value:.2f}"])  # Explicitly format as floating point with 2 decimal places

    return statistics

    return statistics

