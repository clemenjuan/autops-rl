import os
import numpy as np
import pandas as pd
import csv


def log_full_matrices(matrices, simulation_number, folder):
    matrix_folder = os.path.join(folder, f"simulation_{simulation_number}_matrices")
    os.makedirs(matrix_folder, exist_ok=True)

    for key, matrix in matrices.items():
        # Ensure matrix is a numpy array
        matrix = np.array(matrix)
        # Define file path for the matrix
        file_path = os.path.join(matrix_folder, f"{key}.npy")
        # Save the matrix to file in .npy format
        np.save(file_path, matrix)

def log_summary_results(data_summary, simulation_number, folder):
    summary_file = os.path.join(folder, f"simulation_{simulation_number}_summaries.csv")
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
    Get the next simulation number based on existing files in the results folder.
    """
    existing_files = [file for file in os.listdir(results_folder) if file.startswith("simulation_results_")]

    if existing_files:
        latest_simulation_number = max(int(file.split("_")[2].split(".")[0]) for file in existing_files)
        return latest_simulation_number + 1
    else:
        return 1
    
def aggregate_simulation_results(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith('simulation_results_')]
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
                # If data is multi-dimensional, take the mean across all dimensions
                mean_value = np.mean(data) if data.ndim > 0 else data.item()
                all_values.append(mean_value)
        
        # Compute the mean across all simulations for this attribute
        statistics[attr] = np.mean(all_values)
    
    # Write the overall means to a single CSV
    csv_file = os.path.join(folder, "averages.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Attribute", "Average"])
        for attr, avg_value in statistics.items():
            writer.writerow([attr, avg_value])

    return statistics

