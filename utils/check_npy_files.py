import os
import numpy as np

folder = "Results/MonteCarlo"  # Path to the folder containing simulation results

def check_npy_files(folder):
    simulation_dirs = [os.path.join(folder, d) for d in os.listdir(folder) if 'simulation_' in d and os.path.isdir(os.path.join(folder, d))]
    for sim_dir in simulation_dirs:
        print(f"Checking directory: {sim_dir}")
        for file_name in os.listdir(sim_dir):
            if file_name.endswith(".npy"):
                file_path = os.path.join(sim_dir, file_name)
                print(f"File: {file_path}")
                try:
                    data = np.load(file_path)
                    print(f"Data shape: {data.shape}, Data type: {data.dtype}")
                    print(f"Data: {data}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

if __name__ == "__main__":
    check_npy_files(folder)