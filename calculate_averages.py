import os
import numpy as np
import csv
from data_logging import compute_statistics_from_npy

if __name__ == "__main__":
    results_folder = os.path.join("Results", "MonteCarlo")  # Updated folder name
    relevant_attributes = [
            'adjacency_matrix', 'data_matrix', 'contacts_matrix',
            'global_observation_counts', 'max_pointing_accuracy_avg',
            'global_observation_status_matrix', 'batteries', 'storage', 'total_reward', 
            'total_duration', 'total_steps',
            ]

    statistics = compute_statistics_from_npy(results_folder, relevant_attributes)
    print("Averages computed and written to averages.csv")
    print(statistics)