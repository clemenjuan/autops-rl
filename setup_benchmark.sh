#!/bin/bash

echo "Setting up benchmark for LRZ execution..."

# Make scripts executable
chmod +x run_benchmark_lrz.sh

echo "Checking host filesystem paths..."
echo "Available checkpoint directories on host:"
find /dss/dsshome1/05/ge26sav2/autops-rl -name "checkpoints_case*_training_*" -type d 2>/dev/null | sort

echo
echo "Available checkpoint files on host:"
find /dss/dsshome1/05/ge26sav2/autops-rl -name "best_case*_seed*_sim_everyone.ckpt" 2>/dev/null | sort

echo
echo "Note: Inside the Enroot container, these paths will be mounted to /workspace/"
echo "For example: /dss/dsshome1/05/ge26sav2/autops-rl/checkpoints_case1_training_5164693/"
echo "becomes:     /workspace/checkpoints_case1_training_5164693/"
echo

echo "To run benchmark:"
echo "  sbatch run_benchmark_lrz.sh"
echo
echo "To monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f autops_benchmark_<job_id>.out" 