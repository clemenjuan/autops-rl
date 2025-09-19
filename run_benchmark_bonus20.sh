#!/bin/bash

# SLURM directives for bonus20 benchmark
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH -o autops_benchmark_bonus20_%j.out
#SBATCH -e autops_benchmark_bonus20_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=clemente.juan@tum.de

# Create the Enroot container if it doesn't exist
if [ ! -d "$HOME/.local/share/enroot/user-$USER/autops-rl" ]; then
    echo "Creating Enroot container..."
    enroot create --name autops-rl /dss/dsshome1/05/ge26sav2/autops-rl/nvidia+pytorch+25.02-py3.sqsh
fi

# Start the container and run the commands
srun enroot start --root --mount /dss/dsshome1/05/ge26sav2/autops-rl:/workspace autops-rl bash -c "
    # Install dependencies with NumPy compatibility fix
    echo 'Installing dependencies...'
    pip install --no-cache-dir 'numpy<2.0' 
    pip install --no-cache-dir -r /workspace/requirements.txt
    
    # Set environment variables
    export PYTHONPATH=\"\$PYTHONPATH:/workspace\"
    export RAY_DEDUP_LOGS=0
    export CUDA_VISIBLE_DEVICES=0
    export RAY_DISABLE_DOCKER_CPU_WARNING=1
    
    # Create benchmark output directory
    mkdir -p /workspace/benchmark_results
    cd /workspace
    
    echo '=================================================='
    echo 'STARTING BONUS20 SENSITIVITY BENCHMARK'
    echo '=================================================='
    echo 'System Info:'
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo 'CPU Info:'
    lscpu | grep 'Model name'
    echo 'Memory Info:'
    free -h
    echo 'Available checkpoint directories:'
    find /workspace -name 'checkpoints_case*_bonus20_*' -type d | sort
    echo 'Available checkpoint files:'
    find /workspace -name 'best_case*_seed*_sim_everyone.ckpt' | grep bonus20 | sort
    echo '=================================================='
    
    # Run bonus20 benchmark
    echo 'Running BONUS20 benchmark...'
    python /workspace/benchmark_policies_sensitivity.py \\
        --bonus-groups bonus20 \\
        --episodes 15
    
    if [ \$? -eq 0 ]; then
        echo '✓ Bonus20 benchmark completed successfully'
    else
        echo '✗ Bonus20 benchmark failed'
    fi
    
    # Find the most recent experiment folder
    LATEST_EXPERIMENT=\$(find /workspace/experiments -name 'sensitivity_benchmark_*' -type d | sort | tail -1)
    
    if [ -n \"\$LATEST_EXPERIMENT\" ]; then
        echo \"Found latest experiment: \$LATEST_EXPERIMENT\"
        
        # Copy results to a timestamped backup location
        TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR=\"/workspace/benchmark_results/bonus20_backup_\$TIMESTAMP\"
        cp -r \"\$LATEST_EXPERIMENT\" \"\$BACKUP_DIR\"
        echo \"📦 Backup created: \$BACKUP_DIR\"
        
    else
        echo '✗ No experiment folder found'
    fi
    
    echo '=================================================='
    echo 'BONUS20 BENCHMARK COMPLETE!'
    echo '=================================================='
    
    # List all experiment folders
    echo 'Available experiment folders:'
    find /workspace/experiments -name 'sensitivity_benchmark_*' -type d | sort
    echo
    echo 'Available backup folders:'
    find /workspace/benchmark_results -name 'bonus20_backup_*' -type d | sort
    echo '=================================================='
" 