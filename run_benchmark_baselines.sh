#!/bin/bash

# SLURM directives for baseline policies only
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH -o autops_benchmark_baselines_%j.out
#SBATCH -e autops_benchmark_baselines_%j.err
#SBATCH --time=24:00:00
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
    echo 'STARTING BASELINE POLICIES BENCHMARK'
    echo '=================================================='
    echo 'Testing RuleBased and MIP on ALL coordination types'
    echo 'System Info:'
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo 'CPU Info:'
    lscpu | grep 'Model name'
    echo 'Memory Info:'
    free -h
    echo '=================================================='
    
    # Run baseline policies only (RuleBased + MIP)
    echo 'Running BASELINE POLICIES benchmark on all coordination types...'
    python /workspace/benchmark_baselines_only.py \\
        --episodes 15
    
    if [ \$? -eq 0 ]; then
        echo '✓ Baseline benchmark completed successfully'
    else
        echo '✗ Baseline benchmark failed'
    fi
    
    # Find the most recent experiment folder
    LATEST_EXPERIMENT=\$(find /workspace/experiments -name 'baseline_benchmark_*' -type d | sort | tail -1)
    
    if [ -n \"\$LATEST_EXPERIMENT\" ]; then
        echo \"Found latest experiment: \$LATEST_EXPERIMENT\"
        
        # Copy results to a timestamped backup location
        TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
        BACKUP_DIR=\"/workspace/benchmark_results/baselines_backup_\$TIMESTAMP\"
        cp -r \"\$LATEST_EXPERIMENT\" \"\$BACKUP_DIR\"
        echo \"📦 Backup created: \$BACKUP_DIR\"
        
    else
        echo '✗ No experiment folder found'
    fi
    
    echo '=================================================='
    echo 'BASELINE BENCHMARK COMPLETE!'
    echo '=================================================='
    
    # List all experiment folders
    echo 'Available experiment folders:'
    find /workspace/experiments -name 'baseline_benchmark_*' -type d | sort
    echo
    echo 'Available backup folders:'
    find /workspace/benchmark_results -name 'baselines_backup_*' -type d | sort
    echo '=================================================='
" 