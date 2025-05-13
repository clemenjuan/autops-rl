#!/bin/bash

# SLURM directives with reduced resources
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -o autops_test_case1_%j.out
#SBATCH -e autops_test_case1_%j.err
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=END
#SBATCH --mail-user=clemente.juan@tum.de

# Create the Enroot container if it doesn't exist
if [ ! -d "$HOME/.local/share/enroot/user-$USER/pytorch-container" ]; then
    echo "Creating Enroot container..."
    enroot create --name autops-rl /dss/dsshome1/05/ge26sav2/autops-rl/nvidia+pytorch+25.02-py3.sqsh
fi

# Start the container and run the commands
srun enroot start --root --mount /dss/dsshome1/05/ge26sav2/autops-rl:/workspace autops-rl bash -c "
    # Install dependencies
    echo 'Installing dependencies...'
    pip install --no-cache-dir -r /workspace/requirements.txt
    
    # Set environment variables
    export PYTHONPATH=\"\$PYTHONPATH:/workspace\"
    export WANDB_API_KEY=\"4b5c9c4ae3ffb150f67942dec8cc7d9f6fbcd558\"
    export WANDB_PROJECT=\"autops-rl\"
    export WANDB_ENTITY=\"sps-tum\"
    export RAY_DEDUP_LOGS=0
    
    # Print available GPUs
    nvidia-smi
    
    # Run the training script with reduced parameters
    python /workspace/train_scripts/custom_trainable.py \\
    --policy PPO \\
    --checkpoint-dir /workspace/checkpoints_case1_test_${SLURM_JOB_ID} \\
    --mode tune \\
    --iterations 5 \\
    --simulator-types \"everyone\" \\
    --num-env-runners 8 \\
    --num-envs-per-runner 2 \\
    --num-cpus-per-runner 1 \\
    --num-gpus-per-runner 0 \\
    --num-learners 1 \\
    --num-gpus-per-learner 1 \\
    --seeds \"42\" \\
    --num-samples-hyperparameter-tuning 3 \\
    --max-iterations-hyperparameter-tuning 3 \\
    --grace-period-hyperparameter-tuning 2 \\
    --num-targets 5 \\
    --num-observers 5 \\
    --time-step 1 \\
    --duration 3600 \\
    --reward-type case1
" 