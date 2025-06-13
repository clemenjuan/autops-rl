#!/bin/bash

# SLURM directives for Case 1 with bonus factors = 0
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH -o autops_case1_bonus0_%j.out
#SBATCH -e autops_case1_bonus0_%j.err
#SBATCH --time=2-00:00:00
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
    export WANDB_PROJECT=\"autops-rl-sensitivity\"
    export WANDB_ENTITY=\"sps-tum\"
    export RAY_DEDUP_LOGS=0
    
    # Print available GPUs
    nvidia-smi
    
    # Run the training script for Case 1 with bonus = 0
    python /workspace/train_scripts/custom_trainable.py \\
    --policy PPO \\
    --checkpoint-dir /workspace/checkpoints_case1_bonus0_${SLURM_JOB_ID} \\
    --mode train \\
    --iterations 100 \\
    --simulator-types \"everyone\" \\
    --num-env-runners 32 \\
    --num-envs-per-runner 1 \\
    --num-cpus-per-runner 2 \\
    --num-gpus-per-runner 0 \\
    --num-learners 4 \\
    --num-gpus-per-learner 1 \\
    --checkpoint-freq 20 \\
    --train-batch-size 4096 \\
    --minibatch-size 256 \\
    --rollout-fragment-length 128 \\
    --batch-mode \"truncate_episodes\" \\
    --lr 1e-4 \\
    --gamma 0.966 \\
    --lambda_val 0.964 \\
    --seeds \"42,43,44,45,46\" \\
    --num-targets 100 \\
    --num-observers 20 \\
    --time-step 1 \\
    --duration 86400 \\
    --reward-type case1 \\
    --reward-config '{\"targets_bonus_factor\": 0, \"final_targets_bonus\": 0}'
"
