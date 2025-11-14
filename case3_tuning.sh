#!/bin/bash

# SLURM directives
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH -o autops_tuning_case3_%j.out
#SBATCH -e autops_tuning_case3_%j.err
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
    # WANDB_API_KEY should be set in your environment or as a GitLab CI/CD variable
    # export WANDB_API_KEY=\"your_key_here\"
    export WANDB_PROJECT=\"autops-rl\"
    export WANDB_ENTITY=\"sps-tum\"
    export RAY_DEDUP_LOGS=0
    
    # Print available GPUs
    nvidia-smi
    
    # Run the training script
    python /workspace/train_scripts/custom_trainable.py \\
    --policy PPO \\
    --checkpoint-dir /workspace/checkpoints_case3_tuning_${SLURM_JOB_ID} \\
    --mode tune \\
    --iterations 100 \\
    --simulator-types \"everyone,centralized,decentralized\" \\
    --num-env-runners 32 \\
    --num-envs-per-runner 1 \\
    --num-cpus-per-runner 2 \\
    --num-gpus-per-runner 0 \\
    --num-learners 4 \\
    --num-gpus-per-learner 1 \\
    --checkpoint-freq 10 \\
    --train-batch-size 8192 \\
    --minibatch-size 512 \\
    --rollout-fragment-length 256 \\
    --batch-mode "truncate_episodes" \\
    --seeds \"42\" \\
    --num-samples-hyperparameter-tuning 20 \\
    --max-iterations-hyperparameter-tuning 15 \\
    --grace-period-hyperparameter-tuning 5 \\
    --num-targets 20 \\
    --num-observers 20 \\
    --time-step 1 \\
    --duration 86400 \\
    --reward-type case3
" 