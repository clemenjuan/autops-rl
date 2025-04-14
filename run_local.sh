#!/bin/bash

# Source conda for this shell session
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate autops-rl

# Set environment variables
export WANDB_API_KEY="4b5c9c4ae3ffb150f67942dec8cc7d9f6fbcd558"
export WANDB_PROJECT="autops-rl"
export WANDB_ENTITY="TUM"
# echo "WANDB_API_KEY: $WANDB_API_KEY"
# echo "WANDB_PROJECT: $WANDB_PROJECT"
# echo "WANDB_ENTITY: $WANDB_ENTITY"

# Add the project root to PYTHONPATH - use quotes to handle spaces in path
export PYTHONPATH="$PYTHONPATH:/mnt/c/Users/Clemente/OneDrive - TUM/Research/autops-rl"

# Create a local checkpoint directory with absolute path
# Use quotes to handle spaces in the path
LOCAL_CHECKPOINT_DIR="$(pwd)/checkpoints_local_test"
mkdir -p "$LOCAL_CHECKPOINT_DIR"

# Add this line before ray.init() in your script
export RAY_DEDUP_LOGS=0

# Run local test with minimal configuration
python train_scripts/custom_trainable.py \
  --policy PPO \
  --checkpoint-dir "$LOCAL_CHECKPOINT_DIR" \
  --iterations 100 \
  --simulator-types "everyone" \
  --num-env-runners 4 \
  --num-envs-per-runner 1 \
  --num-cpus-per-runner 1 \
  --num-gpus-per-runner 0 \
  --num-learners 1 \
  --num-gpus-per-learner 1 \
  --seeds "42" \
  --num-targets 20 \
  --num-observers 20 \
  --time-step 1 \
  --duration 86400
    #--simulator_type "everyone" 
    #--tune \
    #--num_samples 20 \
    #--max_iterations 25 \
    #--grace_period 10 \

conda deactivate