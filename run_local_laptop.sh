#!/bin/bash

# Source conda for this shell session
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate autops-rl

# Set environment variables
# WANDB_API_KEY should be set in your environment or .env file
# export WANDB_API_KEY="your_key_here"
# export WANDB_PROJECT="autops-rl"
# export WANDB_ENTITY="sps-tum"

# Add the project root to PYTHONPATH - use quotes to handle spaces in path
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Create a local checkpoint directory with absolute path
# Use quotes to handle spaces in the path
LOCAL_CHECKPOINT_DIR="$(pwd)/checkpoints_local_train"
mkdir -p "$LOCAL_CHECKPOINT_DIR"

# Add this line before ray.init() in your script
export RAY_DEDUP_LOGS=0

# Set wandb to offline mode
export WANDB_MODE=offline

# Run hyperparameter tuning with optimized settings
python train_scripts/custom_trainable.py \
  --policy PPO \
  --mode train \
  --checkpoint-dir "$LOCAL_CHECKPOINT_DIR" \
  --iterations 3 \
  --simulator-types "everyone" \
  --num-env-runners 4 \
  --num-envs-per-runner 1 \
  --num-cpus-per-runner 3 \
  --num-gpus-per-runner 0 \
  --num-learners 1 \
  --num-gpus-per-learner 1 \
  --train-batch-size 2048 \
  --minibatch-size 256 \
  --rollout-fragment-length 128 \
  --batch-mode "truncate_episodes" \
  --checkpoint-freq 5 \
  --seeds "42" \
  --num-targets 20 \
  --num-observers 20 \
  --time-step 1 \
  --duration 100 \
  --reward-type case1

conda deactivate