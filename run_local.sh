#!/bin/bash

# Source conda for this shell session
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate autops-rl

# Set environment variables
export WANDB_API_KEY="4b5c9c4ae3ffb150f67942dec8cc7d9f6fbcd558"
export WANDB_PROJECT="autops-rl"
export WANDB_ENTITY="sps-tum"

# Add the project root to PYTHONPATH - use quotes to handle spaces in path
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Create a local checkpoint directory with absolute path
# Use quotes to handle spaces in the path
LOCAL_CHECKPOINT_DIR="$(pwd)/checkpoints_local_tune"
mkdir -p "$LOCAL_CHECKPOINT_DIR"

# Add this line before ray.init() in your script
export RAY_DEDUP_LOGS=0

# Run hyperparameter tuning with optimized settings
python train_scripts/custom_trainable.py \
  --policy PPO \
  --mode tune \
  --checkpoint-dir "$LOCAL_CHECKPOINT_DIR" \
  --iterations 100 \
  --simulator-types "everyone" \
  --num-env-runners 32 \
  --num-envs-per-runner 1 \
  --num-cpus-per-runner 3 \
  --num-gpus-per-runner 0 \
  --num-learners 1 \
  --num-gpus-per-learner 1 \
  --batch-size 8192 \
  --minibatch-size 512 \
  --rollout-fragment-length 256 \
  --batch-mode "truncate_episodes" \
  --checkpoint-freq 5 \
  --seeds "42" \
  --num-samples-hyperparameter-tuning 10 \
  --max-iterations-hyperparameter-tuning 15 \
  --grace-period-hyperparameter-tuning 5 \
  --num-targets 20 \
  --num-observers 20 \
  --time-step 1 \
  --duration 86400 \
  --reward-type case1

conda deactivate