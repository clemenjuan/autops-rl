#!/bin/bash
################################################################################
# SLURM directives##############################################################

# Specify the partition to run the job
#SBATCH -p lrz-hgx-h100-94x4
# #SBATCH -p lrz-dgx-a100-80x8
# #SBATCH -p lrz-dgx-a100-80x4
# #SBATCH -p lrz-v100x2
# #SBATCH -p lrz-hpe-p100x4

# Request GPUs
#SBATCH --gres=gpu:4

# Request CPU cores (leave some for system overhead)
# #SBATCH --cpus-per-task=94

# Specify the file for outputs
#SBATCH -o autops_training_%j.out

# Specify the file for errors
#SBATCH -e autops_training_%j.err

# Set a time limit for the job (maximum of 2 days in LRZ AI Systems: --time=2-00:00:00)
#SBATCH --time=1-00:00:00

# Send an email when the job ends
#SBATCH --mail-type=END

# Email address for notifications
#SBATCH --mail-user=clemente.juan@tum.de

# Container
# This NVIDIA container contains basic python and pytorch dependencies ready to use
#SBATCH --container-image='/dss/dsshome1/05/ge26sav2/autops-rl/nvidia+pytorch+25.02-py3.sqsh'
#SBATCH --container-mounts=/dss/dsshome1/05/ge26sav2/autops-rl:/workspace


################################################################################
# Execution commands ###########################################################

# Print available GPUs
nvidia-smi

pip install pip
pip install torch
pip install -r requirements.txt


# Set Wandb credentials
export WANDB_API_KEY="4b5c9c4ae3ffb150f67942dec8cc7d9f6fbcd558"
export WANDB_PROJECT="autops-rl"
export WANDB_ENTITY="TUM"

# Add this line to your sbatch script
export RAY_DEDUP_LOGS=0

# Run the training script
python train_scripts/custom_trainable.py \
--policy PPO \
--checkpoint-dir /workspace/checkpoints_${SLURM_JOB_ID} \
--tune \
--iterations 10000 \
--simulator-types "everyone,centralized,decentralized" \
--num-env-runners 90 \
--num-envs-per-runner 1 \
--num-cpus-per-runner 1 \
--num-gpus-per-runner 0 \
--num-learners 4 \
--num-gpus-per-learner 1 \
--seeds "42,43,44,45,46" \
--num-samples-hyperparameter-tuning 20 \
--max-iterations-hyperparameter-tuning 25 \
--grace-period-hyperparameter-tuning 10 \
--num-targets 20 \
--num-observers 20 \
--time-step 1 \
--duration 86400\
# --simulator-type "everyone"
