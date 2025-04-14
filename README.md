# AUTOPS-RL: Autonomous Satellite Coordination with Reinforcement Learning

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reinforcement learning framework for autonomous satellite coordination and real-time decision-making. This repository contains the implementation of a multi-agent reinforcement learning approach for coordinating satellite systems in dynamic and partially observable environments.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Local Development Environment](#local-development-environment)
  - [HPC Environment (LRZ)](#hpc-environment-lrz)
- [Usage](#usage)
  - [Training](#training)
  - [Local Testing](#local-testing)
  - [Evaluation](#evaluation)
- [Code Structure](#code-structure)
- [Configuration](#configuration)
- [PPO Training Details](#ppo-training-details)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Introduction

AUTOPS-RL is a framework for training reinforcement learning policies for autonomous satellite systems. The system uses distributed PPO (Proximal Policy Optimization) to train policies that can coordinate multiple satellites to achieve a common goal in various scenarios, including centralized, decentralized, and hybrid network approaches.

## Features

- **Multiple Simulator Types**: Support for different coordination approaches (everyone, centralized, decentralized)
- **Distributed Training**: Efficient training using Ray's distributed computing framework
- **Configurable Environment**: Adjustable number of targets, observers, and simulation parameters
- **Multi-seed Training**: Support for training with multiple random seeds for robust evaluation
- **Wandb Integration**: Experiment tracking and visualization with Weights & Biases

## Installation

### Local Development Environment

For local development and testing, we recommend setting up a conda environment:

```bash
# Create a new conda environment
conda create -n autops-rl python=3.12
conda activate autops-rl

# Clone the repository
git clone https://gitlab.lrz.de/sps/autops/autops-rl.git
cd autops-rl

# Install PyTorch
pip install torch==2.6.0

# Install dependencies
pip install -r requirements.txt

# Deactivate the conda environment
conda deactivate
```



This local setup is useful for quick testing and development before running full-scale training on HPC environments. This environment is used by the `run_local.sh` script (see below).


### HPC Environment (LRZ AI Systems)

For training on the LRZ AI Systems cluster, we use SLURM batch scripts:

1. Upload your code to the LRZ system
2. Submit the job:
   ```bash
   sbatch template_sbatch.sh
   ```

## Usage

### Training on LRZ AI Systems

To train the model on the HPC cluster:

```bash
sbatch sbatch_autops-rl.sh
```

The SLURM script configures the environment and runs the training with appropriate parameters:

```bash
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
--num-cpus-per-learner 1 \
--seeds "42,43,44,45,46"
```

### Local Training
You can run the local training/tuning with:

```bash
# Make the script executable (Unix/Linux/macOS only)
chmod +x run_local.sh

# Run the script    
./run_local.sh
```

You will automatically activate the conda environment previously installed. You can adjust the parameters in the `run_local.sh` script similarly to the HPC training script to test different configurations.


### Evaluation

After training, you can evaluate the trained models:

```bash
python eval_scripts/evaluate_policy.py \
  --checkpoint-dir /path/to/checkpoints \
  --simulator-type "everyone" \
  --seed 42
```

## Code Structure

```
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
├── sbatch_autops-rl.sh           # SLURM batch script for HPC
├── run_local.sh                  # Local test script
├── template_sbatch.sh            # Template for SLURM batch script
├── train_scripts/                # Training scripts
│   └── custom_trainable.py       # Main training script
├── src/                          # Source code
│   └── envs/                     # Environment implementation
│       └── FSS_env.py            # Federated Satellite System environment implementation
│   └── subsystems/               # Subsystem implementation
│   └── satellites.py             # Satellite implementation
│   └── simulator.py              # Simulator implementation
├── eval_scripts/                 # Evaluation scripts
│   └── evaluate_policy.py        # Evaluation script   
├── utils/                        # Utility functions

```


## Configuration

The training process can be configured with various command-line arguments:

- **General Parameters**:
  - `WANDB_API_KEY`: Wandb API key (environment variable)
  - `WANDB_PROJECT`: Wandb project name (environment variable)
  - `WANDB_ENTITY`: Wandb entity name (environment variable)

- **Environment Parameters**:
  - `--num-targets`: Number of target satellites (default: 20)
  - `--num-observers`: Number of observer satellites (default: 20)
  - `--simulator-type`: string to specify the simulator type (default: "everyone"), options: "everyone", "centralized", "decentralized". It can be overridden by the `--simulator-types` argument.
  - `--time-step`: Simulation time step in seconds (default: 1)
  - `--duration`: Total simulation duration in seconds (default: 86400)

- **Training Parameters**:
  - `--policy`: The RL algorithm to use (default: "PPO")
  - `--checkpoint-dir`: Directory to save checkpoints (default: "./checkpoints")
  - `--tune`: Whether to perform hyperparameter tuning (flag)
  - `--num-samples-hyperparameter-tuning`: Number of hyperparameter samples to run (default: 20)
  - `--max-iterations-hyperparameter-tuning`: Maximum number of training iterations for hyperparameter tuning (default: 25)
  - `--grace-period-hyperparameter-tuning`: Grace period for early stopping for hyperparameter tuning (default: 10)
  - `--iterations`: Number of training iterations, only used if `--tune` is not set (default: 1000)
  - `--checkpoint`: Path to checkpoint file for continuing training
  - `--best-config`: Path to best config JSON file for continuing training
  - `--simulator-types`: Comma-separated list of simulator types to run
  - `--seeds`: Comma-separated list of random seeds (default: "42, 43, 44, 45, 46")

- **Distributed Training Parameters**:
  - `--num-env-runners`: Number of environment runners (default: 10)
  - `--num-envs-per-runner`: Number of environments per runner (default: 1)
  - `--num-cpus-per-runner`: Number of CPUs per runner (default: 1)
  - `--num-gpus-per-runner`: Number of GPUs per runner (default: 0)
  - `--num-learners`: Number of learners (default: 1)
  - `--num-gpus-per-learner`: Number of GPUs per learner (default: 1)
  - `--num-cpus-per-learner`: Number of CPUs per learner (default: 1)

## Results

Training results are logged to Weights & Biases for easy visualization and comparison. The system tracks metrics such as:

- Episode reward mean
- Episode length mean
- Training throughput
- Policy loss
- Value function loss

## Citation

If you use this code in your research, please cite our work:

```bibtex
@repository{autops-rl2025,
  title={AUTOPS-RL: Autonomous Satellite Systems with Reinforcement Learning},
  url={https://gitlab.lrz.de/sps/autops/autops-rl},
  author={Clemente J. Juan Oliver},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
