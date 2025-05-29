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
- [Benchmarking](#benchmarking)
  - [Quick Testing](#quick-testing)
  - [Running Full Benchmarks](#running-full-benchmarks)
  - [Analyzing Results](#analyzing-results)
  - [Experiment Organization](#experiment-organization)
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
- **Multiple Reward Cases**: Four different reward structures (case1-case4) for exploring different learning approaches
- **Comprehensive Benchmarking**: Compare RL policies against rule-based and MIP baselines with detailed performance metrics
- **Cross-Network Testing**: Evaluate policy generalization across different network topologies
- **Organized Experiment Management**: Automatic organization of results and analysis in structured experiment folders

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

For hyperparameter tuning with specific reward cases:

```bash
sbatch case1_tuning.sh
sbatch case2_tuning.sh
sbatch case3_tuning.sh
sbatch case4_tuning.sh
```

### Local Testing

For quick testing on your local machine:

```bash
./run_local.sh
```

This runs a minimal training session locally to verify the setup.

### Evaluation

To evaluate a trained policy:

```bash
cd eval_scripts
python evaluate_policy.py --checkpoint-path /path/to/checkpoint --num-episodes 10
```

## Benchmarking

The benchmarking system compares RL policies against rule-based and MIP baselines across different network configurations. It automatically tests policy generalization by evaluating policies trained on one simulator type (e.g., "everyone") on all three simulator types ("everyone", "centralized", "decentralized").

### Prerequisites

Before running benchmarks, make sure to activate the conda environment:

```bash
conda activate autops-rl
```

### Quick Testing

```bash
# Test with small configuration (for development/testing)
python benchmark_policies.py --configs small --episodes 3 --max-steps 50
```

### Running Full Benchmarks

```bash
# Standard benchmarks (recommended for most comparisons)
python benchmark_policies.py --configs standard --episodes 10

# Large-scale benchmarks (for comprehensive analysis)
python benchmark_policies.py --configs large --episodes 5

# All configurations (comprehensive but time-consuming)
python benchmark_policies.py --configs all --episodes 5
```

### Analyzing Results

Results are automatically organized in timestamped experiment folders:

```bash
# Analyze a specific experiment
python analyze_results.py experiments/benchmark_20250529_113520/benchmark_results.json

# Or use the path suggested by the benchmark output
```

### Experiment Organization

Each benchmark run creates a self-contained experiment folder:

```
experiments/
└── benchmark_20250529_113520/
    ├── benchmark_results.json          # Raw results data
    ├── experiment_metadata.json        # Experiment configuration
    ├── README.md                       # Experiment description
    └── analysis/                       # Generated analysis
        ├── summary_statistics.csv
        ├── performance_table.tex
        ├── performance_comparison.png
        └── ...
```

### Organizing Existing Results

If you have old benchmark results files, you can organize them into the new structure:

```bash
# Organize existing results files
python organize_existing_results.py benchmark_results_*.json

# Delete original files after organizing (optional)
python organize_existing_results.py benchmark_results_*.json --delete-originals
```

## Code Structure

```
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
├── sbatch_autops-rl.sh           # SLURM batch script for HPC
├── case1_tuning.sh               # SLURM script for case1
├── case2_tuning.sh               # SLURM script for case2 
├── case3_tuning.sh               # SLURM script for case3
├── case4_tuning.sh               # SLURM script for case4
├── case1_test.sh                 # SLURM script for quick testing
├── run_local.sh                  # Local test script
├── template_sbatch.sh            # Template for SLURM batch script
├── train_scripts/                # Training scripts
│   └── custom_trainable.py       # Main training script
├── src/                          # Source code
│   └── envs/                     # Environment implementation
│       └── FSS_env_v1.py         # Federated Satellite System environment implementation
│   └── subsystems/               # Subsystem implementation
│   └── satellites.py             # Satellite implementation
│   └── simulator.py              # Simulator implementation
│   └── rewards.py                # Reward function implementations
├── eval_scripts/                 # Evaluation scripts
│   └── evaluate_policy.py        # Evaluation script   
├── utils/                        # Utility functions
├── experiments/                  # Organized experiment results
│   ├── benchmark_YYYYMMDD_HHMMSS/ # Individual experiment folders
│   │   ├── benchmark_results.json
│   │   ├── experiment_metadata.json
│   │   ├── README.md
│   │   └── analysis/
│   └── ...
├── benchmark_policies.py         # Main benchmarking script
├── analyze_results.py            # Results analysis script
├── organize_existing_results.py  # Script to organize old results
├── rule_based_policy.py          # Rule-based baseline policy
├── mip_policy.py                 # MIP baseline policy
└── benchmark_utils.py            # Benchmarking utilities
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
  - `--reward-type`: Reward function to use (default: "case1"), options: "case1", "case2", "case3", "case4"

- **Training Parameters**:
  - `--policy`: The RL algorithm to use (default: "PPO")
  - `--checkpoint-dir`: Directory to save checkpoints (default: "./checkpoints")
  - `--mode`: Operation mode - "tune", "train", or "tune_then_train" (default: "train")
  - `--num-samples-hyperparameter-tuning`: Number of hyperparameter samples to run (default: 20)
  - `--max-iterations-hyperparameter-tuning`: Maximum number of training iterations for hyperparameter tuning (default: 25)
  - `--grace-period-hyperparameter-tuning`: Grace period for early stopping for hyperparameter tuning (default: 10)
  - `--iterations`: Number of training iterations (default: 1000)
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

- **Benchmarking Parameters**:
  - `--configs`: Configuration set ("small", "standard", "large", "all")
  - `--episodes`: Number of episodes per configuration (default: 5)
  - `--max-steps`: Maximum steps per episode for testing

## PPO Training Details

We use Proximal Policy Optimization (PPO) with the following key features:

- **Multi-agent coordination**: Each satellite is controlled by the same policy but receives different observations
- **Distributed training**: Leverages Ray's distributed computing capabilities
- **Experience replay**: Efficient sampling and learning from collected experiences
- **Hyperparameter tuning**: Automated search for optimal training parameters

## Results

Training results are logged to Weights & Biases for easy visualization and comparison. The system tracks metrics such as:

- Episode return mean
- Episode length mean
- Training throughput
- Policy loss
- Value function loss

Benchmarking results provide comprehensive performance analysis including:

- Cross-network generalization studies
- Computational efficiency metrics
- Action distribution analysis
- Resource utilization patterns

All benchmarking results are automatically organized in timestamped experiment folders for easy management and reproducibility.

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
