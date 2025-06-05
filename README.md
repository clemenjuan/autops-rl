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
- [Baseline Policies](#baseline-policies)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Introduction

AUTOPS-RL is a framework for training reinforcement learning policies for autonomous satellite systems. The system uses distributed PPO (Proximal Policy Optimization) to train policies that can coordinate multiple satellites to achieve a common goal in various scenarios, including centralized, decentralized, and hybrid network approaches.

## Features

- **Multiple Simulator Types**: Support for different coordination approaches (everyone, centralized, decentralized)
- **Cross-Network Evaluation**: Comprehensive testing of policy generalization across different network topologies
- **Distributed Training**: Efficient training using Ray's distributed computing framework
- **Configurable Environment**: Adjustable number of targets, observers, and simulation parameters
- **Multi-seed Training**: Support for training with multiple random seeds for robust evaluation
- **Wandb Integration**: Experiment tracking and visualization with Weights & Biases
- **Multiple Reward Cases**: Four different reward structures (case1-case4) for exploring different learning approaches
- **Comprehensive Benchmarking**: Compare RL policies against rule-based and MIP baselines with detailed performance metrics
- **Fair Baseline Comparison**: Consistent resource thresholds across all baseline policies for fair evaluation
- **Organized Experiment Management**: Automatic organization of results and analysis in structured experiment folders
- **Advanced Analytics**: Scaling analysis, action distribution studies, and performance visualization

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

The benchmarking system provides comprehensive evaluation of trained RL policies against baseline methods across different network topologies and scales.

### Quick Testing

For rapid development and testing:

```bash
# Quick test with minimal configurations
python benchmark_policies.py --configs small --episodes 2 --max-steps 50

# Test specific policy types
python benchmark_policies.py --configs small --episodes 1 --max-steps 20
```

### Running Full Benchmarks

For comprehensive evaluation:

```bash
# Standard benchmark (recommended)
python benchmark_policies.py --configs standard --episodes 5

# Extended evaluation with longer episodes
python benchmark_policies.py --configs large --episodes 10
```

**Configuration Sets:**

- **`small`**: Quick testing with shorter episodes (500 time steps) across all three simulator types
- **`standard`**: Full-day simulations (86400 time steps) testing cross-network generalization
- **`large`**: Extended 2-day simulations (172800 time steps) for comprehensive cross-network evaluation

**Simulator Types Evaluated:**
- **`centralized`**: Single relay satellite handles all inter-satellite communications
- **`decentralized`**: Satellites communicate based on compatible communication bands  
- **`everyone`**: All satellites can communicate directly with each other

**Note**: All configurations use 20 agents and 100 targets to match the trained RL model's observation space. The focus is on **cross-network generalization** - evaluating how well policies trained on one simulator type perform across different network topologies.

### Analyzing Results

After running benchmarks, analyze the results:

```bash
# Analyze specific experiment
python analyze_results.py experiments/benchmark_YYYYMMDD_HHMMSS/benchmark_results.json

# This generates:
# - Performance comparison plots (NET per agent, mission completion %, resource usage)
# - Action distribution analysis (stacked bar charts by simulator type)
# - Simulator comparison tables (LaTeX format)
# - Comprehensive console summary
```

### Experiment Organization

Results are automatically organized in timestamped experiment folders:

```
experiments/
└── benchmark_YYYYMMDD_HHMMSS/           # Experiment timestamp
    ├── benchmark_results.json           # Raw benchmark data
    ├── experiment_metadata.json         # System info and configs
    ├── analysis/                        # Generated analysis
    │   ├── performance_comparison.png   # Main performance plots
    │   ├── action_distribution.png      # Stacked bar charts by simulator
    │   ├── summary_statistics.csv       # Statistical summary
    │   └── simulator_comparison_table.tex # LaTeX comparison table
    └── README.md                        # Experiment documentation
```

**Generated Visualizations:**

1. **Performance Comparison**: Boxplots comparing NET per agent, mission completion percentage, average resources remaining, and simulation time across policies and simulator types

2. **Action Distribution**: Stacked bar charts showing the percentage breakdown of actions (Idle/Communicate/Observe) for each policy across all three simulator types, providing clear insights into behavioral differences

3. **Statistical Tables**: LaTeX-formatted tables with mean±std statistics for easy inclusion in research papers

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
  - `--configs`: Configuration set ("small", "standard", "large")
  - `--episodes`: Number of episodes per configuration (default: 5)
  - `--max-steps`: Maximum steps per episode for testing

## PPO Training Details

We use Proximal Policy Optimization (PPO) with the following key features:

- **Multi-agent coordination**: Each satellite is controlled by the same policy but receives different observations
- **Distributed training**: Leverages Ray's distributed computing capabilities
- **Experience replay**: Efficient sampling and learning from collected experiences
- **Hyperparameter tuning**: Automated search for optimal training parameters

## Baseline Policies

The benchmarking system includes two carefully designed baseline policies for fair comparison:

### Rule-Based Policy
A heuristic policy that makes decisions based on battery and storage levels:
- **Low battery** (< 30%): Idle to conserve energy
- **High storage** (> 70%): Prioritize communication to offload data
- **Good battery (> 80%) + low storage (< 30%)**: Prioritize observation when targets are available
- **Balanced approach**: Choose between observation and communication based on current state

### MIP Policy  
A simplified Mixed Integer Programming approach that:
- Formulates action selection as a utility maximization problem
- Considers observation utility (unobserved targets × pointing accuracy)
- Considers communication utility (storage level × communication ability)
- Applies the same resource constraints as the rule-based policy

## Results

Training results are logged to Weights & Biases for easy visualization and comparison. The system tracks metrics such as:

- Episode return mean
- Episode length mean
- Training throughput
- Policy loss
- Value function loss

Benchmarking results provide comprehensive performance analysis including:

- **Cross-network generalization studies**: How well policies trained on one simulator type perform on others
- **Computational efficiency metrics**: Performance vs. computational cost analysis
- **Action distribution analysis**: Understanding policy behavior patterns
- **Resource utilization patterns**: Battery and storage management strategies
- **Scaling analysis**: Performance trends with increasing problem size

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
