# AUTOPS-RL: Autonomous Satellite Coordination with Reinforcement Learning

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reinforcement learning framework for autonomous satellite coordination and real-time decision-making. This repository contains the implementation of a multi-agent reinforcement learning approach for coordinating satellite systems in dynamic and partially observable environments.

## Current Implementation

The framework includes comprehensive training, evaluation, and analysis capabilities:

- **Multi-Case Training**: 4 reward cases (Individual Positive/Negative, Collective Positive/Negative) implemented and trained
- **Sensitivity Analysis**: Systematic evaluation across multiple alpha values (0.0, 0.1, 0.5, 1.0) for reward scaling of the mission goal term
- **Baseline Methods**: Rule-Based and MIP centralized baseline policies for performance comparison
- **Cross-Network Evaluation**: Testing across centralized, constrained decentralized, and fully decentralized coordination types
- **Publication-Ready Analysis**: Professional visualizations, LaTeX tables, and statistical analysis tools
- **Comprehensive Benchmarking**: Automated evaluation system with organized result management

**Available Data**: All sensitivity analysis results and baseline comparisons are available in the `results/` directory as compressed .zip files. **Before running the comprehensive analysis, you must first unzip the required result files.**

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
  - [Sensitivity Analysis Benchmarking](#sensitivity-analysis-benchmarking)
  - [Baseline Methods Benchmarking](#baseline-methods-benchmarking)
  - [HPC Benchmarking Scripts](#hpc-benchmarking-scripts)
  - [Results Organization](#results-organization)
- [Sensitivity Analysis](#sensitivity-analysis)
  - [Individual vs Collective Reward Study](#individual-vs-collective-reward-study)
  - [Benchmarking Sensitivity Results](#benchmarking-sensitivity-results)
  - [Training Scripts](#training-scripts)
  - [Analysis](#analysis)
- [Comprehensive Analysis](#comprehensive-analysis)
  - [Quality Outputs](#quality-outputs)
  - [Usage](#usage-1)
  - [Generated Outputs](#generated-outputs)
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
- **Comprehensive Benchmarking**: Evaluate trained RL policies against rule-based and MIP centralized baseline methods with detailed performance metrics
- **Sensitivity Analysis Benchmarking**: Automated evaluation of trained policies across different reward parameter configurations
- **Intelligent Checkpoint Detection**: Robust checkpoint selection with proper seed matching and fallback handling
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

The benchmarking system provides comprehensive evaluation using two specialized scripts: one for RL policies sensitivity analysis and another for baseline methods comparison.

### Sensitivity Analysis Benchmarking

For evaluating trained RL policies across different alpha values:

```bash
# Test specific bonus groups
python benchmark_policies_sensitivity.py --bonus-groups bonus01 --episodes 15
python benchmark_policies_sensitivity.py --bonus-groups bonus0 --episodes 15
python benchmark_policies_sensitivity.py --bonus-groups bonus05 --episodes 15
python benchmark_policies_sensitivity.py --bonus-groups bonus10 --episodes 15

# Test all bonus groups together
python benchmark_policies_sensitivity.py --bonus-groups all --episodes 15

# Quick testing with fewer episodes
python benchmark_policies_sensitivity.py --bonus-groups bonus01 --episodes 2 --max-steps 50
```

### Baseline Methods Benchmarking

For evaluating Rule-Based and MIP baseline policies:

```bash
# Standard baseline benchmark
python benchmark_baselines_only.py --episodes 15

# Quick testing
python benchmark_baselines_only.py --episodes 2 --max-steps 50
```

### HPC Benchmarking Scripts

For running on the LRZ cluster:

```bash
# Submit RL policies benchmarks (run separately for each bonus group)
sbatch run_benchmark_bonus01.sh  # Baseline alpha=0.1
sbatch run_benchmark_bonus0.sh   # Alpha=0.0
sbatch run_benchmark_bonus05.sh  # Alpha=0.5
sbatch run_benchmark_bonus10.sh  # Alpha=1.0

# Submit baseline policies benchmark (can run in parallel)
sbatch run_benchmark_baselines.sh  # Rule-Based and MIP methods
```

**Key Features:**
- **Intelligent Checkpoint Detection**: Automatically finds trained models for each case/bonus combination
- **Cross-Network Evaluation**: Tests all policies across centralized, constrained decentralized, and fully decentralized coordination
- **Organized Results**: Creates timestamped experiment folders with comprehensive metadata
- **Robust Error Handling**: Continues evaluation even if individual policies fail to load

**Simulator Types Evaluated:**
- **`centralized`**: Single relay satellite handles all inter-satellite communications
- **`decentralized`**: Satellites communicate based on compatible communication bands  
- **`everyone`**: All satellites can communicate directly with each other

### Results Organization

Results are automatically organized in timestamped experiment folders:

```
results/
├── baselines_backup_20250613_014229/     # Centralized numerical methods results
│   ├── baseline_results.json
│   └── experiment_metadata.json
├── bonus0_backup_20250612_044111/        # Alpha=0.0 sensitivity results
│   ├── sensitivity_results.json
│   └── experiment_metadata.json
├── bonus01_backup_20250612_064400/       # Alpha=0.1 sensitivity results
├── bonus05_backup_20250612_064312/       # Alpha=0.5 sensitivity results
└── bonus10_backup_20250612_064531/       # Alpha=1.0 sensitivity results
```

## Sensitivity Analysis

### Individual vs Collective Reward Study

The sensitivity analysis framework investigates how reward function parameters affect the performance comparison between individual and collective reinforcement learning approaches.

**Research Question**: How does the magnitude of reward bonuses affect the performance comparison between individual reward schemes (Cases 1&2) and collective reward schemes (Cases 3&4)?

**Parameters Under Study**:
- `targets_bonus_factor`: Individual reward bonus for observed targets (Cases 1&2)  
- `final_targets_bonus`: Collective reward bonus for globally observed targets (Cases 3&4)

**Test Conditions**: Both parameters are varied together to ensure fair comparison:

| Test | targets_bonus_factor | final_targets_bonus | Status |
|------|---------------------|-------------------|--------|
| 1    | 0.0                 | 0.0               | ✅ Available |
| 2    | 0.1                 | 0.1               | ✅ Available |
| 3    | 0.5                 | 0.5               | ✅ Available |
| 4    | 1.0                 | 1.0               | ✅ Available |

### Benchmarking Sensitivity Results

The sensitivity analysis uses specialized benchmarking scripts that automatically evaluate all trained sensitivity models:

**HPC Benchmarking Scripts**:
```bash
# Submit individual bonus group benchmarks (RL policies only)
sbatch run_benchmark_bonus01.sh  # Baseline (bonus=0.1)
sbatch run_benchmark_bonus0.sh   # No bonus (bonus=0.0) 
sbatch run_benchmark_bonus05.sh  # Medium bonus (bonus=0.5)
sbatch run_benchmark_bonus10.sh  # High bonus (bonus=1.0)

# Submit baseline policies benchmark (can run in parallel)
sbatch run_benchmark_baselines.sh  # MIP and RuleBased methods
```

**Local Benchmarking**:
```bash
# Test specific bonus groups locally
python benchmark_policies_sensitivity.py --bonus-groups bonus01 --episodes 15
python benchmark_baselines_only.py --episodes 15
```

**Features**:
- **Intelligent Checkpoint Detection**: Automatically finds the correct seeds for each case/bonus combination
- **Cross-Network Evaluation**: Tests all policies across everyone/centralized/decentralized simulators
- **Separate Baseline Benchmarking**: Independent evaluation of MIP and RuleBased baselines
- **Organized Results**: Creates timestamped result folders with comprehensive metadata
- **Quick Testing**: Built-in quick tests to verify checkpoint loading before full evaluation
- **Robust Error Handling**: Continues evaluation even if individual policies fail to load

**Results Integration**:
All benchmarking results are saved in the `results/` directory and can be combined using the comprehensive analysis system.

### Training Scripts

The sensitivity analysis uses individual training scripts for each case/bonus combination to stay within HPC time limits:

**Available Training Scripts**:
- **Case 1**: `case1_bonus0.sh`, `case1_bonus05.sh`, `case1_bonus10.sh`
- **Case 2**: `case2_bonus0.sh`, `case2_bonus05.sh`, `case2_bonus10.sh`  
- **Case 3**: `case3_bonus0.sh`, `case3_bonus05.sh`, `case3_bonus10.sh`
- **Case 4**: `case4_bonus0.sh`, `case4_bonus05.sh`, `case4_bonus10.sh`

**Quick Submission**:
```bash
# Submit individual jobs
sbatch case1_bonus0.sh
sbatch case1_bonus05.sh
# ... (submit remaining 10 scripts)

# Monitor progress
squeue -u $USER
tail -f autops_case*_bonus*_*.out
```

**Computational Requirements**:
- 12 individual jobs (4 cases × 3 bonus levels)
- ~25 hours per job (5 seeds × 5h each)
- Each job fits within 2-day SLURM limits
- Can run in parallel if nodes available

### Analysis

After training completes, the benchmarking system automatically:

1. **Detects Available Checkpoints**: Intelligently finds trained models for each case/bonus combination
2. **Selects Correct Seeds**: Uses the specific seeds defined for each configuration  
3. **Evaluates Cross-Network Performance**: Tests all policies across different simulator types
4. **Generates Comprehensive Reports**: Creates detailed analysis with performance comparisons

**Expected Insights**:
1. How bonus magnitude affects mission completion rates
2. Whether individual vs collective performance gaps vary with reward intensity  
3. Resource management trade-offs at different bonus levels
4. Behavioral changes in action distributions
5. Cross-network generalization patterns across bonus levels

## Comprehensive Analysis

The comprehensive analysis system provides automated outputs. It combines baseline method results (Rule-Based and MIP) with sensitivity analysis data across different alpha values to generate professional-quality visualizations and LaTeX tables.

### Journal-Quality Outputs

**Key Features**:
- **Visualizations**: Three plots with consistent styling and structure
- **Unified Plot Structure**: All plots use identical alpha grouping (Baseline="-", 0.0, 0.1, 0.5, 1.0) and method organization for easy comparison
- **Method Acronyms**: Clean method identification (R-B, M, IP, IN, CP, CN) with visual grouping rectangles
- **LaTeX Integration**: Direct LaTeX table generation with proper formatting and statistical notation
- **Comprehensive Coverage**: Includes all coordination types (Centralized, Constrained Decentralized, Fully Decentralized)

### Usage

**Prerequisites**: Completed sensitivity analysis benchmarking with baseline methods evaluation.

**Setup**: First, unzip the required result files:
```bash
# Unzip the baseline results
cd results
unzip baselines_backup_20250613_014229.zip

# Unzip the sensitivity analysis results
unzip bonus0_backup_20250612_044111.zip
unzip bonus01_backup_20250612_064400.zip
unzip bonus05_backup_20250612_064312.zip
unzip bonus10_backup_20250612_064531.zip

# Return to project root
cd ..
```

**Command Structure**:
```bash
python comprehensive_analysis.py --baselines <baseline_results.json> --bonus-files <bonus_results_1.json> <bonus_results_2.json> ... --output-dir <output_directory>
```

**Example**:
```bash
python comprehensive_analysis.py \
  --baselines results/baselines_backup_20250613_014229/baseline_results.json \
  --bonus-files \
    results/bonus0_backup_20250612_044111/sensitivity_results.json \
    results/bonus01_backup_20250612_064400/sensitivity_results.json \
    results/bonus05_backup_20250612_064312/sensitivity_results.json \
    results/bonus10_backup_20250612_064531/sensitivity_results.json \
  --output-dir analysis
```

**Input Requirements**:
- **Baseline Results**: JSON file from `benchmark_baselines_only.py` containing Rule-Based and MIP results across all coordination types
- **Bonus Files**: Multiple JSON files from sensitivity benchmarking (`benchmark_policies_sensitivity.py`) with different alpha values (0.0, 0.1, 0.5, 1.0)
- **Output Directory**: Target directory for generated analysis files

### Generated Outputs

**1. LaTeX Performance Table** (`comprehensive_performance_table.tex`):
- Complete comparison of all methods across coordination types
- Mean ± standard deviation format with proper statistical notation
- Professional formatting ready for direct inclusion in LaTeX documents
- Organized by coordination type with clear method grouping

**2. Action Distribution Analysis** (`action_distribution_analysis.png`):
- Stacked bar charts showing action breakdowns (Idle/Communicate/Observe) with percentage labels
- Identical structure to performance plots for easy comparison
- Method color borders and visual grouping rectangles
- Combined legend showing both methods and action types

**3. Mission Accomplishment Boxplots** (`mission_accomplishment_boxplot.png`):
- Individual boxes for each method-alpha combination with visual method grouping
- Same alpha organization and method acronyms as other plots
- Professional styling with shared axis labels and comprehensive legend
- Clear performance comparison across all methods and configurations

**4. Resource Utilization Boxplots** (`resource_utilization_boxplot.png`):
- Identical structure to mission accomplishment plot but showing remaining resources
- Same method grouping, alpha organization, and professional styling
- Enables direct comparison between mission success and resource efficiency

**5. Unified Dataset** (`unified_analysis_data.csv`):
- Combined processed dataset with all methods and configurations
- Standardized column naming and data formatting
- Ready for additional statistical analysis or custom visualizations

**Current Output Organization**:
```
analysis/
├── comprehensive_performance_table.tex    # LaTeX table for journal inclusion
├── action_distribution_analysis.png       # Action breakdown with percentage labels
├── mission_accomplishment_boxplot.png     # Mission success comparison
├── resource_utilization_boxplot.png       # Resource efficiency comparison  
└── unified_analysis_data.csv              # Combined processed dataset
```

**Available Results**: All sensitivity analysis data and baseline comparisons are available in the `results/` directory as compressed .zip files. Unzip the required files before running the comprehensive analysis.

## Code Structure

```
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
├── LICENSE                            # MIT License

# Training Scripts (HPC)
├── sbatch_autops-rl.sh                # SLURM batch script for HPC
├── case1_tuning.sh                    # SLURM script for case1 hyperparameter tuning
├── case2_tuning.sh                    # SLURM script for case2 hyperparameter tuning
├── case3_tuning.sh                    # SLURM script for case3 hyperparameter tuning
├── case4_tuning.sh                    # SLURM script for case4 hyperparameter tuning
├── case1_training.sh                  # SLURM script for case1 training
├── case2_training.sh                  # SLURM script for case2 training
├── case3_training.sh                  # SLURM script for case3 training
├── case4_training.sh                  # SLURM script for case4 training

# Sensitivity Analysis Training Scripts
├── case1_bonus0.sh                    # Case1 with alpha=0.0
├── case1_bonus05.sh                   # Case1 with alpha=0.5
├── case1_bonus10.sh                   # Case1 with alpha=1.0
├── case2_bonus0.sh                    # Case2 with alpha=0.0
├── case2_bonus05.sh                   # Case2 with alpha=0.5
├── case2_bonus10.sh                   # Case2 with alpha=1.0
├── case3_bonus0.sh                    # Case3 with alpha=0.0
├── case3_bonus05.sh                   # Case3 with alpha=0.5
├── case3_bonus10.sh                   # Case3 with alpha=1.0
├── case4_bonus0.sh                    # Case4 with alpha=0.0
├── case4_bonus05.sh                   # Case4 with alpha=0.5
├── case4_bonus10.sh                   # Case4 with alpha=1.0

# Benchmarking Scripts (HPC)
├── run_benchmark_bonus01.sh           # Benchmark alpha=0.1 (baseline)
├── run_benchmark_bonus0.sh            # Benchmark alpha=0.0
├── run_benchmark_bonus05.sh           # Benchmark alpha=0.5
├── run_benchmark_bonus10.sh           # Benchmark alpha=1.0
├── run_benchmark_baselines.sh         # Benchmark Rule-Based and MIP baselines

# Local Development
├── run_local.sh                       # Local test script
├── run_local_laptop.sh                # Local laptop test script
├── setup_container.sh                 # Container setup script

# Core Implementation
├── train_scripts/                     # Training scripts
│   └── custom_trainable.py            # Main training script
├── src/                               # Source code
│   ├── envs/                          # Environment implementation
│   │   └── FSS_env_v1.py              # Federated Satellite System environment
│   ├── subsystems/                    # Subsystem implementation
│   ├── satellites.py                  # Satellite implementation
│   ├── simulator.py                   # Simulator implementation
│   └── rewards.py                     # Reward function implementations
├── eval_scripts/                      # Evaluation scripts
│   └── evaluate_policy.py             # Policy evaluation script
├── utils/                             # Utility functions
├── validation_scripts/                # Validation and testing scripts

# Benchmarking and Analysis
├── benchmark_policies_sensitivity.py  # RL policies sensitivity benchmarking
├── benchmark_baselines_only.py        # Baseline methods benchmarking
├── comprehensive_analysis.py           # Comprehensive analysis with publication-ready outputs
├── rule_based_policy.py               # Rule-based baseline policy
├── mip_policy.py                      # MIP baseline policy
├── benchmark_utils.py                 # Benchmarking utilities

# Results and Analysis
├── results/                           # Experiment results (organized by timestamp)
│   ├── baselines_backup_20250613_014229/     # Rule-Based and MIP results
│   │   ├── baseline_results.json
│   │   └── experiment_metadata.json
│   ├── bonus0_backup_20250612_044111/        # Alpha=0.0 sensitivity results
│   ├── bonus01_backup_20250612_064400/       # Alpha=0.1 sensitivity results
│   ├── bonus05_backup_20250612_064312/       # Alpha=0.5 sensitivity results
│   └── bonus10_backup_20250612_064531/       # Alpha=1.0 sensitivity results
├── analysis/                          # Publication-ready outputs
│   ├── comprehensive_performance_table.tex   # LaTeX table
│   ├── action_distribution_analysis.png      # Action breakdown plots
│   ├── mission_accomplishment_boxplot.png    # Mission success plots
│   ├── resource_utilization_boxplot.png      # Resource efficiency plots
│   └── unified_analysis_data.csv             # Combined dataset
├── checkpoints/                       # Trained model checkpoints
├── checkpoints_local_train/           # Local training checkpoints
└── EUCASS2025/                        # Conference-specific materials
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
  - `--bonus-groups`: Which bonus groups to test for sensitivity analysis ("bonus01", "bonus0", "bonus05", "bonus10", "all")
      - `--include-baselines`: Include MIP and RuleBased baselines in benchmarking (optional, or run separately via `run_benchmark_baselines.sh`)

## PPO Training Details

We use Proximal Policy Optimization (PPO) with the following key features:

- **Multi-agent coordination**: Each satellite is controlled by the same policy but receives different observations
- **Distributed training**: Leverages Ray's distributed computing capabilities
- **Experience replay**: Efficient sampling and learning from collected experiences
- **Hyperparameter tuning**: Automated search for optimal training parameters

## Baseline Policies

The benchmarking system includes two established centralized baseline methods for comparison against the trained RL policies:

### Rule-Based Centralized Method
A heuristic centralized approach that makes decisions based on battery and storage levels:
- **Low battery** (< 30%): Idle to conserve energy
- **High storage** (> 70%): Prioritize communication to offload data
- **Good battery (> 80%) + low storage (< 30%)**: Prioritize observation when targets are available
- **Good battery (> 80%) + low storage (< 30%)**: Prioritize observation when targets are available
- **Balanced approach**: Choose between observation and communication based on current state

### MIP Centralized Method  
A simplified Mixed Integer Programming centralized approach that:
- Formulates action selection as a utility maximization problem
- Considers observation utility (unobserved targets × pointing accuracy)
- Considers communication utility (storage level × communication ability)
- Applies the same resource constraints as the rule-based centralized method

## Results

Training results are logged to Weights & Biases for easy visualization and comparison. The system tracks metrics such as:

- Episode return mean
- Episode length mean
- Training throughput
- Policy loss
- Value function loss

Benchmarking results provide comprehensive performance analysis including:

- **Cross-network generalization studies**: How well policies trained on one simulator type perform on others
- **Sensitivity analysis insights**: Performance trends across different reward parameter configurations
- **Computational efficiency metrics**: Performance vs. computational cost analysis
- **Action distribution analysis**: Understanding policy behavior patterns
- **Resource utilization patterns**: Battery and storage management strategies
- **Scaling analysis**: Performance trends with increasing problem size

All benchmarking results are automatically organized in timestamped experiment folders for easy management and reproducibility.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@repository{autops-rl2025,
  title={AUTOPS-RL: Autonomous Satellite Coordination with Reinforcement Learning},
  subtitle={A Multi-Agent Framework for Real-Time Decision-Making in Dynamic Environments},
  url={https://gitlab.lrz.de/sps/autops/autops-rl},
  author={Clemente J. Juan Oliver},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
