# AUTOPS-RL: Autonomous Satellite Coordination with Reinforcement Learning

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reinforcement learning framework for autonomous satellite coordination and real-time decision-making using multi-agent PPO.

## Current Implementation

- **Multi-Case Training**: 4 reward cases (Individual/Collective Positive/Negative) 
- **Sensitivity Analysis**: 6 scaling factors (0.0, 0.01, 0.05, 1.0, 2.0, 5.0)
- **Baseline Methods**: Rule-Based and MIP centralized policies
- **Cross-Network Evaluation**: Centralized, constrained decentralized, fully decentralized
- **Automated Analysis**: Publication-ready visualizations and LaTeX tables

## Quick Start

### Installation
```bash
conda create -n autops-rl python=3.12
conda activate autops-rl
git clone https://gitlab.lrz.de/sps/autops/autops-rl.git
cd autops-rl
pip install -r requirements.txt

# Set WANDB API key for experiment tracking (required)
export WANDB_API_KEY="your_key_here"
export WANDB_PROJECT="your_project_here"
export WANDB_ENTITY="your_entity_here"
```

### Local Testing
```bash
./run_local.sh
```

### HPC Training
```bash
# Basic training
sbatch sbatch_autops-rl.sh

# Sensitivity analysis training (6 scaling factors)
sbatch case1_bonus0.sh    # Alpha=0.0
sbatch case1_bonus05.sh   # Alpha=0.05  
sbatch case1_bonus10.sh   # Alpha=1.0
sbatch case1_bonus20.sh   # Alpha=2.0
sbatch case1_bonus50.sh   # Alpha=5.0
# (Repeat for cases 2-4)
```

## Benchmarking & Analysis

**Note**: Run benchmarking AFTER training is complete to evaluate the trained models.

### Run Benchmarks
```bash
# RL policies sensitivity analysis
python benchmark_policies_sensitivity.py --bonus-groups all --episodes 15

# Baseline methods
python benchmark_baselines_only.py --episodes 15

# HPC (submit separately for each bonus group)
sbatch run_benchmark_bonus0.sh   # Alpha=0.0
sbatch run_benchmark_bonus05.sh  # Alpha=0.05
sbatch run_benchmark_bonus10.sh  # Alpha=1.0
sbatch run_benchmark_bonus20.sh  # Alpha=2.0
sbatch run_benchmark_bonus50.sh  # Alpha=5.0
sbatch run_benchmark_baselines.sh # Rule-Based and MIP
```

### Generate Analysis
```bash
# Unzip results first
cd results
unzip baselines_backup_20250613_014229.zip
unzip bonus0_backup_20250612_044111.zip
unzip bonus01_backup_20250612_064400.zip
unzip bonus05_backup_20250612_064312.zip
unzip bonus10_backup_20250612_064531.zip
unzip bonus20_backup_20250625_050109.zip
unzip bonus50_backup_20250625_042845.zip
cd ..

# Generate comprehensive analysis
python comprehensive_analysis.py \
  --baselines results/baselines_backup_20250613_014229/baseline_results.json \
  --bonus-files \
    results/bonus0_backup_20250612_044111/sensitivity_results.json \
    results/bonus01_backup_20250612_064400/sensitivity_results.json \
    results/bonus05_backup_20250612_064312/sensitivity_results.json \
    results/bonus10_backup_20250612_064531/sensitivity_results.json \
    results/bonus20_backup_20250625_050109/sensitivity_results.json \
    results/bonus50_backup_20250625_042845/sensitivity_results.json \
  --output-dir analysis

# Generate training curves
python comprehensive_training_plots.py

# Visualize constellation
python constellation_viz.py
```

## Outputs

### Analysis Results (`analysis/`)
- `comprehensive_performance_table.tex` - LaTeX table for papers
- `action_distribution_analysis.png` - Action breakdown plots
- `mission_accomplishment_boxplot.png` - Mission success comparison
- `resource_utilization_boxplot.png` - Resource efficiency plots
- `unified_analysis_data.csv` - Combined dataset

### Training Curves
- `professional_training_curves_all_alpha.png` - 2x3 subplot layout
- `single_training_curves_all_alpha.png` - All curves combined

## Configuration

### Key Parameters
- `--num-targets`: Number of target satellites (default: 20)
- `--num-observers`: Number of observer satellites (default: 20)
- `--simulator-type`: "everyone", "centralized", "decentralized"
- `--reward-type`: "case1", "case2", "case3", "case4"
- `--seeds`: Comma-separated seeds (default: "42,43,44,45,46")
- `--bonus-groups`: "bonus0", "bonus05", "bonus10", "bonus20", "bonus50", "all"

### Reward Cases
- **Case 1**: Positive rewards for observed targets (individual)
- **Case 2**: Negative rewards for unobserved targets (individual)  
- **Case 3**: Individual rewards + global bonus for observed targets (collective)
- **Case 4**: Individual rewards + global bonus for unobserved targets (collective)

## Code Structure

```
├── src/                    # Core implementation
│   ├── envs/FSS_env_v1.py  # Environment
│   ├── satellites.py       # Satellite models
│   ├── simulator.py        # Simulator
│   └── rewards.py          # Reward functions
├── train_scripts/          # Training scripts
├── eval_scripts/           # Evaluation
├── utils/                  # Utilities
├── benchmark_*.py          # Benchmarking scripts
├── comprehensive_*.py      # Analysis scripts
├── case*_bonus*.sh         # HPC training scripts
└── run_benchmark_*.sh      # HPC benchmarking scripts
```

## Citation

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

MIT License - see LICENSE file for details.