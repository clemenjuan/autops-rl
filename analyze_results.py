import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_benchmark_results(file_path):
    """Load benchmark results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_summary_dataframe(results):
    """Create a pandas DataFrame from benchmark results"""
    rows = []
    
    for result in results:
        row = {
            'config': result['config_name'],
            'policy': result['policy_name'],
            'simulator_type': result.get('simulator_type', 'unknown'),  # Add simulator type
            'episode': result['episode_number'],
            'num_agents': result['num_agents'],
            'num_targets': result['num_targets'],
            'net_per_agent': result['metrics']['net_per_agent'],
            'mission_percentage': result['metrics']['mission_percentage'],
            'average_resources': result['metrics']['average_resources_left'],
            'simulation_time': result['metrics']['simulation_time'],
            'total_reward': result['metrics']['total_reward'],
            'action_0_percent': result['metrics']['action_distribution']['action_0'],  # Idle
            'action_1_percent': result['metrics']['action_distribution']['action_1'],  # Communicate
            'action_2_percent': result['metrics']['action_distribution']['action_2'],  # Observe
            'step_count': result.get('step_count', 0),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_latex_table(df, output_file):
    """Generate LaTeX table from summary statistics"""
    
    # Group by config and policy, calculate means and stds
    summary_stats = df.groupby(['config', 'policy']).agg({
        'net_per_agent': ['mean', 'std'],
        'mission_percentage': ['mean', 'std'],
        'average_resources': ['mean', 'std'],
        'simulation_time': ['mean', 'std'],
    }).round(3)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Performance Comparison Across Different Configurations}")
    latex_content.append("\\label{tab:performance_comparison}")
    latex_content.append("\\medskip")
    latex_content.append("")
    latex_content.append("\\begin{tabular}{lcccc}")
    latex_content.append("\\textbf{Method} & \\textbf{NET per Action} & \\textbf{Mission (\\%)} & \\textbf{Avg Resources} & \\textbf{Sim Steps} \\\\")
    latex_content.append("\\hline")
    
    # Group by configuration
    configs = df['config'].unique()
    for config in configs:
        # Extract agent and target numbers from config name
        parts = config.split('_')
        agents = parts[0]
        targets = parts[2]
        
        latex_content.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{agents} Agents, {targets} Targets}}}} \\\\")
        
        config_data = summary_stats.loc[config]
        
        for policy in config_data.index:
            net_mean = config_data.loc[policy, 'net_per_agent_mean']
            net_std = config_data.loc[policy, 'net_per_agent_std']
            mission_mean = config_data.loc[policy, 'mission_percentage_mean']
            mission_std = config_data.loc[policy, 'mission_percentage_std']
            resources_mean = config_data.loc[policy, 'average_resources_mean']
            resources_std = config_data.loc[policy, 'average_resources_std']
            sim_time_mean = config_data.loc[policy, 'simulation_time_mean']
            sim_time_std = config_data.loc[policy, 'simulation_time_std']
            
            latex_content.append(
                f"{policy} & {net_mean:.3f}±{net_std:.3f} & {mission_mean:.1f}±{mission_std:.1f} & "
                f"{resources_mean:.2f}±{resources_std:.2f} & {sim_time_mean:.1f}±{sim_time_std:.1f} \\\\"
            )
        
        latex_content.append("\\hline")
    
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ LaTeX table saved to {output_file}")


def generate_action_distribution_table(df, output_file):
    """Generate a detailed action distribution table"""
    
    # Group by config and policy, calculate action distribution means
    action_stats = df.groupby(['config', 'policy']).agg({
        'action_0_percent': ['mean', 'std'],  # Idle
        'action_1_percent': ['mean', 'std'],  # Communicate
        'action_2_percent': ['mean', 'std'],  # Observe
    }).round(2)
    
    # Flatten column names
    action_stats.columns = ['_'.join(col).strip() for col in action_stats.columns]
    
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Action Distribution Across Different Methods}")
    latex_content.append("\\label{tab:action_distribution}")
    latex_content.append("\\medskip")
    latex_content.append("")
    latex_content.append("\\begin{tabular}{lcccc}")
    latex_content.append("\\textbf{Method} & \\textbf{Idle (\\%)} & \\textbf{Communicate (\\%)} & \\textbf{Observe (\\%)} & \\textbf{Total} \\\\")
    latex_content.append("\\hline")
    
    # Group by configuration
    configs = df['config'].unique()
    for config in configs:
        # Extract agent and target numbers from config name
        parts = config.split('_')
        agents = parts[0]
        targets = parts[2]
        
        latex_content.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{agents} Agents, {targets} Targets}}}} \\\\")
        
        config_data = action_stats.loc[config]
        
        for policy in config_data.index:
            idle_mean = config_data.loc[policy, 'action_0_percent_mean']
            idle_std = config_data.loc[policy, 'action_0_percent_std']
            comm_mean = config_data.loc[policy, 'action_1_percent_mean']
            comm_std = config_data.loc[policy, 'action_1_percent_std']
            obs_mean = config_data.loc[policy, 'action_2_percent_mean']
            obs_std = config_data.loc[policy, 'action_2_percent_std']
            
            total = idle_mean + comm_mean + obs_mean
            
            latex_content.append(
                f"{policy} & {idle_mean:.1f}±{idle_std:.1f} & {comm_mean:.1f}±{comm_std:.1f} & "
                f"{obs_mean:.1f}±{obs_std:.1f} & {total:.1f} \\\\"
            )
        
        latex_content.append("\\hline")
    
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Action distribution table saved to {output_file}")


def generate_simulator_comparison_table(df, output_file):
    """Generate a LaTeX table comparing performance across simulators and policies"""
    
    # Define simulator type mappings and order
    simulator_mapping = {
        'centralized': 'Centralized coordination',
        'decentralized': 'Constrained decentralized coordination', 
        'everyone': 'Fully decentralized coordination'
    }
    
    simulator_order = ['centralized', 'decentralized', 'everyone']
    
    # Define policy name mappings
    policy_mapping = {
        'case1': 'Case1',
        'case2': 'Case2', 
        'case3': 'Case3',
        'case4': 'Case4',
        'mip': 'MIP',
        'rule_based': 'RuleBased'
    }
    
    # Define policy order (RL policies first, then baselines)
    policy_order = ['case1', 'case2', 'case3', 'case4', 'mip', 'rule_based']
    
    # Calculate statistics grouped by simulator type and policy
    stats = {}
    for sim_type in simulator_order:
        if sim_type not in stats:
            stats[sim_type] = {}
            
        sim_data = df[df['config'].str.contains(sim_type)]
        if sim_data.empty:
            continue
            
        for policy in policy_order:
            policy_data = sim_data[sim_data['policy'] == policy]
            if not policy_data.empty:
                stats[sim_type][policy] = {
                    'net_per_agent': (policy_data['net_per_agent'].mean(), policy_data['net_per_agent'].std()),
                    'mission_percentage': (policy_data['mission_percentage'].mean(), policy_data['mission_percentage'].std()),
                    'average_resources': (policy_data['average_resources'].mean(), policy_data['average_resources'].std()),
                    'step_count': (policy_data['step_count'].mean(), policy_data['step_count'].std())
                }
    
    # Generate LaTeX table
    latex_content = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performance Comparison Across Different Configurations}",
        "\\label{tab:performance_comparison}",
        "\\medskip",
        "",
        "\\begin{tabular}{lcccc}",
        "\\textbf{Method} & \\textbf{NET per Action ($\\mu$s)} & \\textbf{Mission (\\%)} & \\textbf{Avg Resources} & \\textbf{Sim Steps} \\\\",
        "\\hline"
    ]
    
    # Add data for each simulator type
    for i, sim_type in enumerate(simulator_order):
        if sim_type not in stats or not stats[sim_type]:
            continue
            
        # Add simulator section header
        latex_content.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{simulator_mapping[sim_type]}}}}} \\\\")
        
        # Add policies for this simulator type
        for policy in policy_order:
            if policy in stats[sim_type]:
                data = stats[sim_type][policy]
                policy_name = policy_mapping[policy]
                
                # Format values with mean±std
                net_mean, net_std = data['net_per_agent']
                mission_mean, mission_std = data['mission_percentage'] 
                resources_mean, resources_std = data['average_resources']
                steps_mean, steps_std = data['step_count']
                
                row = f"{policy_name} & {net_mean:.3f}$\\pm${net_std:.3f} & {mission_mean:.1f}$\\pm${mission_std:.1f} & {resources_mean:.2f}$\\pm${resources_std:.2f} & {steps_mean:.0f}$\\pm${steps_std:.0f} \\\\"
                latex_content.append(row)
        
        # Add horizontal line after each section (except the last)
        if i < len(simulator_order) - 1:
            latex_content.append("\\hline")
    
    # Close the table
    latex_content.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Simulator comparison table saved to {output_file}")


def create_visualizations(df, output_dir):
    """Create visualizations for the benchmark results"""
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Performance comparison across policies and configurations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # NET per agent (now in microseconds)
    sns.boxplot(data=df, x='config', y='net_per_agent', hue='policy', ax=ax1)
    ax1.set_title('NET per Agent by Configuration and Policy (μs)')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('NET per Agent (μs)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Mission percentage (set y-axis from 0 to 100 for better visualization)
    sns.boxplot(data=df, x='config', y='mission_percentage', hue='policy', ax=ax2)
    ax2.set_title('Mission Completion % by Configuration and Policy')
    ax2.set_xlabel('Configuration')
    ax2.set_ylim(0, 100)  # Better visualization
    ax2.tick_params(axis='x', rotation=45)
    
    # Average resources (set y-axis from 0 to 1 for better visualization)
    sns.boxplot(data=df, x='config', y='average_resources', hue='policy', ax=ax3)
    ax3.set_title('Average Resources Left by Configuration and Policy')
    ax3.set_xlabel('Configuration')
    ax3.set_ylim(0, 1)  # Better visualization
    ax3.tick_params(axis='x', rotation=45)
    
    # Simulation steps (check if column exists)
    if 'step_count' in df.columns:
        sns.boxplot(data=df, x='config', y='step_count', hue='policy', ax=ax4)
        ax4.set_title('Simulation Steps Completed by Configuration and Policy')
        ax4.set_xlabel('Configuration')
        ax4.tick_params(axis='x', rotation=45)
    else:
        print("⚠️  step_count not found")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Action distribution visualization (stacked bar charts by configuration)
    configs = df['config'].unique()
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(6*n_configs, 6))
    if n_configs == 1:
        axes = [axes]  # Make it iterable for single config
    
    action_cols = ['action_0_percent', 'action_1_percent', 'action_2_percent']
    action_names = ['Idle', 'Communicate', 'Observe']
    colors = ['lightblue', 'orange', 'lightgreen']
    
    for i, config in enumerate(configs):
        config_data = df[df['config'] == config]
        policies = config_data['policy'].unique()
        
        # Calculate mean percentages for each policy
        policy_means = {}
        for policy in policies:
            policy_data = config_data[config_data['policy'] == policy]
            policy_means[policy] = [
                policy_data['action_0_percent'].mean(),
                policy_data['action_1_percent'].mean(), 
                policy_data['action_2_percent'].mean()
            ]
        
        # Create stacked bar chart
        bottom = np.zeros(len(policies))
        
        for j, (action_name, color) in enumerate(zip(action_names, colors)):
            values = [policy_means[policy][j] for policy in policies]
            bars = axes[i].bar(policies, values, bottom=bottom, label=action_name, color=color)
            bottom += values
        
        axes[i].set_title(f'Action Distribution - {config}')
        axes[i].set_xlabel('Policy')
        axes[i].set_ylabel('Percentage')
        axes[i].set_ylim(0, 100)
        
        # Rotate x-axis labels if needed
        if len(max(policies, key=len)) > 8:  # If policy names are long
            axes[i].tick_params(axis='x', rotation=45)
    
    # Add a single legend outside the plot area
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the legend
    plt.savefig(output_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {output_dir}")


def print_action_distribution_summary(df):
    """Print a summary of action distributions to console"""
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION SUMMARY")
    print("="*80)
    print("Action 0: Idle")
    print("Action 1: Communicate") 
    print("Action 2: Observe")
    print("="*80)
    
    for config in df['config'].unique():
        print(f"\nConfiguration: {config}")
        print("-" * 50)
        
        config_data = df[df['config'] == config]
        action_summary = config_data.groupby('policy')[['action_0_percent', 'action_1_percent', 'action_2_percent']].agg(['mean', 'std']).round(2)
        
        for policy in config_data['policy'].unique():
            policy_data = config_data[config_data['policy'] == policy]
            idle_mean = policy_data['action_0_percent'].mean()
            idle_std = policy_data['action_0_percent'].std()
            comm_mean = policy_data['action_1_percent'].mean()
            comm_std = policy_data['action_1_percent'].std()
            obs_mean = policy_data['action_2_percent'].mean()
            obs_std = policy_data['action_2_percent'].std()
            
            print(f"{policy:15} | Idle: {idle_mean:5.1f}±{idle_std:4.1f}% | Comm: {comm_mean:5.1f}±{comm_std:4.1f}% | Obs: {obs_mean:5.1f}±{obs_std:4.1f}%")


def print_simulator_type_summary(df):
    """Print a summary of results by simulator type"""
    print("\n" + "="*80)
    print("SIMULATOR TYPE COMPARISON")
    print("="*80)
    print("Centralized: One relay satellite for all communications")
    print("Decentralized: Satellites communicate based on band compatibility") 
    print("Everyone: All satellites can communicate with each other")
    print("="*80)
    
    for sim_type in df['simulator_type'].unique():
        print(f"\nSimulator Type: {sim_type.upper()}")
        print("-" * 50)
        
        sim_data = df[df['simulator_type'] == sim_type]
        
        # Group by policy and calculate means
        policy_stats = sim_data.groupby('policy')[['net_per_agent', 'mission_percentage', 'average_resources', 'simulation_time']].agg(['mean', 'std']).round(3)
        
        for policy in sim_data['policy'].unique():
            policy_data = sim_data[sim_data['policy'] == policy]
            net_mean = policy_data['net_per_agent'].mean()
            net_std = policy_data['net_per_agent'].std()
            mission_mean = policy_data['mission_percentage'].mean()
            mission_std = policy_data['mission_percentage'].std()
            
            print(f"{policy:15} | NET: {net_mean:6.3f}±{net_std:5.3f} | Mission: {mission_mean:5.1f}±{mission_std:4.1f}%")


def print_performance_summary(df):
    """Print a summary of key performance metrics"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print("NET: Normalized Execution Time per Agent (microseconds)")
    print("Mission: Mission completion percentage")
    print("="*80)
    
    for config in df['config'].unique():
        # Extract simulator type from config name
        if 'centralized' in config.lower():
            sim_type = 'CENTRALIZED'
        elif 'decentralized' in config.lower():
            sim_type = 'DECENTRALIZED'
        elif 'everyone' in config.lower():
            sim_type = 'EVERYONE'
        else:
            sim_type = config.upper()
            
        print(f"\nSimulator Type: {sim_type}")
        print("-" * 50)
        
        config_data = df[df['config'] == config]
        
        for policy in config_data['policy'].unique():
            policy_data = config_data[config_data['policy'] == policy]
            net_mean = policy_data['net_per_agent'].mean()
            net_std = policy_data['net_per_agent'].std()
            mission_mean = policy_data['mission_percentage'].mean()
            mission_std = policy_data['mission_percentage'].std()
            
            print(f"{policy:15} | NET: {net_mean:6.3f}±{net_std:5.3f}μs | Mission: {mission_mean:5.1f}±{mission_std:4.1f}%")


def process_single_result(result):
    """Process a single benchmark result into a flat dictionary"""
    
    # Extract basic info
    config = result.get('config', 'unknown')
    policy = result.get('policy', 'unknown')
    
    # Extract metrics with proper error handling
    metrics = result.get('metrics', {})
    
    # Scale NET to microseconds for readability
    net_per_agent = metrics.get('net_per_agent', 0) * 1e6  # Convert to microseconds
    
    processed = {
        'config': config,
        'policy': policy,
        'episode_reward': result.get('episode_reward', 0),
        'net_per_agent': net_per_agent,  # Now in microseconds
        'mission_percentage': metrics.get('mission_percentage', 0),
        'average_resources': metrics.get('average_resources_left', 0),
        'simulation_time': metrics.get('simulation_time', 0),
        'step_count': result.get('step_count', 0),
    }
    
    # Add action distribution
    action_dist = metrics.get('action_distribution', {})
    processed.update({
        'action_0_percent': action_dist.get('action_0', 0),
        'action_1_percent': action_dist.get('action_1', 0), 
        'action_2_percent': action_dist.get('action_2', 0),
    })
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default=None, help="Output directory for analysis (default: same folder as results)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_benchmark_results(args.results_file)
    
    # Create DataFrame
    df = create_summary_dataframe(results)
    print(f"Loaded {len(df)} result entries")
    
    # Print simulator type info
    print(f"Simulator types found: {df['simulator_type'].unique()}")
    
    # Determine output directory
    if args.output_dir is None:
        # Use the same directory as the results file
        output_dir = Path(args.results_file).parent / "analysis"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Generate summary statistics
    print(f"Generating analysis in {output_dir}...")
    summary_stats = df.groupby(['config', 'policy', 'simulator_type']).agg({
        'net_per_agent': ['mean', 'std', 'min', 'max'],
        'mission_percentage': ['mean', 'std', 'min', 'max'],
        'average_resources': ['mean', 'std', 'min', 'max'],
        'simulation_time': ['mean', 'std', 'min', 'max'],
        'total_reward': ['mean', 'std', 'min', 'max'],
        'action_0_percent': ['mean', 'std'],
        'action_1_percent': ['mean', 'std'],
        'action_2_percent': ['mean', 'std'],
    }).round(4)
    
    summary_stats.to_csv(output_dir / 'summary_statistics.csv')
    print(f"✓ Summary statistics saved to {output_dir / 'summary_statistics.csv'}")
    
    # Print summaries to console
    print_action_distribution_summary(df)
    print_simulator_type_summary(df)
    print_performance_summary(df)
    
    # Generate LaTeX tables
    generate_latex_table(df, output_dir / 'performance_table.tex')
    generate_action_distribution_table(df, output_dir / 'action_distribution_table.tex')
    generate_simulator_comparison_table(df, output_dir / 'simulator_comparison_table.tex')
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Save processed DataFrame
    df.to_csv(output_dir / 'processed_results.csv', index=False)
    print(f"✓ Processed results saved to {output_dir / 'processed_results.csv'}")
    
    # Create a README for the experiment
    readme_content = f"""# Experiment Analysis

## Experiment Details
- Results file: {Path(args.results_file).name}
- Analysis generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total result entries: {len(df)}
- Simulator types: {', '.join(df['simulator_type'].unique())}

## Files in this directory:
- `benchmark_results.json`: Raw benchmark results
- `experiment_metadata.json`: Experiment configuration and metadata
- `analysis/`: Analysis outputs
  - `summary_statistics.csv`: Detailed statistics by configuration, policy, and simulator type
  - `processed_results.csv`: Processed data in tabular format
  - `performance_table.tex`: LaTeX table for performance comparison
  - `action_distribution_table.tex`: LaTeX table for action distributions
  - `simulator_comparison_table.tex`: LaTeX table comparing simulator types
  - `performance_comparison.png`: Performance visualization
  - `simulator_type_comparison.png`: Simulator type comparison visualization
  - `action_distribution.png`: Action distribution visualization
  - `scaling_analysis.png`: Scaling analysis (if multiple configurations)

## Key Metrics:
- **NET per agent**: Network-based efficiency metric
- **Mission completion**: Percentage of targets successfully observed
- **Average resources**: Remaining computational resources
- **Simulation time**: Wall-clock time per episode

## Simulator Types:
- **Centralized**: One relay satellite handles all inter-satellite communications
- **Decentralized**: Satellites communicate based on compatible communication bands
- **Everyone**: All satellites can communicate directly with each other

## Policy Types:
- **RuleBased**: Heuristic baseline policy
- **MIP**: Mixed Integer Programming optimization baseline
- **Case1-4**: Reinforcement Learning policies trained on different reward functions
"""
    
    with open(output_dir.parent / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"\n✓ Analysis complete!")
    print(f"📁 Experiment folder: {output_dir.parent}")
    print(f"📊 Analysis saved in: {output_dir}")
    print(f"📖 README created: {output_dir.parent / 'README.md'}")


if __name__ == "__main__":
    main() 