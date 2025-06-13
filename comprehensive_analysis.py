#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Journal Paper
Combines baseline methods (Rule-based, MIP) with RL sensitivity analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

# Configure matplotlib for journal paper quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_experiment_metadata(results_file_path=None):
    """Load experiment metadata for CPU specifications"""
    metadata_files = [
        'experiment_metadata.json',
        '../experiment_metadata.json',
        '../../experiment_metadata.json'
    ]
    
    if results_file_path:
        results_dir = Path(results_file_path).parent
        metadata_files.insert(0, results_dir / 'experiment_metadata.json')
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if 'system_info' in metadata:
                    return metadata['system_info']
        except FileNotFoundError:
            continue
    
    # Fallback to defaults
    print("⚠ Using default CPU specifications (metadata not found)")
    return {
        'cpu_cores': 96,
        'cpu_freq': 2.17
    }

def create_unified_dataframe(all_results, results_file_path=None):
    """Create unified DataFrame with all methods and configurations"""
    
    # Load CPU specifications
    system_info = load_experiment_metadata(results_file_path)
    cpu_count = system_info.get('cpu_cores', 96)
    cpu_freq_ghz = system_info.get('cpu_freq', 2.17)
    
    print(f"✓ Using CPU specifications: {cpu_count} cores at {cpu_freq_ghz:.2f} GHz")
    
    unified_data = []
    
    for episode in all_results:
        metrics = episode['metrics']
        timing = episode['timing']
        
        # Simple execution time calculation using average_action_time
        # This is already the average time per step for all agents
        average_action_time_seconds = timing.get('average_action_time', 0)
        num_agents = episode.get('num_agents', 20)
        
        # For baseline policies: divide by number of agents to get per-agent time
        # For RL policies: this is already per-agent time, but divide anyway for consistency
        execution_time_per_agent_seconds = average_action_time_seconds / num_agents
        execution_time_us = execution_time_per_agent_seconds * 1_000_000
        
        # Determine method and scaling factor
        if 'policy_name' in episode:
            policy_name = episode['policy_name']
            if policy_name in ['RuleBased', 'MIP']:
                method_name = policy_name
                scaling_factor = 'baseline'
            elif policy_name.startswith(('case1_', 'case2_', 'case3_', 'case4_')):
                parts = policy_name.split('_')
                method_name = parts[0]
                scaling_factor = '_'.join(parts[1:])
            else:
                method_name = policy_name
                scaling_factor = 'baseline'
        else:
            # Handle different data formats (RL sensitivity analysis format)
            method_name = episode.get('case_type', 'unknown')
            scaling_factor = episode.get('bonus_value', 'baseline')
        
        # Extract metrics
        mission_percentage = metrics.get('mission_percentage', 0)
        avg_resources_left = metrics.get('average_resources_left', 1.0)
        resource_utilization = (1.0 - avg_resources_left) * 100
        
        # Extract action distribution
        action_dist = metrics.get('action_distribution', {})
        action_0_percent = action_dist.get('action_0', 0)
        action_1_percent = action_dist.get('action_1', 0)
        action_2_percent = action_dist.get('action_2', 0)
        
        # Get simulator type
        env_metrics = episode.get('raw_data', {}).get('env_metrics', {})
        simulator_type = env_metrics.get('simulator_type', episode.get('simulator_type', 'unknown'))
        
        unified_data.append({
            'method': method_name,
            'scaling_factor': scaling_factor,
            'simulator_type': simulator_type,
            'execution_time_us': execution_time_us,
            'mission_percentage': mission_percentage,
            'resource_utilization': resource_utilization,
            'action_0_percent': action_0_percent,  # Idle
            'action_1_percent': action_1_percent,  # Communicate
            'action_2_percent': action_2_percent,  # Observe
        })
    
    return pd.DataFrame(unified_data)

def generate_comprehensive_latex_table(df, output_file):
    """Generate comprehensive LaTeX table with all methods"""
    
    # Method name mapping
    method_mapping = {
        'RuleBased': 'Rule-Based',
        'MIP': 'MIP',
        'case1': 'Individual Positive',
        'case2': 'Individual Negative',
        'case3': 'Collective Positive',
        'case4': 'Collective Negative'
    }
    
    # Simulator name mapping
    simulator_mapping = {
        'centralized': 'Centralized coordination',
        'decentralized': 'Constrained decentralized coordination',
        'everyone': 'Fully decentralized coordination'
    }
    
    # Group data
    grouped = df.groupby(['method', 'scaling_factor', 'simulator_type']).agg({
        'execution_time_us': ['mean', 'std'],
        'mission_percentage': ['mean', 'std'],
        'resource_utilization': ['mean', 'std']
    }).round(3)
    
    # Start LaTeX table
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Comprehensive Performance Comparison Across All Methods, Scaling Factors, and Coordination Types}",
        "\\label{tab:comprehensive_analysis}",
        "\\medskip",
        "",
        "\\begin{tabular}{lcccc}",
        "\\textbf{Method} & \\textbf{$\\alpha$ [-]} & \\textbf{Action Exec. Time per Agent [$\\mu$s]} & \\textbf{Mission [\\%]} & \\textbf{Remaining Resources [\\%]} \\\\",
        "\\hline"
    ]
    
    # Order methods and scaling factors
    method_order = ['RuleBased', 'MIP', 'case1', 'case2', 'case3', 'case4']
    simulator_order = ['centralized', 'decentralized', 'everyone']
    scaling_order = ['baseline', 'bonus0', 'bonus01', 'bonus05', 'bonus10']
    
    # Organize by coordination type
    for sim_type in simulator_order:
        if sim_type not in df['simulator_type'].values:
            continue
            
        # Add coordination type header
        sim_display = simulator_mapping[sim_type]
        latex_lines.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{sim_display}}}}} \\\\")
        
        # Add baseline methods first (if they exist for this simulator type)
        for method in ['RuleBased', 'MIP']:
            if method not in df['method'].values:
                continue
                
            if (method, 'baseline', sim_type) in grouped.index:
                row = grouped.loc[(method, 'baseline', sim_type)]
                method_display = method_mapping[method]
                
                # Extract values
                exec_mean = row[('execution_time_us', 'mean')]
                exec_std = row[('execution_time_us', 'std')]
                mission_mean = row[('mission_percentage', 'mean')]
                mission_std = row[('mission_percentage', 'std')]
                resource_mean = row[('resource_utilization', 'mean')]
                resource_std = row[('resource_utilization', 'std')]
                
                # Format table row
                latex_lines.append(
                    f"{method_display} & - & "
                    f"{exec_mean:.0f}$\\pm${exec_std:.0f} & "
                    f"{mission_mean:.1f}$\\pm${mission_std:.1f} & "
                    f"{resource_mean:.1f}$\\pm${resource_std:.1f} \\\\"
                )
        
        # Add RL methods with all their scaling factors
        for method in ['case1', 'case2', 'case3', 'case4']:
            if method not in df['method'].values:
                continue
                
            method_display = method_mapping[method]
            method_has_data = False
            
            # Check all scaling factors for this method and simulator type
            for scaling in ['bonus0', 'bonus01', 'bonus05', 'bonus10']:
                if (method, scaling, sim_type) in grouped.index:
                    row = grouped.loc[(method, scaling, sim_type)]
                    
                    # Fix scaling factor display mapping
                    scaling_display = scaling.replace('bonus0', '0.0').replace('bonus01', '0.01').replace('bonus05', '0.05').replace('bonus10', '1.0')
                    
                    # Extract values
                    exec_mean = row[('execution_time_us', 'mean')]
                    exec_std = row[('execution_time_us', 'std')]
                    mission_mean = row[('mission_percentage', 'mean')]
                    mission_std = row[('mission_percentage', 'std')]
                    resource_mean = row[('resource_utilization', 'mean')]
                    resource_std = row[('resource_utilization', 'std')]
                    
                    # Format table row
                    latex_lines.append(
                        f"{method_display} & {scaling_display} & "
                        f"{exec_mean:.0f}$\\pm${exec_std:.0f} & "
                        f"{mission_mean:.1f}$\\pm${mission_std:.1f} & "
                        f"{resource_mean:.1f}$\\pm${resource_std:.1f} \\\\"
                    )
                    method_has_data = True
        
        # Add separator after each coordination type (except the last one)
        if sim_type != simulator_order[-1]:
            latex_lines.append("\\hline")
    
    # Close table
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"✓ Comprehensive LaTeX table saved to {output_file}")

def generate_action_distribution_plot(df, output_dir):
    """Generate professional action distribution plot matching mission/resource plot structure"""
    
    # Method name mapping with acronyms for legend and labels
    method_mapping = {
        'RuleBased': 'R-B',
        'MIP': 'M',
        'case1': 'IP',  # Individual Positive
        'case2': 'IN',  # Individual Negative
        'case3': 'CP',  # Collective Positive
        'case4': 'CN'   # Collective Negative
    }
    
    # Full method names for legend
    method_full_names = {
        'RuleBased': 'Rule-Based (R-B)',
        'MIP': 'MIP (M)',
        'case1': 'Individual Positive (IP)',
        'case2': 'Individual Negative (IN)',
        'case3': 'Collective Positive (CP)',
        'case4': 'Collective Negative (CN)'
    }
    
    simulator_mapping = {
        'centralized': 'Centralized',
        'decentralized': 'Constrained Decentralized',
        'everyone': 'Fully Decentralized'
    }
    
    # Professional color palette (colorblind-friendly) - same as other plots
    method_colors = {
        'RuleBased': '#D62728',  # Red
        'MIP': '#FF7F0E',        # Orange
        'case1': '#1F77B4',      # Blue
        'case2': '#2CA02C',      # Green
        'case3': '#9467BD',      # Purple
        'case4': '#8C564B'       # Brown
    }
    
    # Action type colors for stacked bars
    action_colors = {
        'idle': '#E6E6E6',      # Light gray
        'observe': '#B3B3B3',   # Medium gray
        'communicate': '#808080' # Dark gray
    }
    
    # Create subplots with better spacing
    simulator_order = ['centralized', 'decentralized', 'everyone']
    available_simulators = [s for s in simulator_order if s in df['simulator_type'].unique()]
    
    # Increased figure size and spacing
    fig, axes = plt.subplots(len(available_simulators), 1, figsize=(14, 5*len(available_simulators)))
    if len(available_simulators) == 1:
        axes = [axes]
    
    # Collect legend information for methods
    method_legend_elements = []
    method_legend_added = set()
    
    # Action type legend elements
    action_legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=action_colors['idle'], label='Idle'),
        plt.Rectangle((0,0),1,1, facecolor=action_colors['observe'], label='Observe'),
        plt.Rectangle((0,0),1,1, facecolor=action_colors['communicate'], label='Communicate')
    ]
    
    for i, sim_type in enumerate(available_simulators):
        sim_data = df[df['simulator_type'] == sim_type]
        
        # Prepare data organized by alpha values (same as other plots)
        alpha_groups = {}  # alpha_value -> [(action_data, method, color)]
        
        method_order = ['RuleBased', 'MIP', 'case1', 'case2', 'case3', 'case4']
        
        for method in method_order:
            method_data = sim_data[sim_data['method'] == method]
            if method_data.empty:
                continue
            
            if method in ['RuleBased', 'MIP']:
                # Baseline methods - add to alpha = "Baseline"
                baseline_data = method_data[method_data['scaling_factor'] == 'baseline']
                if not baseline_data.empty:
                    alpha_key = 'Baseline'
                    if alpha_key not in alpha_groups:
                        alpha_groups[alpha_key] = []
                    
                    # Calculate action percentages
                    action_data = {
                        'idle': baseline_data['action_0_percent'].mean(),
                        'communicate': baseline_data['action_1_percent'].mean(),
                        'observe': baseline_data['action_2_percent'].mean()
                    }
                    
                    alpha_groups[alpha_key].append((
                        action_data,
                        method_mapping[method],
                        method_colors[method]
                    ))
                    
                    # Add to legend if not already added
                    if method not in method_legend_added:
                        method_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=method_colors[method], 
                                                           label=method_full_names[method]))
                        method_legend_added.add(method)
            else:
                # RL methods - group by scaling factor (alpha value)
                scaling_order = ['bonus0', 'bonus01', 'bonus05', 'bonus10']
                scaling_labels = ['0.0', '0.1', '0.5', '1.0']  # Corrected alpha values
                
                for scaling, scaling_label in zip(scaling_order, scaling_labels):
                    scaling_data = method_data[method_data['scaling_factor'] == scaling]
                    if not scaling_data.empty:
                        alpha_key = f'α = {scaling_label}'
                        if alpha_key not in alpha_groups:
                            alpha_groups[alpha_key] = []
                        
                        # Calculate action percentages
                        action_data = {
                            'idle': scaling_data['action_0_percent'].mean(),
                            'communicate': scaling_data['action_1_percent'].mean(),
                            'observe': scaling_data['action_2_percent'].mean()
                        }
                        
                        alpha_groups[alpha_key].append((
                            action_data,
                            method_mapping[method],
                            method_colors[method]
                        ))
                        
                        # Add to legend if not already added (only once per method type)
                        if method not in method_legend_added:
                            method_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=method_colors[method], 
                                                               label=method_full_names[method]))
                            method_legend_added.add(method)
        
        # Create organized stacked bar data (same grouping as other plots)
        # Sort alpha groups: Baseline first, then numeric alpha values
        sorted_alpha_keys = []
        if 'Baseline' in alpha_groups:
            sorted_alpha_keys.append('Baseline')
        
        # Add alpha values in order
        for alpha_val in ['α = 0.0', 'α = 0.1', 'α = 0.5', 'α = 1.0']:
            if alpha_val in alpha_groups:
                sorted_alpha_keys.append(alpha_val)
        
        # Prepare grouped stacked bar data - ONE BAR PER METHOD-ALPHA COMBINATION
        if sorted_alpha_keys:
            all_action_data = []
            all_colors = []
            method_labels = []
            alpha_labels = []
            
            # Create bars in method order, with all alphas for each method
            for method in method_order:
                if method in ['RuleBased', 'MIP']:
                    # Baseline methods - just one bar
                    if 'Baseline' in alpha_groups:
                        for action_data, method_name, color in alpha_groups['Baseline']:
                            if method_name == method_mapping[method]:
                                all_action_data.append(action_data)
                                all_colors.append(method_colors[method])
                                method_labels.append(method_mapping[method])
                                alpha_labels.append('-')
                                break
                else:
                    # RL methods - one bar per alpha value
                    for alpha_key in sorted_alpha_keys:
                        if alpha_key in alpha_groups:
                            for action_data, method_name, color in alpha_groups[alpha_key]:
                                if method_name == method_mapping[method]:
                                    all_action_data.append(action_data)
                                    all_colors.append(method_colors[method])
                                    method_labels.append(method_mapping[method])
                                    alpha_labels.append(alpha_key.replace('α = ', ''))
                                    break
            
            # Create professional stacked bar plot
            if all_action_data:
                n_bars = len(all_action_data)
                x_pos = list(range(n_bars))
                
                # Create stacked bars
                bar_width = 0.8
                
                # Stack the bars
                idle_values = [data['idle'] for data in all_action_data]
                observe_values = [data['observe'] for data in all_action_data]
                communicate_values = [data['communicate'] for data in all_action_data]
                
                # Create stacked bars
                bars1 = axes[i].bar(x_pos, idle_values, bar_width, 
                                  color=action_colors['idle'], label='Idle', 
                                  edgecolor='black', linewidth=0.5)
                
                bars2 = axes[i].bar(x_pos, observe_values, bar_width, 
                                  bottom=idle_values, color=action_colors['observe'], 
                                  label='Observe', edgecolor='black', linewidth=0.5)
                
                bottom_comm = [w + o for w, o in zip(idle_values, observe_values)]
                bars3 = axes[i].bar(x_pos, communicate_values, bar_width, 
                                  bottom=bottom_comm, color=action_colors['communicate'], 
                                  label='Communicate', edgecolor='black', linewidth=0.5)
                
                # Add percentage labels for segments that are large enough
                for j, (idle_val, observe_val, comm_val) in enumerate(zip(idle_values, observe_values, communicate_values)):
                    # Label idle segment if > 8%
                    if idle_val > 8:
                        axes[i].text(j, idle_val/2, f'{idle_val:.1f}%',
                                   ha='center', va='center', fontsize=10, 
                                   fontweight='bold', color='black')
                    
                    # Label observe segment if > 8%
                    if observe_val > 8:
                        axes[i].text(j, idle_val + observe_val/2, f'{observe_val:.1f}%',
                                   ha='center', va='center', fontsize=10, 
                                   fontweight='bold', color='white')
                    
                    # Label communicate segment if > 8%
                    if comm_val > 8:
                        axes[i].text(j, idle_val + observe_val + comm_val/2, f'{comm_val:.1f}%',
                                   ha='center', va='center', fontsize=10, 
                                   fontweight='bold', color='white')
                
                # Add method color borders to bars (subtle indication of method type)
                for j, (bar1, bar2, bar3, method_color) in enumerate(zip(bars1, bars2, bars3, all_colors)):
                    # Add a subtle colored border around the entire bar stack
                    for bar in [bar1, bar2, bar3]:
                        bar.set_edgecolor(method_color)
                        bar.set_linewidth(2)
                
                # Set x-tick labels to show alpha values
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(alpha_labels, fontsize=18)
                
                # Add visual grouping rectangles and method labels
                for method in method_order:
                    method_bars = []
                    for i_bar, (method_label, alpha_label) in enumerate(zip(method_labels, alpha_labels)):
                        if method_label == method_mapping.get(method, method):
                            method_bars.append(i_bar)
                    
                    if method_bars:
                        # Add background rectangle for method group
                        start_pos = method_bars[0] - 0.4
                        end_pos = method_bars[-1] + 0.4
                        width = end_pos - start_pos
                        
                        rect = plt.Rectangle((start_pos, -5), width, 110, 
                                           facecolor='lightgray', alpha=0.3, 
                                           edgecolor='black', linewidth=2, zorder=0)
                        axes[i].add_patch(rect)
                        
                        # Add method name above the group
                        group_center = (method_bars[0] + method_bars[-1]) / 2
                        axes[i].text(group_center, 100, method_mapping.get(method, method), 
                                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Professional styling for axes
        axes[i].set_title(f'{simulator_mapping[sim_type]} Coordination', 
                         fontsize=24, pad=30)
            
        axes[i].set_ylim(0, 105)
        axes[i].tick_params(axis='y', labelsize=18)  # Increase y-axis font size
        axes[i].grid(True, alpha=0.3, axis='y', linestyle='--')
        
    # Set shared x-label and y-label
    fig.text(0.5, 0.06, 'α [-]', ha='center', fontsize=28)
    fig.text(0.03, 0.5, 'Action Distribution [%]', va='center', rotation='vertical', fontsize=28)

    # Create combined legend with both methods and actions
    # First row: Methods, Second row: Actions
    all_legend_elements = method_legend_elements + action_legend_elements
    
    # Add comprehensive legend at bottom center with two rows
    fig.legend(handles=all_legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
              ncol=4, fontsize=16, frameon=True, fancybox=True, shadow=True,
              title='Methods and Actions', title_fontsize=24)
    
    # Better spacing and layout - more space for coordination titles and legend
    plt.subplots_adjust(top=0.90, hspace=0.4, bottom=0.15)  # Reduced bottom space
    
    # Save high-quality PNG
    plt.savefig(output_dir / 'action_distribution_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"✓ Action distribution plot saved to {output_dir}")

def generate_mission_accomplishment_boxplot(df, output_dir):
    """Generate professional box and whiskers plot for mission accomplishment"""
    
    # Method name mapping with acronyms for legend and labels
    method_mapping = {
        'RuleBased': 'R-B',
        'MIP': 'M',
        'case1': 'IP',  # Individual Positive
        'case2': 'IN',  # Individual Negative
        'case3': 'CP',  # Collective Positive
        'case4': 'CN'   # Collective Negative
    }
    
    # Full method names for legend
    method_full_names = {
        'RuleBased': 'Rule-Based (R-B)',
        'MIP': 'MIP (M)',
        'case1': 'Individual Positive (IP)',
        'case2': 'Individual Negative (IN)',
        'case3': 'Collective Positive (CP)',
        'case4': 'Collective Negative (CN)'
    }
    
    simulator_mapping = {
        'centralized': 'Centralized',
        'decentralized': 'Constrained Decentralized',
        'everyone': 'Fully Decentralized'
    }
    
    # Professional color palette (colorblind-friendly)
    method_colors = {
        'RuleBased': '#D62728',  # Red
        'MIP': '#FF7F0E',        # Orange
        'case1': '#1F77B4',      # Blue
        'case2': '#2CA02C',      # Green
        'case3': '#9467BD',      # Purple
        'case4': '#8C564B'       # Brown
    }
    
    # Create subplots with better spacing
    simulator_order = ['centralized', 'decentralized', 'everyone']
    available_simulators = [s for s in simulator_order if s in df['simulator_type'].unique()]
    
    # Increased figure size and spacing
    fig, axes = plt.subplots(len(available_simulators), 1, figsize=(14, 5*len(available_simulators)))
    if len(available_simulators) == 1:
        axes = [axes]
    
    # Collect legend information
    legend_elements = []
    legend_added = set()
    
    for i, sim_type in enumerate(available_simulators):
        sim_data = df[df['simulator_type'] == sim_type]
        
        # Prepare data organized by alpha values (same as action distribution)
        alpha_groups = {}  # alpha_value -> [(data, method, color)]
        
        method_order = ['RuleBased', 'MIP', 'case1', 'case2', 'case3', 'case4']
        
        for method in method_order:
            method_data = sim_data[sim_data['method'] == method]
            if method_data.empty:
                continue
            
            if method in ['RuleBased', 'MIP']:
                # Baseline methods - add to alpha = "Baseline"
                baseline_data = method_data[method_data['scaling_factor'] == 'baseline']
                if not baseline_data.empty:
                    alpha_key = 'Baseline'
                    if alpha_key not in alpha_groups:
                        alpha_groups[alpha_key] = []
                    alpha_groups[alpha_key].append((
                        baseline_data['mission_percentage'].values,
                        method_mapping[method],
                        method_colors[method]
                    ))
                    
                    # Add to legend if not already added
                    if method not in legend_added:
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=method_colors[method], 
                                                           label=method_full_names[method]))
                        legend_added.add(method)
            else:
                # RL methods - group by scaling factor (alpha value)
                scaling_order = ['bonus0', 'bonus01', 'bonus05', 'bonus10']
                scaling_labels = ['0.0', '0.1', '0.5', '1.0']  # Corrected alpha values
                
                for scaling, scaling_label in zip(scaling_order, scaling_labels):
                    scaling_data = method_data[method_data['scaling_factor'] == scaling]
                    if not scaling_data.empty:
                        alpha_key = f'α = {scaling_label}'
                        if alpha_key not in alpha_groups:
                            alpha_groups[alpha_key] = []
                        alpha_groups[alpha_key].append((
                            scaling_data['mission_percentage'].values,
                            method_mapping[method],
                            method_colors[method]
                        ))
                        
                        # Add to legend if not already added (only once per method type)
                        if method not in legend_added:
                            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=method_colors[method], 
                                                               label=method_full_names[method]))
                            legend_added.add(method)
        
        # Create organized box plot data (same grouping as action distribution)
        # Sort alpha groups: Baseline first, then numeric alpha values
        sorted_alpha_keys = []
        if 'Baseline' in alpha_groups:
            sorted_alpha_keys.append('Baseline')
        
        # Add alpha values in order
        for alpha_val in ['α = 0.0', 'α = 0.1', 'α = 0.5', 'α = 1.0']:
            if alpha_val in alpha_groups:
                sorted_alpha_keys.append(alpha_val)
        
        # Prepare grouped box plot data - ONE BOX PER METHOD-ALPHA COMBINATION
        if sorted_alpha_keys:
            all_box_data = []
            all_colors = []
            method_labels = []
            alpha_labels = []
            
            # Create boxes in method order, with all alphas for each method
            for method in method_order:
                if method in ['RuleBased', 'MIP']:
                    # Baseline methods - just one box
                    if 'Baseline' in alpha_groups:
                        for data, method_name, color in alpha_groups['Baseline']:
                            if method_name == method_mapping[method]:
                                all_box_data.append(data)
                                all_colors.append(method_colors[method])
                                method_labels.append(method_mapping[method])
                                alpha_labels.append('-')
                                break
                else:
                    # RL methods - one box per alpha value
                    for alpha_key in sorted_alpha_keys:
                        if alpha_key in alpha_groups:
                            for data, method_name, color in alpha_groups[alpha_key]:
                                if method_name == method_mapping[method]:
                                    all_box_data.append(data)
                                    all_colors.append(method_colors[method])
                                    method_labels.append(method_mapping[method])
                                    alpha_labels.append(alpha_key.replace('α = ', ''))
                                    break
            
            # Create professional box plot
            if all_box_data:
                n_boxes = len(all_box_data)
                x_pos = list(range(n_boxes))
                
                bp = axes[i].boxplot(all_box_data, positions=x_pos, patch_artist=True,
                                   boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.2),
                                   medianprops=dict(color='black', linewidth=2),
                                   whiskerprops=dict(color='black', linewidth=1.2),
                                   capprops=dict(color='black', linewidth=1.2),
                                   flierprops=dict(marker='o', markerfacecolor='red', 
                                                 markersize=3, alpha=0.6, markeredgecolor='darkred'))
                
                # Color the boxes according to method type
                for patch, color in zip(bp['boxes'], all_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.2)
                
                # Set x-tick labels to show alpha values
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(alpha_labels, fontsize=18)
                
                # Add visual grouping rectangles and method labels
                current_pos = 0
                for method in method_order:
                    method_boxes = []
                    for i_box, (method_label, alpha_label) in enumerate(zip(method_labels, alpha_labels)):
                        if method_label == method_mapping.get(method, method):
                            method_boxes.append(i_box)
                    
                    if method_boxes:
                        # Add background rectangle for method group
                        start_pos = method_boxes[0] - 0.4
                        end_pos = method_boxes[-1] + 0.4
                        width = end_pos - start_pos
                        
                        rect = plt.Rectangle((start_pos, -5), width, 110, 
                                           facecolor='lightgray', alpha=0.3, 
                                           edgecolor='black', linewidth=2, zorder=0)
                        axes[i].add_patch(rect)
                        
                        # Add method name above the group
                        group_center = (method_boxes[0] + method_boxes[-1]) / 2
                        axes[i].text(group_center, 100, method_mapping.get(method, method), 
                                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Professional styling for axes
        axes[i].set_title(f'{simulator_mapping[sim_type]} Coordination', 
                         fontsize=24, pad=30)
            
        # axes[i].set_ylabel('Mission Accomplishment [%]', fontsize=2)
        # axes[i].set_xlabel('α [-]', fontsize=12)
        axes[i].set_ylim(0, 105)
        axes[i].tick_params(axis='y', labelsize=18)  # Increase y-axis font size
        axes[i].grid(True, alpha=0.3, axis='y', linestyle='--')
        # axes[i].spines['top'].set_visible(False)
        # axes[i].spines['right'].set_visible(False)
        
    # Set shared x-label and y-label
    fig.text(0.5, 0.06, 'α [-]', ha='center', fontsize=28)
    fig.text(0.03, 0.5, 'Mission Accomplishment [%]', va='center', rotation='vertical', fontsize=28)

    # Add comprehensive legend at bottom center
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), 
              ncol=3, fontsize=20, frameon=True, fancybox=True, shadow=True,
              title='Methods', title_fontsize=28)
    
    # Better spacing and layout - more space for coordination titles and legend
    plt.subplots_adjust(top=0.90, hspace=0.4)  # More space between subplots for coordination titles
    
    # Save high-quality PNG
    plt.savefig(output_dir / 'mission_accomplishment_boxplot.png', 
                dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"✓ Mission accomplishment box plot saved to {output_dir}")

def generate_resource_utilization_boxplot(df, output_dir):
    """Generate professional box and whiskers plot for resource utilization"""
    
    # Method name mapping with acronyms for legend and labels
    method_mapping = {
        'RuleBased': 'R-B',
        'MIP': 'M',
        'case1': 'IP',  # Individual Positive
        'case2': 'IN',  # Individual Negative
        'case3': 'CP',  # Collective Positive
        'case4': 'CN'   # Collective Negative
    }
    
    # Full method names for legend
    method_full_names = {
        'RuleBased': 'Rule-Based (R-B)',
        'MIP': 'MIP (M)',
        'case1': 'Individual Positive (IP)',
        'case2': 'Individual Negative (IN)',
        'case3': 'Collective Positive (CP)',
        'case4': 'Collective Negative (CN)'
    }
    
    simulator_mapping = {
        'centralized': 'Centralized',
        'decentralized': 'Constrained Decentralized',
        'everyone': 'Fully Decentralized'
    }
    
    # Professional color palette (colorblind-friendly)
    method_colors = {
        'RuleBased': '#D62728',  # Red
        'MIP': '#FF7F0E',        # Orange
        'case1': '#1F77B4',      # Blue
        'case2': '#2CA02C',      # Green
        'case3': '#9467BD',      # Purple
        'case4': '#8C564B'       # Brown
    }
    
    # Create subplots with better spacing
    simulator_order = ['centralized', 'decentralized', 'everyone']
    available_simulators = [s for s in simulator_order if s in df['simulator_type'].unique()]
    
    # Increased figure size and spacing
    fig, axes = plt.subplots(len(available_simulators), 1, figsize=(14, 5*len(available_simulators)))
    if len(available_simulators) == 1:
        axes = [axes]
    
    # Collect legend information
    legend_elements = []
    legend_added = set()
    
    for i, sim_type in enumerate(available_simulators):
        sim_data = df[df['simulator_type'] == sim_type]
        
        # Prepare data organized by alpha values (same as action distribution)
        alpha_groups = {}  # alpha_value -> [(data, method, color)]
        
        method_order = ['RuleBased', 'MIP', 'case1', 'case2', 'case3', 'case4']
        
        for method in method_order:
            method_data = sim_data[sim_data['method'] == method]
            if method_data.empty:
                continue
            
            if method in ['RuleBased', 'MIP']:
                # Baseline methods - add to alpha = "Baseline"
                baseline_data = method_data[method_data['scaling_factor'] == 'baseline']
                if not baseline_data.empty:
                    alpha_key = 'Baseline'
                    if alpha_key not in alpha_groups:
                        alpha_groups[alpha_key] = []
                    alpha_groups[alpha_key].append((
                        baseline_data['resource_utilization'].values,
                        method_mapping[method],
                        method_colors[method]
                    ))
                    
                    # Add to legend if not already added
                    if method not in legend_added:
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=method_colors[method], 
                                                           label=method_full_names[method]))
                        legend_added.add(method)
            else:
                # RL methods - group by scaling factor (alpha value)
                scaling_order = ['bonus0', 'bonus01', 'bonus05', 'bonus10']
                scaling_labels = ['0.0', '0.1', '0.5', '1.0']  # Corrected alpha values
                
                for scaling, scaling_label in zip(scaling_order, scaling_labels):
                    scaling_data = method_data[method_data['scaling_factor'] == scaling]
                    if not scaling_data.empty:
                        alpha_key = f'α = {scaling_label}'
                        if alpha_key not in alpha_groups:
                            alpha_groups[alpha_key] = []
                        alpha_groups[alpha_key].append((
                            scaling_data['resource_utilization'].values,
                            method_mapping[method],
                            method_colors[method]
                        ))
                        
                        # Add to legend if not already added (only once per method type)
                        if method not in legend_added:
                            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=method_colors[method], 
                                                               label=method_full_names[method]))
                            legend_added.add(method)
        
        # Create organized box plot data (same grouping as action distribution)
        # Sort alpha groups: Baseline first, then numeric alpha values
        sorted_alpha_keys = []
        if 'Baseline' in alpha_groups:
            sorted_alpha_keys.append('Baseline')
        
        # Add alpha values in order
        for alpha_val in ['α = 0.0', 'α = 0.1', 'α = 0.5', 'α = 1.0']:
            if alpha_val in alpha_groups:
                sorted_alpha_keys.append(alpha_val)
        
        # Prepare grouped box plot data - ONE BOX PER METHOD-ALPHA COMBINATION
        if sorted_alpha_keys:
            all_box_data = []
            all_colors = []
            method_labels = []
            alpha_labels = []
            
            # Create boxes in method order, with all alphas for each method
            for method in method_order:
                if method in ['RuleBased', 'MIP']:
                    # Baseline methods - just one box
                    if 'Baseline' in alpha_groups:
                        for data, method_name, color in alpha_groups['Baseline']:
                            if method_name == method_mapping[method]:
                                all_box_data.append(data)
                                all_colors.append(method_colors[method])
                                method_labels.append(method_mapping[method])
                                alpha_labels.append('-')
                                break
                else:
                    # RL methods - one box per alpha value
                    for alpha_key in sorted_alpha_keys:
                        if alpha_key in alpha_groups:
                            for data, method_name, color in alpha_groups[alpha_key]:
                                if method_name == method_mapping[method]:
                                    all_box_data.append(data)
                                    all_colors.append(method_colors[method])
                                    method_labels.append(method_mapping[method])
                                    alpha_labels.append(alpha_key.replace('α = ', ''))
                                    break
            
            # Create professional box plot
            if all_box_data:
                n_boxes = len(all_box_data)
                x_pos = list(range(n_boxes))
                
                bp = axes[i].boxplot(all_box_data, positions=x_pos, patch_artist=True,
                                   boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.2),
                                   medianprops=dict(color='black', linewidth=2),
                                   whiskerprops=dict(color='black', linewidth=1.2),
                                   capprops=dict(color='black', linewidth=1.2),
                                   flierprops=dict(marker='o', markerfacecolor='red', 
                                                 markersize=3, alpha=0.6, markeredgecolor='darkred'))
                
                # Color the boxes according to method type
                for patch, color in zip(bp['boxes'], all_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.2)
                
                # Set x-tick labels to show alpha values
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(alpha_labels, fontsize=18)
                
                # Add visual grouping rectangles and method labels
                current_pos = 0
                for method in method_order:
                    method_boxes = []
                    for i_box, (method_label, alpha_label) in enumerate(zip(method_labels, alpha_labels)):
                        if method_label == method_mapping.get(method, method):
                            method_boxes.append(i_box)
                    
                    if method_boxes:
                        # Add background rectangle for method group
                        start_pos = method_boxes[0] - 0.4
                        end_pos = method_boxes[-1] + 0.4
                        width = end_pos - start_pos
                        
                        rect = plt.Rectangle((start_pos, -5), width, 110, 
                                           facecolor='lightgray', alpha=0.3, 
                                           edgecolor='black', linewidth=2, zorder=0)
                        axes[i].add_patch(rect)
                        
                        # Add method name above the group
                        group_center = (method_boxes[0] + method_boxes[-1]) / 2
                        axes[i].text(group_center, 100, method_mapping.get(method, method), 
                                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Professional styling for axes
        axes[i].set_title(f'{simulator_mapping[sim_type]} Coordination', 
                         fontsize=24, pad=30)
            
        # axes[i].set_ylabel('Resource Utilization [%]', fontsize=2)
        # axes[i].set_xlabel('α [-]', fontsize=12)
        axes[i].set_ylim(0, 105)
        axes[i].tick_params(axis='y', labelsize=18)  # Increase y-axis font size
        axes[i].grid(True, alpha=0.3, axis='y', linestyle='--')
        # axes[i].spines['top'].set_visible(False)
        # axes[i].spines['right'].set_visible(False)
        
    # Set shared x-label and y-label
    fig.text(0.5, 0.06, 'α [-]', ha='center', fontsize=28)
    fig.text(0.03, 0.5, 'Remaining Resources [%]', va='center', rotation='vertical', fontsize=28)

    # Add comprehensive legend at bottom center
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), 
              ncol=3, fontsize=20, frameon=True, fancybox=True, shadow=True,
              title='Methods', title_fontsize=28)
    
    # Better spacing and layout - more space for coordination titles and legend
    plt.subplots_adjust(top=0.90, hspace=0.4)  # More space between subplots for coordination titles
    
    # Save high-quality PNG
    plt.savefig(output_dir / 'resource_utilization_boxplot.png', 
                dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"✓ Resource utilization box plot saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis for journal paper")
    parser.add_argument("--baselines", help="Baseline results JSON file", required=True)
    parser.add_argument("--bonus-files", nargs='+', help="Bonus results JSON files", required=True)
    parser.add_argument("--output-dir", help="Output directory", default="journal_analysis")
    
    args = parser.parse_args()
    
    # Load all results
    print("Loading all result files...")
    all_results = []
    
    # Load baseline results
    baseline_results = load_results(args.baselines)
    all_results.extend(baseline_results)
    print(f"✓ Loaded {len(baseline_results)} baseline episodes")
    
    # Load bonus results
    for bonus_file in args.bonus_files:
        bonus_results = load_results(bonus_file)
        all_results.extend(bonus_results)
        print(f"✓ Loaded {len(bonus_results)} episodes from {Path(bonus_file).name}")
    
    print(f"✓ Total episodes: {len(all_results)}")
    
    # Create unified dataframe
    df = create_unified_dataframe(all_results, args.baselines)
        
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate comprehensive analysis
    print("\nGenerating comprehensive analysis...")
    
    # 1. Comprehensive performance table
    generate_comprehensive_latex_table(df, output_dir / 'comprehensive_performance_table.tex')
    
    # 2. Action distribution plot
    generate_action_distribution_plot(df, output_dir)
    
    # 3. Mission accomplishment box plot
    generate_mission_accomplishment_boxplot(df, output_dir)
    
    # 4. Resource utilization box plot
    generate_resource_utilization_boxplot(df, output_dir)
    
    # 5. Save processed data
    df.to_csv(output_dir / 'unified_analysis_data.csv', index=False)
    
    print(f"\n✓ All journal paper outputs saved to {output_dir}")
    print("📊 LaTeX table: comprehensive_performance_table.tex")
    print("📈 Action plot: action_distribution_analysis.png")
    print("📦 Mission plot: mission_accomplishment_boxplot.png")
    print("📋 Resource utilization plot: resource_utilization_boxplot.png")
    print("📋 Data: unified_analysis_data.csv")

if __name__ == "__main__":
    main()