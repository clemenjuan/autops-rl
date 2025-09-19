#!/usr/bin/env python3
"""
Enhanced Training Curves Analysis - Exact styling match to comprehensive_analysis.py
Creates professional training plots for all reward cases and alpha values
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Configure matplotlib for journal paper quality (exact same as comprehensive_analysis.py)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def smooth(scalars, weight=0.9):
    """Same smoothing function as used in tensorboard smoothing"""
    if len(scalars) == 0:
        return []
    
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def load_all_training_data(training_dir="results/trainings"):
    """Load all training CSV files and organize by alpha values"""
    
    # Get all CSV files
    csv_files = glob.glob(f"{training_dir}/*.csv")
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = {}
    
    for csv_file in csv_files:
        print(f"Processing {Path(csv_file).name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Extract data for each case
            for case_num in [1, 2, 3, 4]:
                case_name = f'case{case_num}'
                
                # Column names for this case
                step_col = f'env_config.reward_type: {case_name} - _step'
                return_col = f'env_config.reward_type: {case_name} - env_runners/episode_return_mean'
                
                if step_col in df.columns and return_col in df.columns:
                    # Filter out NaN values
                    valid_mask = ~(pd.isna(df[step_col]) | pd.isna(df[return_col]))
                    
                    if valid_mask.any():
                        steps = df[step_col][valid_mask].values
                        returns = df[return_col][valid_mask].values
                        
                        # Use filename as alpha identifier
                        file_id = Path(csv_file).stem
                        alpha_key = file_id
                        
                        if alpha_key not in all_data:
                            all_data[alpha_key] = {}
                        
                        all_data[alpha_key][case_name] = {
                            'steps': steps,
                            'returns': returns
                        }
                        
                        print(f"  ✓ Loaded {len(steps)} data points for {case_name}")
                    
        except Exception as e:
            print(f"  ❌ Error processing {csv_file}: {e}")
    
    # Map file IDs to alpha values based on timestamp patterns
    # Sort by timestamp to maintain chronological order
    sorted_keys = sorted(all_data.keys())
    
    # Define alpha values in the order they were trained
    alpha_values = ['0.0', '0.1', '0.5', '1.0', '2.0', '5.0']
    
    # Create mapping based on the order of files (oldest to newest)
    alpha_mapping = {}
    for i, key in enumerate(sorted_keys):
        if i < len(alpha_values):
            alpha_mapping[key] = alpha_values[i]
        else:
            # Handle any extra files with generic labels
            alpha_mapping[key] = f'α_{i}'
    
    # Print mapping for verification
    print("\n📋 File to Alpha mapping:")
    for file_id, alpha_val in alpha_mapping.items():
        print(f"  {file_id} → α = {alpha_val}")
    
    # Reorganize data with proper alpha labels
    organized_data = {}
    for file_id, alpha_val in alpha_mapping.items():
        if file_id in all_data:
            organized_data[alpha_val] = all_data[file_id]
    
    return organized_data

def create_professional_training_plot(all_data, output_file="professional_training_curves.png"):
    """Create professional training curves plot with exact comprehensive_analysis.py styling"""
    
    # Method name mapping with acronyms for legend and labels (exact same as comprehensive_analysis.py)
    case_mapping = {
        'case1': 'IP',  # Individual Positive
        'case2': 'IN',  # Individual Negative
        'case3': 'CP',  # Collective Positive
        'case4': 'CN'   # Collective Negative
    }
    
    # Full method names for legend
    case_full_names = {
        'case1': 'Individual Positive (IP)',
        'case2': 'Individual Negative (IN)',
        'case3': 'Collective Positive (CP)',
        'case4': 'Collective Negative (CN)'
    }
    
    # Professional color palette (exact same as comprehensive_analysis.py)
    case_colors = {
        'case1': '#1F77B4',  # Blue
        'case2': '#2CA02C',  # Green
        'case3': '#9467BD',  # Purple
        'case4': '#8C564B'   # Brown
    }
    
    # Create subplots for each alpha value
    alpha_values = sorted(all_data.keys())
    n_alphas = len(alpha_values)
    
    if n_alphas == 0:
        print("❌ No data to plot!")
        return
    
    # Create figure with professional layout (2 rows, 3 columns for 6 alpha values)
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes = axes.flatten()
    
    # Collect data bounds for consistent axes
    all_steps = []
    all_returns = []
    
    # First pass: collect all data for consistent axis limits
    for alpha_val in alpha_values:
        if alpha_val in all_data:
            alpha_data = all_data[alpha_val]
            for case_name in ['case1', 'case2', 'case3', 'case4']:
                if case_name in alpha_data:
                    data = alpha_data[case_name]
                    all_steps.extend(data['steps'])
                    all_returns.extend(data['returns'])
    
    # Calculate consistent axis limits
    if all_steps and all_returns:
        x_min, x_max = min(all_steps), max(all_steps)
        y_min, y_max = min(all_returns), max(all_returns)
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.1
        
        x_limits = (max(0, x_min - x_padding), x_max + x_padding)
        y_limits = (max(0, y_min - y_padding), y_max + y_padding)
    else:
        # Fallback limits
        x_limits = (0, 100)
        y_limits = (50, 200)
    
    print(f"📊 Consistent axis limits: X=[{x_limits[0]:.1f}, {x_limits[1]:.1f}], Y=[{y_limits[0]:.1f}, {y_limits[1]:.1f}]")
    
    # Collect legend information
    legend_elements = []
    legend_added = set()
    
    for i, alpha_val in enumerate(alpha_values):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if alpha_val in all_data:
            alpha_data = all_data[alpha_val]
            
            # Plot each case for this alpha value
            for case_name in ['case1', 'case2', 'case3', 'case4']:
                if case_name in alpha_data:
                    data = alpha_data[case_name]
                    steps = data['steps']
                    returns = data['returns']
                    
                    # Smooth the data
                    smoothed_returns = smooth(returns.tolist(), weight=0.9)
                    
                    # Calculate rolling standard deviation for confidence bands
                    window_size = min(5, len(returns) // 8)
                    if len(returns) > window_size:
                        rolling_std = pd.Series(returns).rolling(window=window_size, center=True).std().fillna(2.0)
                    else:
                        rolling_std = np.full_like(returns, 2.0)
                    
                    # Plot the line
                    color = case_colors[case_name]
                    label = case_full_names[case_name]
                    
                    ax.plot(steps, smoothed_returns, label=label, color=color, linewidth=2)
                    ax.fill_between(steps, 
                                   np.array(smoothed_returns) - rolling_std.values, 
                                   np.array(smoothed_returns) + rolling_std.values, 
                                   alpha=0.2, color=color)
                    
                    # Add to legend if not already added
                    if case_name not in legend_added:
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, 
                                                            label=case_full_names[case_name]))
                        legend_added.add(case_name)
        
        # Professional styling for each subplot with consistent axes
        ax.set_title(f'α = {alpha_val}', fontsize=24, pad=30)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set consistent axis limits across all subplots
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
    
    # Hide unused subplots if any
    for j in range(len(alpha_values), len(axes)):
        axes[j].set_visible(False)
    
    # Set shared labels (exact same as comprehensive_analysis.py)
    fig.text(0.5, 0.06, 'Training Iteration [-]', ha='center', fontsize=28)
    fig.text(0.03, 0.5, 'Episode Return Mean [-]', va='center', rotation='vertical', fontsize=28)
    
    # Add comprehensive legend at bottom center (exact same as comprehensive_analysis.py)
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06), 
              ncol=4, fontsize=20, frameon=True, fancybox=True, shadow=True,
              title='Reward Functions', title_fontsize=28)
    
    # Better spacing and layout (exact same as comprehensive_analysis.py)
    plt.subplots_adjust(top=0.90, hspace=0.4, bottom=0.12)
    
    # Save high-quality PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"✓ Professional training plot saved to {output_file}")

def create_single_plot_styled(all_data, output_file="single_training_curves_styled.png"):
    """Create a single plot with all cases and alpha values using professional styling"""
    
    # Case name mapping
    case_full_names = {
        'case1': 'Individual Positive (IP)',
        'case2': 'Individual Negative (IN)',
        'case3': 'Collective Positive (CP)',
        'case4': 'Collective Negative (CN)'
    }
    
    # Professional color palette
    case_colors = {
        'case1': '#1F77B4',  # Blue
        'case2': '#2CA02C',  # Green
        'case3': '#9467BD',  # Purple
        'case4': '#8C564B'   # Brown
    }
    
    # Line styles for different alpha values
    alpha_styles = {
        '0.0': '-',     # Solid
        '0.1': '--',    # Dashed
        '0.5': '-.',    # Dash-dot
        '1.0': ':',     # Dotted
        '2.0': '-',     # Solid (repeat pattern)
        '5.0': '--'     # Dashed
    }
    
    # Create single large figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot all combinations
    for alpha_val in sorted(all_data.keys()):
        alpha_data = all_data[alpha_val]
        
        for case_name in ['case1', 'case2', 'case3', 'case4']:
            if case_name in alpha_data:
                data = alpha_data[case_name]
                steps = data['steps']
                returns = data['returns']
                
                # Smooth the data
                smoothed_returns = smooth(returns.tolist(), weight=0.9)
                
                # Plot styling
                color = case_colors[case_name]
                label = f"{case_full_names[case_name]} (α={alpha_val})"
                linestyle = alpha_styles.get(alpha_val, '-')
                
                ax.plot(steps, smoothed_returns, label=label, color=color, 
                       linewidth=2, linestyle=linestyle, alpha=0.8)
    
    # Professional styling (same as comprehensive_analysis.py)
    ax.set_title('Training Curves: All Reward Cases and Alpha Values', fontsize=24, pad=20)
    ax.set_xlabel('Training Iteration [-]', fontsize=28)
    ax.set_ylabel('Episode Return Mean [-]', fontsize=28)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=18)
    
    # Legend with professional styling
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, 
             frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close()
    
    print(f"✓ Single styled plot saved to {output_file}")

def main():
    """Main function to create all training plots"""
    
    print("🚀 Starting enhanced training curves analysis...")
    print("📁 Scanning for training data in results/trainings/")
    
    # Load all training data
    all_data = load_all_training_data()
    
    if not all_data:
        print("❌ No training data found!")
        return
    
    print(f"\n✓ Loaded data for {len(all_data)} alpha values:")
    for alpha_val, cases in all_data.items():
        case_info = []
        for case_name, case_data in cases.items():
            n_points = len(case_data['steps'])
            case_info.append(f"{case_name}({n_points} pts)")
        print(f"  α = {alpha_val}: {', '.join(case_info)}")
    
    # Create professional training curves plot with consistent axes
    create_professional_training_plot(all_data, "professional_training_curves_all_alpha.png")
    
    # Also create a single combined plot
    create_single_plot_styled(all_data, "single_training_curves_all_alpha.png")
    
    print("\n✅ Enhanced training curves analysis complete!")
    print("📊 Files created:")
    print("   - professional_training_curves_all_alpha.png (2x3 subplots with consistent axes)")
    print("   - single_training_curves_all_alpha.png (all curves in one plot)")
    print("\n📋 Notes:")
    print("   - All subplots use consistent X and Y axis limits for easy comparison")
    print(f"   - Includes training results for α = {sorted(all_data.keys())}")
    print("   - Smoothed curves with confidence bands for better visualization")
    print("   - Single plot version available for overall comparison")

if __name__ == "__main__":
    main() 