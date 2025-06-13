import os
import time
import argparse
import numpy as np
import json
import psutil
import platform
from datetime import datetime
from pathlib import Path
import ray
import torch
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.tune.registry import register_env
from src.envs.FSS_env_v1 import FSS_env
from rule_based_policy import RuleBasedPolicy
from mip_policy import MIPPolicy
from benchmark_utils import BenchmarkMetrics, get_system_info
import signal

# Add timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Episode execution timed out")

def env_creator(env_config):
    """Environment creator function for Ray registration"""
    return FSS_env(env_config)

class PolicyBenchmark:
    def __init__(self, env_config, system_info):
        self.env_config = env_config
        self.system_info = system_info
        
        # Register environment with Ray (following compute_actions.py pattern)
        register_env("FSS_env", env_creator)
        
        # Initialize environment directly (not through Ray's env creation)
        print(f"Initializing environment with config: {env_config}")
        try:
            self.env = FSS_env(env_config)
            print(f"✓ Environment initialized successfully")
        except Exception as e:
            print(f"✗ Environment initialization failed: {e}")
            raise
        
    def load_rl_module(self, checkpoint_path, case_name):
        """Load RL module using official RLlib documentation approach"""
        try:
            print(f"Loading {case_name} from {checkpoint_path}")
            
            # Convert to absolute path
            from pathlib import Path
            import os
            checkpoint_path = Path(os.path.abspath(checkpoint_path))
            print(f"Absolute path: {checkpoint_path}")
            
            if not checkpoint_path.exists():
                print(f"Checkpoint path does not exist: {checkpoint_path}")
                return None
            
            # Method 1: Load RLModule directly (official documentation method)
            try:
                from ray.rllib.core.rl_module.rl_module import RLModule
                
                # Correct path structure from official docs: 
                # checkpoint_dir / "learner_group" / "learner" / "rl_module" / "autops-rl_policy"
                rl_module_checkpoint_dir = checkpoint_path / "learner_group" / "learner" / "rl_module" / "autops-rl_policy"
                
                print(f"Looking for RLModule at: {rl_module_checkpoint_dir}")
                
                if rl_module_checkpoint_dir.exists():
                    print("✓ RLModule checkpoint directory found")
                    rl_module = RLModule.from_checkpoint(str(rl_module_checkpoint_dir))
                    print(f"✓ Successfully loaded RLModule for {case_name}")
                    return rl_module
                else:
                    print(f"✗ RLModule checkpoint directory not found: {rl_module_checkpoint_dir}")
                    # List what's actually in the checkpoint directory
                    print("Checkpoint directory contents:")
                    if checkpoint_path.exists():
                        for item in checkpoint_path.rglob("*"):
                            if item.is_dir():
                                print(f"  📁 {item.relative_to(checkpoint_path)}")
                
            except Exception as rl_module_error:
                print(f"RLModule loading failed: {rl_module_error}")
                import traceback
                traceback.print_exc()
            
            # Method 2: Algorithm loading (fallback)
            try:
                print("Falling back to Algorithm loading...")
                from ray.rllib.algorithms.algorithm import Algorithm
                
                algo = Algorithm.from_checkpoint(str(checkpoint_path))
                
                # Get policy (following official docs pattern)
                policy = algo.get_policy("autops-rl_policy")
                if policy is not None:
                    print(f"✓ Loaded policy for {case_name}")
                    
                    # Create wrapper for consistent interface
                    class PolicyWrapper:
                        def __init__(self, policy):
                            self.policy = policy
                        
                        def forward_inference(self, inputs):
                            obs = inputs["obs"]
                            actions, state_outs, extra = self.policy.compute_actions_from_input_dict(
                                {"obs": obs}
                            )
                            return {"action_dist_inputs": actions}
                    
                    wrapper = PolicyWrapper(policy)
                    algo.stop()
                    return wrapper
                else:
                    print(f"No policy found for {case_name}")
                    algo.stop()
                    return None
                
            except Exception as algo_error:
                print(f"Algorithm loading failed: {algo_error}")
                import traceback
                traceback.print_exc()
            
            print(f"✗ All loading methods failed for {case_name}")
            return None
            
        except Exception as e:
            print(f"✗ Failed to load {case_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_episode(self, policy, policy_name, max_steps=None):
        """Run a complete episode with the given policy"""
        metrics = BenchmarkMetrics(
            policy_name=policy_name,
            num_agents=self.env_config["num_observers"],
            num_targets=self.env_config["num_targets"],
            system_info=self.system_info
        )
        
        # Reset environment
        try:
            obs, info = self.env.reset()
            print(f"✓ Environment reset successful for {policy_name}")
        except Exception as e:
            print(f"✗ Environment reset failed: {e}")
            raise
            
        done = False
        step_count = 0
        
        print(f"Running episode with {policy_name} on {self.env_config['simulator_type']} simulator...")
        episode_start_time = time.perf_counter()
        
        while not done and (max_steps is None or step_count < max_steps):
            step_start_time = time.perf_counter()
            
            # Compute actions for all agents
            if policy_name.startswith("Case") or "bonus" in policy_name.lower():
                # RL policy - use RLModule directly with proper tensor formatting (from working LRZ version)
                action_start_time = time.perf_counter()
                actions = {}
                
                # For each agent, compute action using the RL module
                for agent_id, agent_obs in obs.items():
                    if agent_id.startswith("observer_"):
                        try:
                            # Convert observation to tensor and add batch dimension (from working version)
                            import torch
                            
                            # Handle dict observations like working version
                            if isinstance(agent_obs, dict):
                                # Flatten all observation components into a single array
                                # This matches how RLLib processes dict observations
                                obs_components = []
                                for key in sorted(agent_obs.keys()):  # Sort for consistency
                                    component = agent_obs[key]
                                    if isinstance(component, np.ndarray):
                                        obs_components.append(component.flatten())
                                    else:
                                        obs_components.append(np.array([component]).flatten())
                                
                                actual_obs = np.concatenate(obs_components)
                            else:
                                actual_obs = agent_obs
                            
                            if isinstance(actual_obs, np.ndarray):
                                obs_tensor = torch.from_numpy(actual_obs).float()
                            else:
                                obs_tensor = torch.tensor(actual_obs, dtype=torch.float32)
                            
                            # Add batch dimension (RLModule expects batched input)
                            obs_batch = obs_tensor.unsqueeze(0)  # Shape: (1, obs_dim)
                            
                            # Use forward_inference with proper input format
                            with torch.no_grad():  # No gradients needed for inference
                                action_result = policy.forward_inference({"obs": obs_batch})
                            
                            # Extract the action from the result - we know it's action_dist_inputs
                            if isinstance(action_result, dict) and 'action_dist_inputs' in action_result:
                                action_logits = action_result['action_dist_inputs']  # Shape: [1, 3]
                                
                                # For discrete actions, use deterministic (argmax) like working version
                                action = action_logits.argmax(dim=-1).item()  # Get scalar action
                                
                            else:
                                print(f"Unexpected action result format: {action_result}")
                                action = 0
                            
                            # Ensure action is a valid integer for the environment
                            action = int(action)
                            actions[agent_id] = action
                            
                        except Exception as e:
                            print(f"Warning: Action computation failed for {agent_id}, using default action: {e}")
                            import traceback
                            traceback.print_exc()
                            actions[agent_id] = 0  # Default to idle action
                
                action_time = time.perf_counter() - action_start_time
            
            elif policy_name == "RuleBased":
                # Rule-based policy
                action_start_time = time.perf_counter()
                actions = policy.compute_actions(obs, self.env)
                action_time = time.perf_counter() - action_start_time
                
            elif policy_name == "MIP":
                # MIP policy
                action_start_time = time.perf_counter()
                actions = policy.compute_actions(obs, self.env)
                action_time = time.perf_counter() - action_start_time
            
            # Record action timing and choices
            metrics.add_action_time(action_time)
            for agent_id, action in actions.items():
                metrics.record_action(agent_id, action)
            
            # Step environment
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            step_time = time.perf_counter() - step_start_time
            
            # Record step metrics
            metrics.add_step_time(step_time)
            metrics.add_rewards(rewards)
            
            # Check if done
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            step_count += 1
        
        # Record episode timing
        episode_time = time.perf_counter() - episode_start_time
        metrics.set_episode_time(episode_time)
        
        # Get final environment metrics
        final_metrics = self.env.collect_comprehensive_metrics()
        metrics.set_environment_metrics(final_metrics)
        
        print(f"✓ Completed episode with {policy_name} on {self.env_config['simulator_type']} in {episode_time:.2f}s ({step_count} steps)")
        
        # Add step_count to the results before returning
        results = metrics.get_results()
        results['step_count'] = step_count
        
        return results

def get_sensitivity_checkpoint_paths(cases_to_test, best_seeds):
    """
    Define checkpoint paths for sensitivity analysis cases with specific bonus values.
    For bonus01 cases, finds the directory with the highest SLURM job number.
    For other bonus values, uses the new naming convention.
    """
    print("Getting sensitivity analysis checkpoint paths...")
    
    # Import required modules
    import glob
    import os
    
    auto_paths = {}
    
    for case_bonus in cases_to_test:
        case_num = case_bonus.split('_')[0]  # e.g., "case1" from "case1_bonus0"
        bonus_key = case_bonus.split('_')[1]  # e.g., "bonus0" from "case1_bonus0"
        
        # Get the best seed for this case and bonus
        seed = best_seeds.get(case_bonus, 42)  # fallback to seed 42
        
        print(f"Looking for {case_bonus} with seed {seed}...")
        
        if bonus_key == "bonus01":
            # For bonus01 cases, find the directory with the highest SLURM job number
            pattern = f"/workspace/checkpoints_{case_num}_training_*"
            print(f"  Searching pattern: {pattern}")
            
            try:
                matching_dirs = glob.glob(pattern)
                print(f"  Found {len(matching_dirs)} matching directories: {matching_dirs}")
                
                if matching_dirs:
                    # Extract job numbers and find the highest one
                    job_numbers = []
                    for dir_path in matching_dirs:
                        try:
                            # Extract the job number from the directory name
                            job_num = int(dir_path.split('_')[-1])
                            job_numbers.append((job_num, dir_path))
                            print(f"    Parsed job number {job_num} from {dir_path}")
                        except (ValueError, IndexError) as e:
                            print(f"    Failed to parse job number from {dir_path}: {e}")
                            continue
                    
                    if job_numbers:
                        # Sort by job number and get the highest
                        job_numbers.sort(key=lambda x: x[0])
                        highest_job_num, highest_dir = job_numbers[-1]
                        
                        print(f"  Found directory with highest job number {highest_job_num}: {highest_dir}")
                        
                        # For bonus01, look for the specific seed first in the highest SLURM job directory
                        print(f"  Looking for specific seed {seed} in highest job directory...")
                        specific_pattern = os.path.join(highest_dir, "everyone", f"best_{case_num}_seed{seed}_sim_everyone.ckpt")
                        print(f"  Trying specific pattern: {specific_pattern}")
                        specific_matches = glob.glob(specific_pattern)
                        
                        if specific_matches:
                            # Found the specific seed - use it
                            auto_paths[case_bonus] = specific_matches[0]
                            print(f"  ✓ Found specific seed {seed}: {specific_matches[0]}")
                        else:
                            # Fallback: use any available checkpoint from the highest SLURM job directory
                            print(f"  ❌ Specific seed {seed} not found, falling back to any available seed...")
                            alt_pattern = os.path.join(highest_dir, "everyone", f"best_{case_num}_seed*_sim_everyone.ckpt")
                            print(f"  Looking for any checkpoint matching: {alt_pattern}")
                            alt_matches = glob.glob(alt_pattern)
                            print(f"  Found checkpoint files: {alt_matches}")
                            
                            if alt_matches:
                                # Use the first available checkpoint from the highest SLURM job directory
                                auto_paths[case_bonus] = alt_matches[0]
                                print(f"  ⚠️  Using fallback checkpoint: {alt_matches[0]}")
                                # Extract and show what seed was actually selected
                                try:
                                    actual_seed = os.path.basename(alt_matches[0]).split('_seed')[1].split('_')[0]
                                    print(f"      📍 Actually using seed: {actual_seed}")
                                except:
                                    print(f"      📍 Could not extract seed from: {alt_matches[0]}")
                            else:
                                print(f"  ✗ No checkpoint files found in {highest_dir}/everyone/")
                    else:
                        print(f"  ✗ No valid job numbers found in directory names")
                else:
                    print(f"  ✗ No training directories found for pattern: {pattern}")
                    
            except Exception as e:
                print(f"  ✗ Error processing bonus01 case {case_bonus}: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            # For other bonus values, use the new naming convention
            print(f"  Processing non-bonus01 case: {case_bonus}")
            print(f"  Looking for seed: {seed}")
            
            # Try specific seed first (highest priority)
            specific_seed_patterns = [
                f"/workspace/checkpoints_{case_num}_{bonus_key}_*/everyone/best_{case_num}_seed{seed}_sim_everyone.ckpt",
                f"checkpoints_{case_num}_{bonus_key}_*/everyone/best_{case_num}_seed{seed}_sim_everyone.ckpt",
            ]
            
            specific_matches = []
            for pattern in specific_seed_patterns:
                print(f"    Trying specific seed pattern: {pattern}")
                pattern_matches = glob.glob(pattern, recursive=True)
                specific_matches.extend(pattern_matches)
                print(f"    Found {len(pattern_matches)} specific matches: {pattern_matches}")
            
            if specific_matches:
                # Use the most recent specific seed match (largest job ID if multiple)
                selected_path = sorted(specific_matches)[-1]
                auto_paths[case_bonus] = os.path.abspath(selected_path)
                print(f"  ✓ Found {case_bonus} with specific seed {seed}: {auto_paths[case_bonus]}")
            else:
                # Fallback to wildcard seed patterns if specific seed not found
                print(f"    ❌ No specific seed {seed} found, trying wildcard patterns...")
                wildcard_patterns = [
                    f"/workspace/checkpoints_{case_num}_{bonus_key}_*/everyone/best_{case_num}_seed*_sim_everyone.ckpt",
                    f"checkpoints_{case_num}_{bonus_key}_*/everyone/best_{case_num}_seed*_sim_everyone.ckpt",
                ]
                
                wildcard_matches = []
                for pattern in wildcard_patterns:
                    print(f"    Trying wildcard pattern: {pattern}")
                    pattern_matches = glob.glob(pattern, recursive=True)
                    wildcard_matches.extend(pattern_matches)
                    print(f"    Found {len(pattern_matches)} wildcard matches: {pattern_matches}")
                
                if wildcard_matches:
                    # Use the most recent wildcard match (largest job ID if multiple)
                    selected_path = sorted(wildcard_matches)[-1]
                    auto_paths[case_bonus] = os.path.abspath(selected_path)
                    print(f"  ⚠️  Found {case_bonus} with fallback seed (not {seed}): {auto_paths[case_bonus]}")
                    # Extract and show what seed was actually selected
                    try:
                        actual_seed = os.path.basename(selected_path).split('_seed')[1].split('_')[0]
                        print(f"      📍 Actually using seed: {actual_seed}")
                    except:
                        print(f"      📍 Could not extract seed from: {selected_path}")
                else:
                    print(f"  ✗ No checkpoint found for {case_bonus}")
                    print(f"    All searched patterns: {specific_seed_patterns + wildcard_patterns}")
    
    # Verify paths exist
    print(f"\nVerifying {len(auto_paths)} checkpoint paths...")
    verified_paths = {}
    for case_bonus, path in auto_paths.items():
        print(f"Checking {case_bonus}: {path}")
        try:
            if os.path.exists(path):
                verified_paths[case_bonus] = path
                print(f"✓ Verified {case_bonus}: {path}")
            else:
                print(f"✗ Missing {case_bonus}: {path}")
        except Exception as e:
            print(f"✗ Error checking {case_bonus}: {e}")
    
    print(f"Final verified paths: {len(verified_paths)} out of {len(auto_paths)}")
    return verified_paths

def run_sensitivity_benchmark(cases_to_test, best_seeds, num_episodes=5, max_steps_per_episode=None, include_baselines=True):
    """Run benchmark across sensitivity analysis configurations"""
    
    print("="*80)
    print("SENSITIVITY ANALYSIS BENCHMARK")
    print("="*80)
    
    # Print current directory and contents
    import os
    print(f"Current working directory: {os.getcwd()}")
    print("Contents of current directory:")
    for item in os.listdir('.'):
        if 'checkpoint' in item.lower() or 'case' in item.lower():
            print(f"  📁 {item}")
    
    # Test checkpoint detection
    checkpoint_paths = get_sensitivity_checkpoint_paths(cases_to_test, best_seeds)
    
    if not checkpoint_paths:
        print("❌ No checkpoint paths found! Please check:")
        print("1. Are you in the right directory?")
        print("2. Do checkpoint directories exist?")
        print("3. Are checkpoint files named correctly?")
        return [], None
    
    # Get system info
    system_info = get_system_info()
    print(f"System: {system_info['cpu_cores']} cores @ {system_info['cpu_freq']:.1f} GHz")
    
    # Configuration for benchmarking
    base_config = {
        "num_observers": 20, 
        "num_targets": 100, 
        "time_step": 1, 
        "duration": 86400, 
        "seed": 47, 
        "reward_type": "case1"  # Will be updated per case
    }
    
    # Simulator types to test across
    simulator_types = ["everyone", "centralized", "decentralized"]
    
    all_results = []
    
    print(f"Testing cases: {cases_to_test}")
    print(f"Episodes per case: {num_episodes}")
    print("="*80)
    
    for sim_type in simulator_types:
        print(f"\n--- Testing on {sim_type.upper()} simulator ---")
        
        # Create environment config for this simulator type
        env_config = base_config.copy()
        env_config["simulator_type"] = sim_type
        
        # Initialize benchmark for this configuration
        benchmark = PolicyBenchmark(env_config, system_info)
        
        # Test baseline policies if requested
        if include_baselines:
            baseline_policies = [
                ("RuleBased", RuleBasedPolicy()),
                ("MIP", MIPPolicy()),
            ]
            
            # Run baseline policies
            for policy_name, policy in baseline_policies:
                print(f"\n✓ Running {policy_name}...")
                
                for episode in range(num_episodes):
                    try:
                        result = benchmark.run_episode(
                            policy, 
                            policy_name, 
                            max_steps=max_steps_per_episode
                        )
                        
                        # Add metadata to result
                        result["config_name"] = "sensitivity_analysis"
                        result["simulator_type"] = sim_type
                        result["training_simulator"] = "everyone"
                        result["test_simulator"] = sim_type
                        result["episode_number"] = episode + 1
                        result["case_type"] = "baseline"
                        result["bonus_value"] = "N/A"
                        result["base_config"] = base_config
                        
                        all_results.append(result)
                        
                        print(f"  Episode {episode + 1}/{num_episodes}:")
                        print(f"    NET per agent: {result['metrics']['net_per_agent']:.6f}")
                        print(f"    Mission completion: {result['metrics']['mission_percentage']:.1f}%")
                        print(f"    Avg resources left: {result['metrics']['average_resources_left']:.3f}")
                        
                    except Exception as e:
                        print(f"✗ Episode {episode + 1} failed: {e}")
                        continue
        
        # Test RL policies for each case
        for case_bonus, checkpoint_path in checkpoint_paths.items():
            case_num = case_bonus.split('_')[0]  # e.g., "case1"
            bonus_key = case_bonus.split('_')[1]  # e.g., "bonus0"
            
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Checkpoint not found for {case_bonus}: {checkpoint_path}")
                continue
            
            print(f"\n✓ Running {case_bonus}...")
            
            # Update environment config for this case
            env_config["reward_type"] = case_num
            benchmark = PolicyBenchmark(env_config, system_info)
            
            # Load RL module for this case
            rl_module = benchmark.load_rl_module(checkpoint_path, case_bonus)
            if rl_module is None:
                continue
            
            policy_name = f"{case_num}_{bonus_key}"
            
            for episode in range(num_episodes):
                print(f"\n🚀 STARTING Episode {episode + 1}/{num_episodes} for {case_bonus}")
                print(f"🕐 Started at: {time.strftime('%H:%M:%S')}")
                
                try:
                    episode_start_time = time.perf_counter()
                    result = benchmark.run_episode(
                        rl_module, 
                        policy_name, 
                        max_steps=max_steps_per_episode
                    )
                    episode_duration = time.perf_counter() - episode_start_time
                    
                    # Add metadata to result
                    result["config_name"] = "sensitivity_analysis"
                    result["simulator_type"] = sim_type
                    result["training_simulator"] = "everyone"
                    result["test_simulator"] = sim_type
                    result["episode_number"] = episode + 1
                    result["case_type"] = case_num
                    result["bonus_value"] = bonus_key
                    result["best_seed"] = best_seeds.get(case_bonus, 42)
                    result["base_config"] = base_config
                    
                    all_results.append(result)
                    
                    print(f"✅ COMPLETED Episode {episode + 1}/{num_episodes} in {episode_duration:.1f}s")
                    print(f"    NET per agent: {result['metrics']['net_per_agent']:.6f}")
                    print(f"    Mission completion: {result['metrics']['mission_percentage']:.1f}%")
                    print(f"    Avg resources left: {result['metrics']['average_resources_left']:.3f}")
                    
                except Exception as e:
                    print(f"❌ Episode {episode + 1} FAILED: {e}")
                    continue
    
    # Only create experiment folder if we have actual results
    if not all_results:
        print(f"\n{'='*80}")
        print(f"❌ NO RESULTS TO SAVE!")
        print(f"No policies could be loaded or episodes completed successfully.")
        print(f"Check checkpoint paths and loading methods.")
        print(f"{'='*80}")
        return [], None
    
    # Create experiment folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/sensitivity_benchmark_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results in the experiment folder
    results_filename = experiment_dir / "sensitivity_results.json"
    
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        "timestamp": timestamp,
        "cases_tested": cases_to_test,
        "best_seeds": best_seeds,
        "simulator_types": simulator_types,
        "episodes_per_case": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "total_episodes": len(all_results),
        "system_info": system_info,
        "checkpoint_paths": checkpoint_paths,
        "include_baselines": include_baselines
    }
    
    metadata_filename = experiment_dir / "experiment_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SENSITIVITY BENCHMARK COMPLETE!")
    print(f"Experiment folder: {experiment_dir}")
    print(f"Results saved to: {results_filename}")
    print(f"Metadata saved to: {metadata_filename}")
    print(f"Cases tested: {len(cases_to_test)}")
    print(f"Total episodes: {len(all_results)}")
    print(f"")
    print(f"This benchmark tests sensitivity analysis results across:")
    print(f"- Cases: {[c.split('_')[0] for c in cases_to_test]}")
    print(f"- Bonus values: {list(set([c.split('_')[1] for c in cases_to_test]))}")
    print(f"- Simulator types: {simulator_types}")
    print(f"")
    print(f"To analyze results, run:")
    print(f"    python analyze_sensitivity_results.py {results_filename}")
    print(f"{'='*80}")
    
    return all_results, experiment_dir

def main():
    parser = argparse.ArgumentParser(description="Benchmark sensitivity analysis results")
    parser.add_argument("--bonus-groups", type=str, required=True,
                       choices=["bonus01", "bonus0", "bonus05", "bonus10", "all"],
                       help="Which bonus groups to test")
    parser.add_argument("--episodes", type=int, default=15,
                       help="Number of episodes per case")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum steps per episode (for testing)")
    parser.add_argument("--include-baselines", action="store_true", default=False,
                       help="Include MIP and RuleBased baselines (slow)")
    
    args = parser.parse_args()
    
    # Best seeds for each case and bonus combination (provided by user)
    best_seeds = {
        # Bonus 0.1, correctly gets the folders, not the seeds
        "case1_bonus01": 42, 
        "case2_bonus01": 45,
        "case3_bonus01": 43,
        "case4_bonus01": 43,
        # Bonus 0, correctly gets checkpoints_caseX_bonus0_XXXXXX/ folder, not the seeds
        "case1_bonus0": 44,
        "case2_bonus0": 46,
        "case3_bonus0": 42,
        "case4_bonus0": 43,
        # Bonus 0.5, correctly gets checkpoints_caseX_bonus05_XXXXXX/ folder, not the seeds
        "case1_bonus05": 45,
        "case2_bonus05": 42,
        "case3_bonus05": 42,
        "case4_bonus05": 42,
        # Bonus 1.0, correctly gets checkpoints_caseX_bonus10_XXXXXX/ folder, not the seeds
        "case1_bonus10": 45,
        "case2_bonus10": 45,
        "case3_bonus10": 46,
        "case4_bonus10": 42,
    }
    
    # Define which cases to test based on argument
    if args.bonus_groups == "all":
        cases_to_test = list(best_seeds.keys())
    elif args.bonus_groups == "bonus01":
        cases_to_test = [f"case{i}_bonus01" for i in range(1, 5)]
    elif args.bonus_groups == "bonus0":
        cases_to_test = [f"case{i}_bonus0" for i in range(1, 5)]
    elif args.bonus_groups == "bonus05":
        cases_to_test = [f"case{i}_bonus05" for i in range(1, 5)]
    elif args.bonus_groups == "bonus10":
        cases_to_test = [f"case{i}_bonus10" for i in range(1, 5)]
    
    # Initialize Ray in local mode to keep things simple (like working benchmark_policies.py)
    try:
        print("Initializing Ray in local mode...")
        ray.init(
            local_mode=True,  # Single process, no distributed overhead
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        
        # Register the environment with Ray
        register_env("FSS_env", env_creator)
        print("✓ Ray initialized successfully")
        
    except Exception as e:
        print(f"Warning: Ray initialization failed: {e}")
        print("Continuing without Ray")
    
    # Run benchmark
    try:
        results, experiment_dir = run_sensitivity_benchmark(
            cases_to_test, 
            best_seeds,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            include_baselines=args.include_baselines
        )
        if experiment_dir:  # Check if experiment_dir is not None
            print(f"📁 Experiment saved in: {experiment_dir}")
            print(f"📊 To analyze results, run:")
            print(f"    python analyze_sensitivity_results.py {experiment_dir / 'sensitivity_results.json'}")
        else:
            print("❌ No experiment directory created - benchmark failed")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            ray.shutdown()
        except:
            pass

if __name__ == "__main__":
    main() 