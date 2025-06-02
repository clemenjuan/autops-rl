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
        """Load RL module from checkpoint using simplified approach"""
        try:
            print(f"Loading {case_name} from {checkpoint_path}")
            
            from pathlib import Path
            checkpoint_path = Path(checkpoint_path).resolve()
            
            if not checkpoint_path.exists():
                print(f"Checkpoint path does not exist: {checkpoint_path}")
                return None
            
            print(f"Checkpoint directory contents:")
            for item in checkpoint_path.rglob("*"):
                if item.is_file():
                    print(f"  {item.relative_to(checkpoint_path)}")
            
            # Try Ray Algorithm loading (simplest approach)
            try:
                print("Trying Ray Algorithm loading...")
                import os
                # Minimize resources
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
                
                from ray.rllib.algorithms.algorithm import Algorithm
                
                # Try loading with string path
                algo = Algorithm.from_checkpoint(str(checkpoint_path))
                
                # Get the policy/module
                policy = algo.get_policy("autops-rl_policy")
                if policy is not None:
                    print(f"✓ Loaded policy for {case_name}")
                    # Create a wrapper that mimics RLModule interface
                    class PolicyWrapper:
                        def __init__(self, policy):
                            self.policy = policy
                        
                        def forward_inference(self, inputs):
                            obs = inputs["obs"]
                            # Use policy.compute_actions_from_input_dict
                            actions, state_outs, extra = self.policy.compute_actions_from_input_dict(
                                {"obs": obs}
                            )
                            return {"action_dist_inputs": actions}
                    
                    wrapper = PolicyWrapper(policy)
                    algo.stop()
                    return wrapper
                
                # Try getting the module directly
                rl_module = algo.get_module()
                if rl_module is not None:
                    if hasattr(rl_module, '_rl_modules'):
                        rl_module = rl_module._rl_modules.get("autops-rl_policy", rl_module)
                    print(f"✓ Loaded RL module for {case_name}")
                    algo.stop()
                    return rl_module
                
                algo.stop()
                print(f"Could not extract policy or module from {case_name}")
                
            except Exception as algo_error:
                print(f"Algorithm loading failed: {algo_error}")
                import traceback
                traceback.print_exc()
            
            return None
            
        except Exception as e:
            print(f"✗ Failed to load RL module for {case_name}: {e}")
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
            if policy_name.startswith("Case"):
                # RL policy - use RLModule directly with proper tensor formatting
                actions = {}
                # For each agent, compute action using the RL module
                for agent_id, agent_obs in obs.items():
                    if agent_id.startswith("observer_"):
                        try:
                            # Convert observation to tensor and add batch dimension
                            import torch
                            
                            # Debug: Print observation format
                            # print(f"Debug: agent_obs type: {type(agent_obs)}")
                            if isinstance(agent_obs, dict):
                                # print(f"Debug: agent_obs keys: {list(agent_obs.keys())}")
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
                                # print(f"Debug: flattened obs shape: {actual_obs.shape}")
                            else:
                                actual_obs = agent_obs
                                # print(f"Debug: direct obs shape: {actual_obs.shape}")
                            
                            if isinstance(actual_obs, np.ndarray):
                                obs_tensor = torch.from_numpy(actual_obs).float()
                            else:
                                obs_tensor = torch.tensor(actual_obs, dtype=torch.float32)
                            
                            # Add batch dimension (RLModule expects batched input)
                            obs_batch = obs_tensor.unsqueeze(0)  # Shape: (1, obs_dim)
                            
                            # Use forward_inference with proper input format
                            with torch.no_grad():  # No gradients needed for inference
                                action_start_time = time.perf_counter()
                                action_result = policy.forward_inference({"obs": obs_batch})
                                action_time = time.perf_counter() - action_start_time
                            
                            # Extract the action from the result - we know it's action_dist_inputs
                            if isinstance(action_result, dict) and 'action_dist_inputs' in action_result:
                                action_logits = action_result['action_dist_inputs']  # Shape: [1, 3]
                                
                                # For discrete actions, we have several options:
                                # Option 1: Deterministic (argmax) - use the most likely action
                                action = action_logits.argmax(dim=-1).item()  # Get scalar action
                                
                                # Option 2: Stochastic sampling (uncomment to use instead)
                                # import torch.nn.functional as F
                                # from torch.distributions import Categorical
                                # dist = Categorical(logits=action_logits)
                                # action = dist.sample().item()
                                
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
    
    
def get_checkpoint_paths():
    """
    Define checkpoint paths for each case with auto-detection.
    """
    # Base directory for checkpoints (adjust for your server)
    checkpoint_base = "checkpoints" 
    
    print("Auto-detecting checkpoint paths...")
    import glob
    
    auto_paths = {}
    for case_name in ["case1", "case2", "case3", "case4"]:
        # Look for training directories for this case
        patterns = [
            f"{checkpoint_base}/checkpoints_{case_name}_training_*/everyone/best_{case_name}_seed*_sim_everyone.ckpt",
            f"{checkpoint_base}/checkpoints_{case_name}_training_*/**/best_{case_name}_seed*_sim_everyone.ckpt",
            f"./checkpoints_{case_name}_training_*/everyone/best_{case_name}_seed*_sim_everyone.ckpt",
            f"./checkpoints_{case_name}_training_*/**/best_{case_name}_seed*_sim_everyone.ckpt",
            f"checkpoints_{case_name}_training_*/everyone/best_{case_name}_seed*_sim_everyone.ckpt",
            f"checkpoints_{case_name}_training_*/**/best_{case_name}_seed*_sim_everyone.ckpt",
        ]
        
        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern, recursive=True))
        
        if matches:
            # Use the most recent one (largest job ID)
            auto_paths[case_name] = sorted(matches)[-1]
            print(f"  Found {case_name}: {auto_paths[case_name]}")
        else:
            print(f"  ⚠️  No checkpoint found for {case_name}")
            print(f"    Searched patterns: {patterns}")
    
    # Verify paths exist
    verified_paths = {}
    for case_name, path in auto_paths.items():
        if os.path.exists(path):
            verified_paths[case_name] = path
            print(f"✓ Verified {case_name}: {path}")
        else:
            print(f"✗ Missing {case_name}: {path}")
    
    return verified_paths

def run_benchmark_suite(config_sets, num_episodes=5, max_steps_per_episode=None):
    """Run benchmark across all configurations and policies"""
    
    print("="*80)
    print("DEBUGGING CHECKPOINT DETECTION")
    print("="*80)
    
    # Print current directory and contents
    import os
    print(f"Current working directory: {os.getcwd()}")
    print("Contents of current directory:")
    for item in os.listdir('.'):
        if 'checkpoint' in item.lower() or 'case' in item.lower():
            print(f"  📁 {item}")
    
    # Test checkpoint detection
    checkpoint_paths = get_checkpoint_paths()
    
    if not checkpoint_paths:
        print("❌ No checkpoint paths found! Please check:")
        print("1. Are you in the right directory?")
        print("2. Do checkpoint directories exist?")
        print("3. Are checkpoint files named correctly?")
        return [], None
    
    # Continue with rest of function...
    system_info = get_system_info()
    test_config = {"num_observers": 20, "num_targets": 100, "time_step": 1, "duration": 100, "seed": 47, "reward_type": "case1", "simulator_type": "everyone"}
    test_benchmark = PolicyBenchmark(test_config, system_info)
    
    working_policies = 0
    for case_name, checkpoint_path in checkpoint_paths.items():
        if os.path.exists(checkpoint_path):
            rl_module = test_benchmark.load_rl_module(checkpoint_path, case_name)
            if rl_module is not None:
                working_policies += 1
    
    if working_policies == 0:
        print("❌ No RL policies could be loaded - stopping benchmark")
        return [], None
    
    print("="*80)
    print("STARTING COMPREHENSIVE BENCHMARK")
    print("="*80)
    print(f"Testing trained policies (from 'everyone' simulator) across all simulator types")
    print(f"Configurations: {list(config_sets.keys())}")
    print(f"Episodes per config: {num_episodes}")
    print("="*80)
    
    # Get system info
    system_info = get_system_info()
    print(f"System: {system_info['cpu_cores']} cores @ {system_info['cpu_freq']:.1f} GHz")
    
    # Get checkpoint paths
    checkpoint_paths = get_checkpoint_paths()
    
    # Simulator types to test across
    simulator_types = ["everyone", "centralized", "decentralized"]
    
    all_results = []
    
    for config_name, base_config in config_sets.items():
        print(f"\n{'='*60}")
        print(f"CONFIGURATION: {config_name}")
        print(f"Agents: {base_config['num_observers']}, Targets: {base_config['num_targets']}")
        print(f"{'='*60}")
        
        for sim_type in simulator_types:
            print(f"\n--- Testing on {sim_type.upper()} simulator ---")
            
            # Create environment config for this simulator type
            env_config = base_config.copy()
            env_config["simulator_type"] = sim_type
            
            # Initialize benchmark for this configuration
            benchmark = PolicyBenchmark(env_config, system_info)
            
            # Test baseline policies first (these don't have loading issues)
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
                        result["config_name"] = config_name
                        result["simulator_type"] = sim_type
                        result["training_simulator"] = "everyone"
                        result["test_simulator"] = sim_type
                        result["episode_number"] = episode + 1
                        result["base_config"] = base_config
                        
                        all_results.append(result)
                        
                        print(f"  Episode {episode + 1}/{num_episodes}:")
                        print(f"    NET per agent: {result['metrics']['net_per_agent']:.6f}")
                        print(f"    Mission completion: {result['metrics']['mission_percentage']:.1f}%")
                        print(f"    Avg resources left: {result['metrics']['average_resources_left']:.3f}")
                        
                    except Exception as e:
                        print(f"✗ Episode {episode + 1} failed: {e}")
                        continue
            
            # Test RL policies one by one
            for case_name, checkpoint_path in checkpoint_paths.items():
                if not os.path.exists(checkpoint_path):
                    print(f"⚠️  Checkpoint not found for {case_name}: {checkpoint_path}")
                    continue
                
                print(f"\n✓ Running Case{case_name[-1]}...")
                
                # Load RL module for this case
                rl_module = benchmark.load_rl_module(checkpoint_path, case_name)
                if rl_module is None:
                    continue
                
                policy_name = f"Case{case_name[-1]}"
                
                for episode in range(num_episodes):
                    try:
                        result = benchmark.run_episode(
                            rl_module, 
                            policy_name, 
                            max_steps=max_steps_per_episode
                        )
                        
                        # Add metadata to result
                        result["config_name"] = config_name
                        result["simulator_type"] = sim_type
                        result["training_simulator"] = "everyone"
                        result["test_simulator"] = sim_type
                        result["episode_number"] = episode + 1
                        result["base_config"] = base_config
                        
                        all_results.append(result)
                        
                        print(f"  Episode {episode + 1}/{num_episodes}:")
                        print(f"    NET per agent: {result['metrics']['net_per_agent']:.6f}")
                        print(f"    Mission completion: {result['metrics']['mission_percentage']:.1f}%")
                        print(f"    Avg resources left: {result['metrics']['average_resources_left']:.3f}")
                        
                    except Exception as e:
                        print(f"✗ Episode {episode + 1} failed: {e}")
                        continue
    
    # Create experiment folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiments/benchmark_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results in the experiment folder
    results_filename = experiment_dir / "benchmark_results.json"
    
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        "timestamp": timestamp,
        "configurations": list(config_sets.keys()),
        "simulator_types": simulator_types,
        "episodes_per_config": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "total_episodes": len(all_results),
        "system_info": system_info,
        "checkpoint_paths": checkpoint_paths
    }
    
    metadata_filename = experiment_dir / "experiment_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE!")
    print(f"Experiment folder: {experiment_dir}")
    print(f"Results saved to: {results_filename}")
    print(f"Metadata saved to: {metadata_filename}")
    print(f"Total configurations: {len(config_sets)} x {len(simulator_types)} = {len(config_sets) * len(simulator_types)}")
    print(f"Total episodes: {len(all_results)}")
    print(f"")
    print(f"This benchmark tests how policies trained on 'everyone' simulator")
    print(f"generalize to 'centralized' and 'decentralized' configurations.")
    print(f"")
    print(f"To analyze results, run:")
    print(f"    python analyze_results.py {results_filename}")
    print(f"{'='*80}")
    
    return all_results, experiment_dir
def main():
    parser = argparse.ArgumentParser(description="Benchmark RL policies against baselines across simulator types")
    parser.add_argument("--configs", type=str, default="small", 
                       choices=["small", "standard", "large", "all"],
                       help="Configuration set to run")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes per configuration")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum steps per episode (for testing)")
    
    args = parser.parse_args()
    
    # Initialize Ray in local mode to keep things simple
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
    
    # Define configuration sets
    if args.configs == "small":
        config_sets = {
            # Quick testing with shorter episodes across all simulator types
            "20_agents_100_targets_centralized": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 500,  # Shorter for testing
                "seed": 47, "reward_type": "case1",
                "simulator_type": "centralized"
            },
            "20_agents_100_targets_decentralized": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 500,  # Shorter for testing
                "seed": 47, "reward_type": "case1",
                "simulator_type": "decentralized"
            },
            "20_agents_100_targets_everyone": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 500,  # Shorter for testing
                "seed": 47, "reward_type": "case1",
                "simulator_type": "everyone"
            }
        }
    elif args.configs == "large":
        config_sets = {
            # Extended duration testing across all simulator types
            "20_agents_100_targets_centralized": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 172800,  # 2 days
                "seed": 47, "reward_type": "case1",
                "simulator_type": "centralized"
            },
            "20_agents_100_targets_decentralized": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 172800,  # 2 days
                "seed": 47, "reward_type": "case1",
                "simulator_type": "decentralized"
            },
            "20_agents_100_targets_everyone": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 172800,  # 2 days
                "seed": 47, "reward_type": "case1",
                "simulator_type": "everyone"
            }
        }
    else:  # standard
        config_sets = {
            # Standard cross-network evaluation - matches training config exactly
            "20_agents_100_targets_centralized": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 86400,
                "seed": 47, "reward_type": "case1",
                "simulator_type": "centralized"
            },
            "20_agents_100_targets_decentralized": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 86400,
                "seed": 47, "reward_type": "case1",
                "simulator_type": "decentralized"
            },
            "20_agents_100_targets_everyone": {
                "num_observers": 20, "num_targets": 100,
                "time_step": 1, "duration": 86400,
                "seed": 47, "reward_type": "case1",
                "simulator_type": "everyone"
            }
        }
    
    # Run benchmark
    try:
        results, experiment_dir = run_benchmark_suite(
            config_sets, 
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps
        )
        if experiment_dir:  # Check if experiment_dir is not None
            print(f"📁 Experiment saved in: {experiment_dir}")
            print(f"📊 To analyze results, run:")
            print(f"    python analyze_results.py {experiment_dir / 'benchmark_results.json'}")
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
