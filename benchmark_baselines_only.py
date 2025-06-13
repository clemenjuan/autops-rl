import os
import time
import argparse
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import ray
from ray.tune.registry import register_env
from src.envs.FSS_env_v1 import FSS_env
from rule_based_policy import RuleBasedPolicy
from mip_policy import MIPPolicy
from benchmark_utils import BenchmarkMetrics, get_system_info

def env_creator(env_config):
    """Environment creator function for Ray registration"""
    return FSS_env(env_config)

class BaselineBenchmark:
    def __init__(self, env_config, system_info):
        self.env_config = env_config
        self.system_info = system_info
        
        # Register environment with Ray
        register_env("FSS_env", env_creator)
        
        # Initialize environment
        print(f"Initializing environment with config: {env_config}")
        try:
            self.env = FSS_env(env_config)
            print(f"✓ Environment initialized successfully")
        except Exception as e:
            print(f"✗ Environment initialization failed: {e}")
            raise
    
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
            
            # Compute actions
            if policy_name == "RuleBased":
                action_start_time = time.perf_counter()
                actions = policy.compute_actions(obs, self.env)
                action_time = time.perf_counter() - action_start_time
                
            elif policy_name == "MIP":
                action_start_time = time.perf_counter()
                actions = policy.compute_actions(obs, self.env)
                action_time = time.perf_counter() - action_start_time
            
            # Record metrics
            metrics.add_action_time(action_time)
            for agent_id, action in actions.items():
                metrics.record_action(agent_id, action)
            
            # Step environment
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            step_time = time.perf_counter() - step_start_time
            
            metrics.add_step_time(step_time)
            metrics.add_rewards(rewards)
            
            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            step_count += 1
        
        # Finalize metrics
        episode_time = time.perf_counter() - episode_start_time
        metrics.set_episode_time(episode_time)
        
        final_metrics = self.env.collect_comprehensive_metrics()
        metrics.set_environment_metrics(final_metrics)
        
        print(f"✓ Completed episode with {policy_name} in {episode_time:.2f}s ({step_count} steps)")
        
        results = metrics.get_results()
        results['step_count'] = step_count
        
        return results

def run_baseline_benchmark(num_episodes=15, max_steps_per_episode=None):
    """Run benchmark with only baseline policies"""
    
    print("="*80)
    print("STARTING BASELINE POLICIES BENCHMARK")
    print("="*80)
    print(f"Testing RuleBased and MIP policies on ALL coordination types")
    print(f"Episodes per policy per coordination: {num_episodes}")
    print("="*80)
    
    # Get system info
    system_info = get_system_info()
    print(f"System: {system_info['cpu_cores']} cores @ {system_info['cpu_freq']:.1f} GHz")
    
    # Base environment configuration
    base_env_config = {
        "num_observers": 20, 
        "num_targets": 100,
        "time_step": 1, 
        "duration": 86400,
        "seed": 47, 
        "reward_type": "case1",  # Doesn't matter for baselines
    }
    
    all_results = []
    
    # Test all coordination types
    simulator_types = ["centralized", "decentralized", "everyone"]
    simulator_names = {
        "centralized": "Centralized coordination",
        "decentralized": "Constrained decentralized coordination", 
        "everyone": "Fully decentralized coordination"
    }
    
    # Test baseline policies
    baseline_policies = [
        ("RuleBased", RuleBasedPolicy()),
        ("MIP", MIPPolicy()),
    ]
    
    for sim_type in simulator_types:
        print(f"\n{'='*80}")
        print(f"TESTING ON {simulator_names[sim_type].upper()}")
        print(f"{'='*80}")
        
        # Create environment config for this simulator type
        env_config = base_env_config.copy()
        env_config["simulator_type"] = sim_type
        
        # Initialize benchmark for this coordination type
        benchmark = BaselineBenchmark(env_config, system_info)
        
        for policy_name, policy in baseline_policies:
            print(f"\n{'-'*60}")
            print(f"TESTING {policy_name.upper()} on {sim_type}")
            print(f"{'-'*60}")
            
            for episode in range(num_episodes):
                try:
                    result = benchmark.run_episode(
                        policy, 
                        policy_name, 
                        max_steps=max_steps_per_episode
                    )
                    
                    # Add metadata
                    result["policy_type"] = "baseline"
                    result["policy_name"] = policy_name
                    result["simulator_type"] = sim_type
                    result["episode_number"] = episode + 1
                    
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
    experiment_dir = Path(f"experiments/baseline_benchmark_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_filename = experiment_dir / "baseline_results.json"
    
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save experiment metadata
    metadata = {
        "timestamp": timestamp,
        "policies_tested": ["RuleBased", "MIP"],
        "simulator_types_tested": simulator_types,
        "episodes_per_policy_per_simulator": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "total_episodes": len(all_results),
        "system_info": system_info,
        "base_env_config": base_env_config
    }
    
    metadata_filename = experiment_dir / "experiment_metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BASELINE BENCHMARK COMPLETE!")
    print(f"Experiment folder: {experiment_dir}")
    print(f"Results saved to: {results_filename}")
    print(f"Total episodes: {len(all_results)}")
    print(f"Coordination types tested: {', '.join(simulator_types)}")
    print(f"{'='*80}")
    
    return all_results, experiment_dir

def main():
    parser = argparse.ArgumentParser(description="Baseline policies benchmark (RuleBased + MIP only)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes per policy")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum steps per episode (for testing)")
    
    args = parser.parse_args()
    
    # Initialize Ray in local mode
    try:
        print("Initializing Ray in local mode...")
        ray.init(
            local_mode=True,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        
        # Register the environment with Ray
        register_env("FSS_env", env_creator)
        print("✓ Ray initialized successfully")
        
    except Exception as e:
        print(f"Warning: Ray initialization failed: {e}")
        print("Continuing without Ray")
    
    # Run baseline benchmark
    try:
        results, experiment_dir = run_baseline_benchmark(
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps
        )
        if experiment_dir:
            print(f"📁 Experiment saved in: {experiment_dir}")
        else:
            print("❌ No experiment directory created - benchmark failed")
        
    except Exception as e:
        print(f"❌ Baseline benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            ray.shutdown()
        except:
            pass

if __name__ == "__main__":
    main() 