import psutil
import platform
import time
import numpy as np
from collections import defaultdict


def get_system_info():
    """Get system information for normalization"""
    try:
        # Get CPU information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_ghz = cpu_freq.current / 1000.0 if cpu_freq else 2.5  # Default to 2.5 GHz
        
        # Get memory information
        memory = psutil.virtual_memory()
        
        system_info = {
            "cpu_cores": cpu_count,
            "cpu_freq": cpu_freq_ghz,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
    except Exception as e:
        print(f"Warning: Could not get complete system info: {e}")
        # Fallback values
        system_info = {
            "cpu_cores": 4,
            "cpu_freq": 2.5,
            "memory_total_gb": 8.0,
            "memory_available_gb": 4.0,
            "platform": "unknown",
            "python_version": platform.python_version(),
        }
    
    return system_info


class BenchmarkMetrics:
    def __init__(self, policy_name, num_agents, num_targets, system_info):
        self.policy_name = policy_name
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.system_info = system_info
        
        # Timing metrics
        self.action_times = []
        self.step_times = []
        self.episode_time = 0
        self.step_count = 0
        # Action tracking
        self.action_counts = defaultdict(int)  # Total count of each action
        self.total_actions = defaultdict(int)  # Total actions per agent
        
        # Reward tracking
        self.rewards = []
        
        # Environment metrics (set at end of episode)
        self.env_metrics = {}
        
        self.results = []
    
    def add_action_time(self, time_seconds):
        """Add time taken to compute actions"""
        self.action_times.append(time_seconds)
    
    def add_step_time(self, time_seconds):
        """Add time taken for environment step"""
        self.step_times.append(time_seconds)
    
    def set_episode_time(self, time_seconds):
        """Set total episode time"""
        self.episode_time = time_seconds
    
    def record_action(self, agent_id, action):
        """Record an action taken by an agent"""
        self.action_counts[f"action_{action}"] += 1
        self.total_actions[agent_id] += 1
    
    def add_rewards(self, rewards_dict):
        """Add step rewards"""
        self.rewards.append(sum(rewards_dict.values()))
    
    def set_environment_metrics(self, env_metrics):
        """Set final environment metrics"""
        self.env_metrics = env_metrics
    
    def calculate_net_per_agent(self):
        """Calculate Normalized Execution Time per agent"""
        total_action_time = sum(self.action_times)
        cpu_power = self.system_info["cpu_cores"] * self.system_info["cpu_freq"]
        net_per_agent = (total_action_time / cpu_power) / self.num_agents
        return net_per_agent
    
    def calculate_mission_percentage(self):
        """Calculate mission completion percentage"""
        if "observation_stats" in self.env_metrics:
            obs_stats = self.env_metrics["observation_stats"]
            return obs_stats.get("observation_percentage", 0.0)
        else:
            # Fallback calculation
            if "matrix_stats" in self.env_metrics:
                avg_obs_status = self.env_metrics["matrix_stats"].get("global_observation_status_avg", 0)
                return (avg_obs_status / 3.0) * 100
        return 0.0
    
    def calculate_average_resources(self):
        """Calculate average resources left (battery + storage)"""
        if "resource_stats" in self.env_metrics:
            battery_avg = self.env_metrics["resource_stats"].get("average_battery", 0)
            storage_avg = self.env_metrics["resource_stats"].get("average_storage", 0)
            return (battery_avg + storage_avg) / 2.0
        return 0.0
    
    def calculate_action_distribution(self):
        """Calculate percentage distribution of actions"""
        total_actions = sum(self.action_counts.values())
        action_distribution = {}
        
        for action in [0, 1, 2]:  # Idle, Communicate, Observe
            count = self.action_counts.get(f"action_{action}", 0)
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            action_distribution[f"action_{action}"] = percentage
        
        return action_distribution
    
    def get_results(self):
        """Get all metrics as a dictionary"""
        return {
            "policy_name": self.policy_name,
            "num_agents": self.num_agents,
            "num_targets": self.num_targets,
            "simulator_type": getattr(self, 'simulator_type', 'unknown'),
            "step_count": self.step_count,
            "system_info": self.system_info,
            "timing": {
                "episode_time": self.episode_time,
                "total_action_time": sum(self.action_times),
                "average_action_time": np.mean(self.action_times) if self.action_times else 0,
                "total_step_time": sum(self.step_times),
                "average_step_time": np.mean(self.step_times) if self.step_times else 0,
            },
            "metrics": {
                "net_per_agent": self.calculate_net_per_agent(),
                "mission_percentage": self.calculate_mission_percentage(),
                "average_resources_left": self.calculate_average_resources(),
                "simulation_time": self.episode_time,
                "total_reward": sum(self.rewards),
                "average_reward_per_step": np.mean(self.rewards) if self.rewards else 0,
                "action_distribution": self.calculate_action_distribution(),
            },
            "raw_data": {
                "action_times": self.action_times,
                "step_times": self.step_times,
                "rewards": self.rewards,
                "action_counts": dict(self.action_counts),
                "total_actions": dict(self.total_actions),
                "env_metrics": self.env_metrics,
            }
        }