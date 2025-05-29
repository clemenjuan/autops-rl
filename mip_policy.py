import numpy as np
from scipy.optimize import linprog
import warnings


class MIPPolicy:
    """
    Mixed Integer Programming policy for satellite task scheduling.
    This is a simplified implementation that uses linear programming
    relaxation as a heuristic for the actual MIP problem.
    """
    
    def __init__(self, time_horizon=5):
        self.time_horizon = time_horizon
    
    def compute_actions(self, observations, env):
        """
        Compute actions using a simplified MIP formulation.
        
        For now, this implements a greedy heuristic that tries to
        maximize immediate utility while respecting resource constraints.
        """
        actions = {}
        
        # For each agent, solve a simple optimization problem
        for agent_id, obs in observations.items():
            if agent_id not in env.agents:
                continue
                
            action = self._solve_agent_mip(agent_id, obs, env)
            actions[agent_id] = action
        
        return actions
    
    def _solve_agent_mip(self, agent_id, obs, env):
        """
        Solve a simplified MIP for a single agent.
        
        Variables:
        - x_0: binary variable for "do nothing"
        - x_1: binary variable for "communicate"
        - x_2: binary variable for "observe"
        
        Objective: maximize expected utility
        Constraints: 
        - exactly one action selected
        - resource constraints (battery, storage)
        """
        
        agent_idx = env.agent_name_mapping[agent_id]
        
        # Extract agent state
        battery_level = obs["battery"][agent_idx]
        storage_level = obs["storage"][agent_idx]
        availability = obs["availability"][0]
        
        # Calculate utilities for each action
        utility_idle = 0.1  # Small positive utility for conserving resources
        utility_communicate = self._calculate_communication_utility(agent_idx, obs)
        utility_observe = self._calculate_observation_utility(agent_idx, obs)
        
        # Apply resource constraints
        if battery_level < 0.2:  # Low battery constraint
            utility_communicate *= 0.1
            utility_observe *= 0.1
        
        if storage_level > 0.9:  # High storage constraint
            utility_observe *= 0.1  # Discourage more observation
            utility_communicate *= 2.0  # Encourage communication
        
        if availability == 0:  # Not available
            utility_observe = 0
            utility_communicate = 0
        
        # Select action with highest utility
        utilities = [utility_idle, utility_communicate, utility_observe]
        action = np.argmax(utilities)
        
        return action
    
    def _calculate_observation_utility(self, agent_idx, obs):
        """Calculate utility for observation action"""
        observation_status = obs["observation_status"]
        pointing_accuracy = obs["pointing_accuracy"][agent_idx]
        
        # Utility is higher for unobserved targets with good pointing accuracy
        unobserved_mask = observation_status < 3
        accuracy_weighted_targets = pointing_accuracy * unobserved_mask
        
        # Sum of potential utility from all observable targets
        utility = np.sum(accuracy_weighted_targets)
        
        return utility
    
    def _calculate_communication_utility(self, agent_idx, obs):
        """Calculate utility for communication action"""
        communication_ability = obs["communication_ability"]
        storage_level = obs["storage"][agent_idx]
        
        # Utility is higher when:
        # 1. Can actually communicate with other agents
        # 2. Have data to share (high storage)
        if np.any(communication_ability == 1):
            utility = storage_level * 2.0  # Weight by amount of data to share
        else:
            utility = 0.0  # Can't communicate
        
        return utility
    
    def get_policy_info(self):
        """Return information about the policy configuration"""
        return {
            "policy_type": "mip",
            "time_horizon": self.time_horizon,
            "note": "Simplified MIP using greedy heuristic"
        } 