import numpy as np

class RewardFunction:
    """Base class for all reward functions"""
    def __init__(self, config=None):
        # Default parameters
        self.standby_penalty = 0.01 # Penalty for doing nothing
        self.k = 0.01 # Resource consumption penalty factor
        self.rho = 0.03 # Communication success reward factor
        self.lambda_val = 0.1 # New observation reward factor
        self.mu = 0.01 # Failed communication penalty factor
        self.observation_reward = 0.1 # Reward for successful observation
        self.mission_complete_bonus = 0 # Bonus for completing mission
        self.targets_bonus_factor = 0.01 # Bonus factor for remaining/observed targets
        self.final_targets_bonus = 1.0 # Bonus factor given at the end of the mission (cases 3 and 4)
        self.depletion_penalty = 0
        
        # Add global scaling factor
        self.reward_scale = 0.01  # Scale all rewards to 1/100
        
        # Apply configuration if provided
        if config:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def resource_penalty(self, observer):
        """Calculate penalty for resource consumption"""
        battery_penalty = self.k * (1 - observer.epsys['EnergyAvailable'] / observer.epsys['EnergyStorage'])
        storage_penalty = self.k * (1 - observer.DataHand['StorageAvailable'] / observer.DataHand['DataStorage'])
        return -(battery_penalty + storage_penalty)
    
    def check_depletion(self, observer):
        """Check if resources are depleted and return appropriate penalty"""
        if observer.epsys['EnergyAvailable'] < 0 or observer.DataHand['StorageAvailable'] < 0:
            return -self.depletion_penalty
        return 0.0
    
    def failed_action_penalty(self):
        """Penalty for a failed action"""
        return -self.mu
    
    def successful_communication_reward(self, result_info):
        """Reward for successful communication"""
        return self.rho * result_info.get("data_transmitted", 0)
    
    def new_observation_reward(self, pointing_accuracy):
        """Reward for new observation, scaled by pointing accuracy"""
        return self.lambda_val * pointing_accuracy
    
    def targets_bonus(self, targets, total_targets):
        """Bonus reward for number of targets compared to total number of targets"""
        return self.targets_bonus_factor * targets / total_targets
    
    def successful_observation_reward(self):
        """Reward for completing an observation"""
        return self.observation_reward
    
    def calculate_reward(self, action_type, observer, result_info):
        """Calculate reward based on action type and result"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def mission_completion_bonus(self, time_remaining, total_duration):
        """Bonus reward for completing the mission with time remaining"""
        return self.mission_complete_bonus * time_remaining / total_duration


class Case1RewardFunction(RewardFunction):
    """
    Case 1: Positive rewards for observed targets
    """
    def calculate_reward(self, action_type, observer, result_info):
        reward = self.resource_penalty(observer)
        
        if action_type == "standby":
            # Small penalty for doing nothing
            reward -= self.standby_penalty
        
        elif action_type == "communication":
            if result_info.get("successful", False):
                reward += self.successful_communication_reward(result_info)
            else:
                reward += self.failed_action_penalty()
        
        elif action_type == "observation":
            if result_info.get("successful", False):
                reward += self.successful_observation_reward()
                reward += self.new_observation_reward(result_info.get("pointing_accuracy"))
            else:
                reward += self.failed_action_penalty()
        
        # Add positive reward for each acknowledged target
        total_targets = len(observer.observation_status_matrix)
        acknowledged_targets = np.sum(observer.observation_status_matrix > 0)
        reward += self.targets_bonus(acknowledged_targets, total_targets)
        
        # Apply global scaling
        return reward * self.reward_scale


class Case2RewardFunction(RewardFunction):
    """
    Case 2: Negative rewards for unobserved targets
    """
    def calculate_reward(self, action_type, observer, result_info):
        reward = self.resource_penalty(observer)
        
        if action_type == "standby":
            # Small penalty for doing nothing
            reward -= self.standby_penalty
        
        elif action_type == "communication":
            if result_info.get("successful", False):
                reward += self.successful_communication_reward(result_info)
            else:
                reward += self.failed_action_penalty()
        
        elif action_type == "observation":
            if result_info.get("successful", False):
                reward += self.successful_observation_reward()
                reward += self.new_observation_reward(result_info.get("pointing_accuracy"))
            else:
                reward += self.failed_action_penalty()
        
        # Add negative reward for each unacknowledged target
        total_targets = len(observer.observation_status_matrix)
        unacknowledged_targets = total_targets - np.sum(observer.observation_status_matrix > 0)
        reward -= self.targets_bonus(unacknowledged_targets, total_targets)
        
        # Apply global scaling
        return reward * self.reward_scale


class Case3RewardFunction(RewardFunction):
    """
    Case 3: Individual rewards with global bonus for observed targets
    """
    def calculate_reward(self, action_type, observer, result_info):
        reward = self.resource_penalty(observer)
        
        if action_type == "standby":
            # Small penalty for doing nothing
            reward -= self.standby_penalty
        
        elif action_type == "communication":
            if result_info.get("successful", False):
                reward += self.successful_communication_reward(result_info)
            else:
                reward += self.failed_action_penalty()
        
        elif action_type == "observation":
            if result_info.get("successful", False):
                reward += self.successful_observation_reward()
                reward += self.new_observation_reward(result_info.get("pointing_accuracy"))
            else:
                reward += self.failed_action_penalty()
        
        # Apply global scaling
        return reward * self.reward_scale
    
    def calculate_global_bonus(self, global_observation_status):
        """Calculate global bonus based on total observed targets"""
        # Count how many targets have been observed by at least one agent
        observed_targets = np.sum(np.any(global_observation_status == 3, axis=0))
        return self.final_targets_bonus * observed_targets  # Adjust multiplier as needed


class Case4RewardFunction(RewardFunction):
    """
    Case 4: Individual rewards with global penalty for unobserved targets
    """
    def calculate_reward(self, action_type, observer, result_info):
        reward = self.resource_penalty(observer)
        
        if action_type == "standby":
            # Small penalty for doing nothing
            reward -= self.standby_penalty
        
        elif action_type == "communication":
            if result_info.get("successful", False):
                reward += self.successful_communication_reward(result_info)
            else:
                reward += self.failed_action_penalty()
        
        elif action_type == "observation":
            if result_info.get("successful", False):
                reward += self.successful_observation_reward()
                reward += self.new_observation_reward(result_info.get("pointing_accuracy"))
            else:
                reward += self.failed_action_penalty()
        
        # Apply global scaling
        return reward * self.reward_scale
    
    def calculate_global_penalty(self, global_observation_status, num_targets):
        """Calculate global penalty based on unobserved targets"""
        # Count how many targets have not been observed by any agent
        observed_targets = np.sum(np.any(global_observation_status == 3, axis=0))
        unobserved_targets = num_targets - observed_targets
        return -self.final_targets_bonus * unobserved_targets  # Adjust multiplier as needed


def get_reward_function(reward_type, config=None):
    """Factory function to get the appropriate reward function"""
    reward_functions = {
        "case1": Case1RewardFunction,
        "case2": Case2RewardFunction,
        "case3": Case3RewardFunction,
        "case4": Case4RewardFunction
    }
    
    if reward_type not in reward_functions:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    return reward_functions[reward_type](config) 