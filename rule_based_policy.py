import numpy as np


class RuleBasedPolicy:
    """
    Rule-based policy that makes decisions based on battery level and storage level.
    
    Action space:
    0 - Do nothing (idle)
    1 - Communicate with other satellites
    2 - Observe targets
    """
    
    def __init__(self, battery_threshold_high=0.8, battery_threshold_low=0.3,
                 storage_threshold_high=0.7, storage_threshold_low=0.3):
        self.battery_threshold_high = battery_threshold_high
        self.battery_threshold_low = battery_threshold_low
        self.storage_threshold_high = storage_threshold_high
        self.storage_threshold_low = storage_threshold_low
    
    def compute_actions(self, observations, env):
        """
        Compute actions for all agents based on rule-based logic.
        
        Rules:
        1. If battery is low (< low_threshold), do nothing (action 0)
        2. If storage is high (> high_threshold), prioritize communication (action 1)
        3. If battery is high and storage is low, prioritize observation (action 2)
        4. Otherwise, choose based on availability and opportunities
        """
        actions = {}
        
        for agent_id, obs in observations.items():
            if agent_id not in env.agents:
                continue
                
            # Get agent index
            agent_idx = env.agent_name_mapping[agent_id]
            
            # Extract current agent's battery and storage levels
            battery_level = obs["battery"][agent_idx]
            storage_level = obs["storage"][agent_idx]
            availability = obs["availability"][0]
            
            # Rule 1: Low battery - conserve energy
            if battery_level < self.battery_threshold_low:
                actions[agent_id] = 0  # Do nothing
                continue
            
            # Rule 2: High storage - prioritize communication to offload data
            if storage_level > self.storage_threshold_high:
                # Check if communication is possible
                communication_ability = obs["communication_ability"]
                if np.any(communication_ability == 1):  # Can communicate with someone
                    actions[agent_id] = 1  # Communicate
                    continue
            
            # Rule 3: Good battery and low storage - prioritize observation
            if (battery_level > self.battery_threshold_high and 
                storage_level < self.storage_threshold_low and 
                availability == 1):
                
                # Check if there are observable targets
                observation_status = obs["observation_status"]
                pointing_accuracy = obs["pointing_accuracy"][agent_idx]
                
                # Check if there are targets that can be observed with good accuracy
                observable_targets = np.where((observation_status < 3) & (pointing_accuracy > 0.5))[0]
                if len(observable_targets) > 0:
                    actions[agent_id] = 2  # Observe
                    continue
            
            # Rule 4: Balanced approach based on current situation
            if availability == 1 and battery_level > self.battery_threshold_low:
                # Check what's more beneficial: observing or communicating
                
                # Priority for observation if:
                # - There are unobserved targets with good pointing accuracy
                # - Storage is not too high
                observation_status = obs["observation_status"]
                pointing_accuracy = obs["pointing_accuracy"][agent_idx]
                
                unobserved_targets = np.where((observation_status < 3) & (pointing_accuracy > 0.6))[0]
                communication_ability = obs["communication_ability"]
                can_communicate = np.any(communication_ability == 1)
                
                if (len(unobserved_targets) > 0 and storage_level < self.storage_threshold_high):
                    actions[agent_id] = 2  # Observe
                elif can_communicate and storage_level > self.storage_threshold_low:
                    actions[agent_id] = 1  # Communicate
                else:
                    actions[agent_id] = 0  # Do nothing
            else:
                actions[agent_id] = 0  # Do nothing if not available or low battery
        
        return actions
    
    def get_policy_info(self):
        """Return information about the policy configuration"""
        return {
            "policy_type": "rule_based",
            "battery_threshold_high": self.battery_threshold_high,
            "battery_threshold_low": self.battery_threshold_low,
            "storage_threshold_high": self.storage_threshold_high,
            "storage_threshold_low": self.storage_threshold_low
        } 