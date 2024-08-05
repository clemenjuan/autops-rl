import os
from copy import copy
import time

import torch
from torch import nn

from ray import train, tune
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from simulator import Simulator, CentralizedSimulator, MixedSimulator, DecentralizedSimulator
from plotting import plot
from data_logging import log_full_matrices, log_summary_results, compute_statistics_from_npy
class FSS_env(MultiAgentEnv):
    metadata = {
        "name": "FSS_env-v0",
    }
    def __init__(self, num_targets: int, 
                 num_observers: int, 
                 simulator_type: str = 'everyone', 
                 time_step: float = 1, 
                 duration: float = 24*60*60):
        # Environment setup
        super().__init__()
        self.num_targets = num_targets
        self.num_observers = num_observers
        self.num_satellites = num_targets + num_observers
        self.time_step = time_step
        self.duration = duration
        self.sim_time = 0
        self.latest_step_duration = 0
        self.simulator_type = simulator_type
        assert self.num_observers > 0
        assert self.num_targets > 0
        self.special_events_count = 0

        self.possible_agents = ["observer_" + str(r) for r in range(num_observers)]
        self._agent_ids = set(self.possible_agents)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        ) # Mapping of agent names to indices

        self.agents = copy(self.possible_agents)
        self.orbital_params_order = ['semimajoraxis', 'inclination', 'eccentricity',
                    'raan', 'arg_of_perigee', 'true_anomaly', 'mean_anomaly',
                    'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        self.orbital_params_order_targets = ['x', 'y', 'z', 'vx', 'vy', 'vz']

        # Define the observation and action spaces for each agent
        self._observation_spaces = spaces.Dict({
            "observer_satellites": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, len(self.orbital_params_order))),
            "band": spaces.Box(low=1, high=5, shape=(1,), dtype=np.int8),
            "target_satellites": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, len(self.orbital_params_order_targets))),
            "availability": spaces.MultiBinary(1),
            "battery": spaces.Box(low=0, high=1, shape=(self.num_observers,)),
            "storage": spaces.Box(low=0, high=1, shape=(self.num_observers,)),
            "observation_status": spaces.Box(low=0, high=3, shape=(self.num_targets,)),
            "pointing_accuracy": spaces.Box(low=-0, high=1, shape=(self.num_observers, self.num_targets)),
            "communication_status": spaces.Box(low=0, high=1, shape=(self.num_observers,), dtype=np.int8),
            "communication_ability": spaces.MultiBinary(self.num_observers)
            })
        self._action_spaces = spaces.Discrete(2 + self.num_targets)
        self.observation_spaces = {agent: self._observation_spaces for agent in self.agents}
        self.observation_space = self._observation_spaces
        self.action_spaces = {agent: self._action_spaces for agent in self.agents}
        self.action_space = self._action_spaces
        
        self.infos = {agent: {} for agent in self.possible_agents}
        

    
    def get_agent_ids(self):
        return self._agent_ids

    
    def observation_space_contains(self, obs):
        """Check if the observation is valid within the observation space"""
        return all(self.observation_spaces[agent].contains(obs[agent]) for agent in self.agents)

    def action_space_contains(self, act):
        """Check if the action is valid within the action space"""
        return all(self.action_spaces[agent].contains(act[agent]) for agent in self.agents)

    def action_space_sample(self, agent_ids=None):
        """Generate a sample action for each agent based on their action spaces.
        If agent_ids is provided, only sample actions for those agents.
        """
        if agent_ids is None:
            agent_ids = self.agents  # Default to all agents if none are specified.
        sample = {agent: self.action_spaces[agent].sample() for agent in self.agents}
        print(f"Sampled action: {sample}")
        return sample

    def observation_space_sample(self):
        """Generate a sample observation for each agent based on their observation spaces"""
        sample = {agent: self.observation_spaces[agent].sample() for agent in self.agents}
        print(f"Sampled observation: {sample}")
        return sample

    def reset(self, seed=None, options=None): 
        # here 13173302.10772834; 1; 19200.0 are printed idk why

        self.special_events_count = 0
        # print("Resetting...")
        self.agents = copy(self.possible_agents)
        self._agent_ids = set(self.possible_agents)

        # Initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        else:
            raise ValueError("Invalid simulator type. Choose from 'centralized', 'decentralized', or 'everyone'.")    

        observations = {
            agent: self._generate_observation(agent)
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        self.simulator.time_step_number = 0

        # print("Reset done")
        # print(f"Reset returning observations for agents: {observations.keys()}, infos for agents: {infos.keys()}")
        return observations, infos

    def observe(self, agent):
        return self._generate_observation(agent)

    def step(self, actions):
        # print(f"Step starting...")
        assert self.agents, "Cannot step an environment with no agents"
        assert set(actions.keys()) == self._agent_ids, "Actions must be provided for all agents"

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # Create zero actions
        zero_actions = {agent: 0 for agent in self.agents}
        
        # Initialize rewards, terminations, and truncations for each agent
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        terminations["__all__"] = False
        truncations["__all__"] = False

        special_event_detected = False

        while not special_event_detected and self.simulator.time_step_number < (self.duration / self.time_step):
            # Detect special events without processing actions
            special_event_detected = self.detect_special_event()
            if not special_event_detected:
                # Step the simulator with zero actions
                rewards, done = self.simulator.step(zero_actions, self.simulator_type, self.agents)
                if done:
                    break
            else:
                # print(f"Environment special event detected at step {self.simulator.time_step_number}")
                break

        if special_event_detected:
            # Step the simulator with actual actions
            rewards, done = self.simulator.step(actions, self.simulator_type, self.agents)

        observations = {
            agent: self._generate_observation(agent)
            for agent in self.agents
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if self.simulator.time_step_number%10000 == 0:
            print(f"Step {self.simulator.time_step_number} done")

        if done:
            for agent in self.agents:
                terminations[agent] = True
                truncations[agent] = True
            terminations["__all__"] = True
            truncations["__all__"] = True
            self.agents = []
            print(f"Special events detected: {self.special_events_count}")
            print(f"Forced termination at step {self.simulator.time_step_number}")
                
        # print(f"Observations: {observations}")

        # print("Step done")
        # print(f"Step {self.simulator.time_step_number} reward: {rewards}")
        # print(f"Step returning observations for agents: {observations.keys()}, rewards for agents: {rewards.keys()}, terminations for agents: {terminations.keys()}, truncations for agents: {truncations.keys()}, infos for agents: {infos.keys()}")
        return observations, rewards, terminations, truncations, infos
    
    def detect_special_event(self):
        can_observe = self.simulator.get_global_targets(self.simulator.observer_satellites, self.simulator.target_satellites)
        can_communicate = self.simulator.get_global_communication_ability(self.simulator.observer_satellites,self.simulator.time_step, self.simulator_type)

        if can_observe or can_communicate:
            self.special_events_count += 1
            return True

        return False

    def render(self):
        pass
    
    def _generate_observation(self, agent):
        """
        Generates observation for one agent
        Now all matrixes are synchronized for each timestep, so observation will depend on them
        Each observer has its own states as observation (at least)
        """
        # Generate observations based on the current state and communication history
        # Update observer's own state in the observation vector
        agent_idx = self.agent_name_mapping[agent]
        orbital_params_order = ['semimajoraxis', 'inclination', 'eccentricity',
                    'raan', 'arg_of_perigee', 'true_anomaly', 'mean_anomaly',
                    'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        orbital_params_order_targets = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        observer = self.simulator.observer_satellites[agent_idx]
        # Start by initializing the observation dictionary with proper shapes and default values
        observation = {key: np.zeros(shape=space.shape, dtype=space.dtype) 
                   for key, space in self._observation_spaces.spaces.items()}
        
        i = agent_idx
        observer_orbit_params = np.array([observer.orbit[param] for param in orbital_params_order], dtype=np.float32)
        observation['observer_satellites'][i] = observer_orbit_params
        observation['band'] = np.array([observer.commsys['band']], dtype=np.int8)
        observation['availability'] = np.array([observer.availability], dtype=np.int8)
        observation['battery'][i] = np.array([observer.epsys['EnergyAvailable'] / observer.epsys['EnergyStorage']], dtype=np.float32)
        observation['storage'][i] = np.array([observer.DataHand['StorageAvailable'] / observer.DataHand['DataStorage']], dtype=np.float32)
        observation['observation_status'] = np.array(observer.observation_status_matrix, dtype=np.int8)
        observation['pointing_accuracy'][i] = np.array(observer.pointing_accuracy_matrix, dtype=np.float32)
        observation['communication_status'] = np.array(self.simulator.adjacency_matrix_acc[i], dtype=np.int8)
        observation['communication_ability'] = np.array(observer.get_communication_ability(self.simulator.observer_satellites, self.simulator.time_step, self.simulator_type), dtype=np.int8)

        for j, other_observer in enumerate(self.simulator.observer_satellites):
            if i != j:
                if self.simulator.adjacency_matrix_acc[i, j] == 1:
                    other_observer_orbit_params = np.array([other_observer.orbit[param] for param in orbital_params_order], dtype=np.float32)
                    observation['observer_satellites'][j] = other_observer_orbit_params
                if self.simulator.adjacency_matrix[i, j] == 1:
                    observation['battery'][j] = np.array(other_observer.epsys['EnergyAvailable'] / other_observer.epsys['EnergyStorage'], dtype=np.float32)
                    observation['storage'][j] = np.array(other_observer.DataHand['StorageAvailable'] / other_observer.DataHand['DataStorage'], dtype=np.float32)
                    # observation['communication_ability'][j] = np.array(other_observer.get_communication_ability(self.simulator.observer_satellites, self.simulator.time_step, self.simulator_type), dtype=np.int8)
                    observation['pointing_accuracy'][j] = np.array(other_observer.pointing_accuracy_matrix, dtype=np.float32)

        for k, target in enumerate(self.simulator.target_satellites):
            if self.simulator.contacts_matrix[i, k] == 1:
                target_orbit_params = np.array([target.orbit[param] for param in orbital_params_order_targets], dtype=np.float32)
                observation['target_satellites'][k] = target_orbit_params

        # Ensure all values are within the defined bounds
        observation['observer_satellites'] = np.array(observation['observer_satellites'], dtype=np.float32)
        observation['band'] = np.array(observation['band'], dtype=np.int8)
        observation['availability'] = np.array(observation['availability'], dtype=np.int8)
        observation['battery'] = np.clip(np.array(observation['battery'], dtype=np.float32), 0, 1)  # Also ensures values are within bounds
        observation['storage'] = np.clip(np.array(observation['storage'], dtype=np.float32), 0, 1)  # Also ensures values are within bounds
        observation['communication_ability'] = np.array(observation['communication_ability'], dtype=np.int8)
        observation['communication_status'] = np.array(observation['communication_status'], dtype=np.int8)
        observation['observation_status'] = np.clip(np.array(observation['observation_status'], dtype=np.float32), 0, 3)  # Also ensures values are within bounds
        observation['pointing_accuracy'] = np.clip(np.array(observation['pointing_accuracy'], dtype=np.float32), 0, 1)  # Also ensures values are within bounds
        observation['target_satellites'] = np.array(observation['target_satellites'], dtype=np.float32)

        
        # print(f"Observation for {agent}: {observation}")
        # Ensure everything matches the expected types and shapes
        for key, value in observation.items():
            # print(f"Observation key: {key}, shape: {value.shape}")
            assert observation[key].dtype == self._observation_spaces[key].dtype, f"Type mismatch for {key}: expected {self._observation_spaces[key].dtype}, got {observation[key].dtype}"
            assert observation[key].shape == self._observation_spaces[key].shape, f"Shape mismatch for {key}: expected {self._observation_spaces[key].shape}, got {observation[key].shape}"
            if not self._observation_spaces[key].contains(observation[key]):
                raise ValueError(f"{key} does not match the expected space")
            
        # Use np.clip to ensure all values are within the defined bounds for continuous spaces
        for key in observation:
            if isinstance(self._observation_spaces.spaces[key], spaces.Box):
                observation[key] = np.clip(
                    observation[key],
                    self._observation_spaces.spaces[key].low,
                    self._observation_spaces.spaces[key].high
                )
        # print(f"Generated observation for {agent}: {observation}")
        return observation
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
    
if __name__ == "__main__":
    ### Example of how to use the environment for a Monte Carlo simulation
    ############################ EDIT HERE ############################
    num_simulations = 1000  # desired number of simulations
    num_targets = 20 # Number of target satellites
    num_observers = 20 # Number of observer satellites
    simulator_type = 'everyone' # choose from 'centralized', 'decentralized', or 'everyone'
    time_step = 1 # Time step in seconds
    duration = 24*60*60 # Duration of the simulation in seconds
    steps_batch_size = 1000 # Number of steps before printing new information

    # Define the folder name
    results_folder = os.path.join("Results", "MonteCarlo") # v0, MonteCarlo, PPO, DQN, etc.
    results_folder_plots = os.path.join(results_folder, "plots")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(results_folder_plots, exist_ok=True)

    write_to_csv_file_flag = True  # Write the output to a file
    plot_flag = True  # Plot the output
    # Define the relevant attributes to be logged (from env.simulator)
    # add in matrices that you want to log
    relevant_attributes = [
        'adjacency_matrix', 'data_matrix', 'contacts_matrix',
        'global_observation_counts', 'max_pointing_accuracy_avg',
        'global_observation_status_matrix', 'batteries', 'storage', 'total_reward', 
        'total_duration', 'total_steps',
        ]
    ####################################################################

    env = FSS_env(num_targets, num_observers, simulator_type, time_step, duration)

    for i in range(num_simulations):
        print()
        print(f"######## Starting simulation {i+1} ########")
        # Run the simulation until timeout or agent failure
        env = FSS_env(num_targets, num_observers, simulator_type, time_step, duration)
        total_reward = 0
        observation, infos = env.reset()

        action_counts = {}
        start_time = time.time()

        while env.agents:
            step_start_time = time.time()
            # Sample actions for all agents
            actions = {agent: env.action_space.sample() for agent in env.agents}
            # actions = {agent: 0 for agent in env.agents}
            observation, rewards, terminated, truncated, infos = env.step(actions)
            total_reward += sum(rewards.values())
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            if any(terminated.values()) or any(truncated.values()):
                print("Episode finished")
                break

        end_time = time.time()
        total_duration = end_time - start_time
        total_reward_sum = np.sum(total_reward)
        print(f"Total steps: {env.simulator.time_step_number}")
        print(f"Total duration of episode: {total_duration:.3f} seconds")
        print(f"Total reward: {total_reward}")

        # Prepare the data for logging
        matrices = {
            'adjacency_matrix': env.simulator.adjacency_matrix_acc,
            'data_matrix': env.simulator.data_matrix_acc,
            'contacts_matrix': env.simulator.contacts_matrix_acc,
            'global_observation_counts': np.sum(env.simulator.global_observation_counts, axis=0),
            'max_pointing_accuracy_avg': env.simulator.max_pointing_accuracy_avg,
            'global_observation_status_matrix': env.simulator.global_observation_status_matrix,
            'batteries': env.simulator.batteries,
            'storage': env.simulator.storage,
            'total_reward': total_reward_sum,
            'total_duration': total_duration,
            'total_steps': env.simulator.time_step_number,
        }

        if write_to_csv_file_flag:
            log_full_matrices(matrices,results_folder)
            data_summary = {
                'Total Reward': total_reward_sum,
                'Total Duration': total_duration,
            }
            log_summary_results(data_summary, results_folder)

        if plot_flag:
            plot(matrices, results_folder_plots, total_duration, total_reward_sum, env.simulator.time_step_number)
    
    if write_to_csv_file_flag:
        compute_statistics_from_npy(results_folder, relevant_attributes)
        print("Averages written to averages.csv")

