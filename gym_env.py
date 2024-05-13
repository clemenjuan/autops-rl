import os
import numpy as np
import functools
from copy import copy
from gymnasium import spaces
# from gymnasium.envs import MultiAgentEnv
from pettingzoo.utils import ParallelEnv
from pettingzoo.utils import agent_selector
from pettingzoo.test import parallel_api_test


import logging
from copy import deepcopy
from collections import OrderedDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from simulator import Simulator, CentralizedSimulator, MixedSimulator, DecentralizedSimulator
import time
from plotting import plot
from data_logging import log_full_matrices, log_summary_results, compute_statistics_from_npy


class SatelliteEnv(ParallelEnv, MultiAgentEnv):
    metadata = {
        "name": "SatelliteEnv",
    }
    def __init__(self, num_targets: int, 
                 num_observers: int, 
                 simulator_type: str = 'everyone', 
                 time_step: float = 1, 
                 duration: float = 24*60*60):
        super(SatelliteEnv, self).__init__()
        
        # Environment setup
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

        self.agents = ["observer_" + str(r) for r in range(num_observers)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_observers)))) # Mapping of agent names to indices
        self._agent_selector = agent_selector(self.agents)
        self.orbital_params_order = ['semimajoraxis', 'inclination', 'eccentricity',
                    'raan', 'arg_of_perigee', 'true_anomaly', 'mean_anomaly',
                    'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        self.orbital_params_order_targets = ['x', 'y', 'z', 'vx', 'vy', 'vz']

        # Action, observation and state spaces
         # self.action_space
        self._action_spaces = dict(
                zip(self.agents, [spaces.Discrete(2 + num_targets)] * self.num_observers)
            )
        # self.observation_space
        self._observation_spaces = dict(
            zip(self.agents, [spaces.Dict({
                'observer_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, len(self.orbital_params_order))),
                'target_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, len(self.orbital_params_order_targets))),
                'availability': spaces.MultiBinary(1),
                'battery': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),
                'storage': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),
                'observation_status': spaces.Box(low=0, high=3, shape=(self.num_targets, 1)),
                'pointing_accuracy': spaces.Box(low=0, high=np.inf, shape=(self.num_observers, self.num_targets)),
                'communication_status': spaces.Box(low=0, high=1, shape=(self.num_observers, self.num_observers), dtype=np.int8),
                'communication_ability': spaces.MultiBinary(self.num_observers)
            })] * self.num_observers)
        )

        '''
        self._action_spaces = {agent: spaces.Discrete(2 + num_targets) for agent in self.agents}
        self._observation_spaces = {
            agent: spaces.Dict({
                'observer_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, len(orbital_params_order))),
                'target_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, len(orbital_params_order_targets))),
                'availability': spaces.MultiBinary(1),
                'battery': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),
                'storage': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),
                'observation_status': spaces.Box(low=0, high=3, shape=(self.num_targets, 1)),
                'pointing_accuracy': spaces.Box(low=0, high=np.inf, shape=(self.num_observers, self.num_targets)),
                'communication_status': spaces.Box(low=0, high=1, shape=(self.num_observers, self.num_observers), dtype=np.int8),
                'communication_ability': spaces.MultiBinary(self.num_observers)
            }) for agent in self.agents
        }
        '''

        
        # self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, len(orbital_params_order))) # to be fixed


        # Initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(num_targets, num_observers, time_step, duration)
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(num_targets, num_observers, time_step, duration)
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(num_targets, num_observers, time_step, duration)
        else:
            raise ValueError("Invalid simulator type. Choose from 'centralized', 'decentralized', or 'everyone'.")
        

        
    
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def get_observation_space(self, agent):
        if agent not in self._observation_spaces:
            raise ValueError(f"Invalid agent ID {agent}. Available agents: {list(self._observation_spaces.keys())}")
        return self._observation_spaces[agent]

    

    @functools.lru_cache(maxsize=None)
    def get_action_space(self, agent):
        if agent not in self._action_spaces:
            raise ValueError(f"Invalid agent ID {agent}. Available agents: {list(self._action_spaces.keys())}")
        return self._action_spaces[agent]
    
    

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

        observation = {
            'observer_satellites': np.zeros((self.num_observers, len(orbital_params_order)), dtype=np.float32),
            'availability': np.zeros((1,), dtype=np.int8),  # MultiBinary should be dtype int8 or similar
            'battery': np.zeros((self.num_observers, 1), dtype=np.float32),
            'communication_ability': np.zeros((self.num_observers,), dtype=np.int8),  # MultiBinary adjusted
            'communication_status': np.zeros((self.num_observers,self.num_observers), dtype=np.int8),  # MultiBinary adjusted
            'observation_status': np.zeros((self.num_targets, 1), dtype=np.float32),
            'pointing_accuracy': np.zeros((self.num_observers, self.num_targets), dtype=np.float32),
            'storage': np.zeros((self.num_observers, 1), dtype=np.float32),
            'observer_satellites': np.zeros((self.num_observers, len(orbital_params_order)), dtype=np.float32),
            'target_satellites': np.zeros((self.num_targets, len(orbital_params_order_targets)), dtype=np.float32)
            }
        
        i = agent_idx
        observer_orbit_params = np.array([observer.orbit[param] for param in orbital_params_order], dtype=np.float32)
        observation['observer_satellites'][i] = observer_orbit_params
        observation['availability'] = np.array([observer.availability], dtype=np.int8)
        observation['battery'][i] = np.array([observer.epsys['EnergyAvailable'] / observer.epsys['EnergyStorage']], dtype=np.float32)
        observation['storage'][i] = np.array([observer.DataHand['StorageAvailable'] / observer.DataHand['DataStorage']], dtype=np.float32)
        observation['observation_status'] = np.array(observer.observation_status_matrix, dtype=np.int8)
        observation['pointing_accuracy'] = np.array(observer.pointing_accuracy_matrix, dtype=np.float32)
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
                    observation['communication_ability'][j] = np.array(other_observer.get_communication_ability(self.simulator.observer_satellites, self.simulator.time_step, self.simulator_type),dtype=np.int8)
                    observation['pointing_accuracy'][j] = np.array(other_observer.pointing_accuracy_matrix, dtype=np.float32)

        for k, target in enumerate(self.simulator.target_satellites):
            if self.simulator.contacts_matrix[i, k] == 1:
                target_orbit_params = np.array([target.orbit[param] for param in orbital_params_order_targets], dtype=np.float32)
                observation['target_satellites'][k] = target_orbit_params

        
        # print(f"Observation for {agent}: {observation}")
        return observation
        
        '''
        total_elements = (
            np.prod(observation['observer_satellites'].shape) +
            np.prod(observation['target_satellites'].shape) +
            np.prod(observation['battery'].shape) +
            np.prod(observation['storage'].shape) +
            np.prod(observation['observation_status'].shape) +
            np.prod(observation['pointing_accuracy'].shape) +
            np.prod(observation['communication_status'].shape) +
            np.prod(observation['communication_ability'].shape) +
            1 # availability
        )
        print("Observer Satellites Shape:", observation['observer_satellites'].shape)
        print("Target Satellites Shape:", observation['target_satellites'].shape)
        print("Battery Shape:", observation['battery'].shape)
        print("Storage Shape:", observation['storage'].shape)
        print("Observation Status Shape:", observation['observation_status'].shape)
        print("Pointing Accuracy Shape:", observation['pointing_accuracy'].shape)
        print("Communication Status Shape:", observation['communication_status'].shape)
        print("Communication Ability Shape:", observation['communication_ability'].shape)
        print("Availability Shape:", np.array([observation['availability']]).shape)

        print(f"Total elements after observation: {total_elements}")
        '''
        
        
    
    def observe(self, agent):
        """Format the observation per the PettingZoo Parallel API."""
        # print("Observing in observe...")
        # observation = OrderedDict()
        #for agent in self.possible_agents:
        #observation = self._generate_observation(agent)
        observation = self.get_obs()

        # print("Finished observing")
        return observation
    
    def get_obs(self):
        """Format the observation per the PettingZoo Parallel API."""
        # print("Getting observations (get_obs)...")
        observations = {}
        for agent in self.possible_agents:
            observations[agent] = self._generate_observation(agent)
        # print(f"Observations: {observations}")
        # print("Finished getting observations")
        return observations
    
    def _generate_state(self): # to be fixed
        """
        Generate the state of the environment
        """
        state = np.zeros((self.num_observers, 14))
        for i, observer in enumerate(self.simulator.observer_satellites):
            state[i] = observer.orbit
        return state

    def delete_simulator(self):
        """
        Delete the simulator object to free up memory
        """
        try:
            del self.simulator
        except AttributeError:
            pass

    def reset_main(self, seed=None, options=None):
        self.delete_simulator()
        self.agents = copy(self.possible_agents)
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        else:
            raise ValueError("Invalid simulator type. Choose from 'centralized', 'decentralized', or 'everyone'.")
        
        
        observation = self.get_obs()
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        infos = self.infos
        self.observation = observation
        # print(f"Agents: {self.agents}")
        return observation, infos
    
    def step_main(self, action_vector):
        # Implement the logic to execute one step in the environment
        # print("Executing step...")
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        self.simulator.reset_matrices_for_timestep()
        for observer in self.simulator.observer_satellites:
            observer.check_and_update_processing_state(self.time_step)
        self.simulator.get_global_targets(self.simulator.observer_satellites, self.simulator.target_satellites)

        # Execute actions in the simulator
        self.start_time_step = time.time()

        reward, done = self.simulator.step(action_vector, self.simulator_type)
        # Distribute reward among agents
        rewards = {agent: reward / len(self.agents) for agent in self.agents}
        self.step_timer = time.time() - self.start_time_step
        self.time_elapsed = time.time() - self.simulator.start_time
        average_time_per_step = self.time_elapsed / self.simulator.time_step_number
        remaining_steps = self.simulator.num_steps - self.simulator.time_step_number
        remaining_time_estimate = remaining_steps * average_time_per_step
        remaining_hours = int(remaining_time_estimate // 3600)  # Calculate remaining hours (3600 seconds in an hour)
        remaining_minutes = int((remaining_time_estimate % 3600) // 60)  # Calculate remaining minutes
        remaining_seconds = int(remaining_time_estimate % 60)  # Calculate remaining seconds

        if self.simulator.time_step_number % steps_batch_size == 0:
            print(f"Step {self.simulator.time_step_number} completed in {self.step_timer:.6f} seconds. Estimated time remaining: {remaining_hours} hours, {remaining_minutes} minutes, and {remaining_seconds} seconds.")

        print("Beginning observations in step...")
        self.observations = self.get_obs()
        print(f"Observations: {self.observations}")
        observation = self.observations
        # observation = self.get_obs()
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: done for agent in self.agents}

        
        return observation, rewards, terminated, truncated, infos

    def shield_actions(self, actions):
        """
            Shield actions for each satellite
            if observer.is_processing:
                actions[i] = 0 # standby

            if observer.battery < 10:
                actions[i] = 0 # standby
        """
        pass

    def render(self):
        pass

    
    
    def reset(self, seed=None, options=None):
        self.delete_simulator()
        self.agents = copy(self.possible_agents)
        
        # Initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        else:
            raise ValueError("Invalid simulator type. Choose from 'centralized', 'decentralized', or 'everyone'.")
        
        # print("Getting observations in reset...")
        # observations = self.get_obs()
        observations = {
            agent: (
            self._generate_observation(agent)
            )
            for agent in self.agents
        }
        # print(f"Observation: {self.observation}")

        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminate = False
        self.truncate = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        #return observation, self.infos
        return observations, self.infos

    
    def step(self, action_vector):
        # Implement the logic to execute one step in the environment
        print("Executing step...")
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        ##### removed and moved to simulator.step()
        '''
        self.simulator.reset_matrices_for_timestep()
        for observer in self.simulator.observer_satellites:
            observer.check_and_update_processing_state(self.time_step)
        self.simulator.get_global_targets(self.simulator.observer_satellites, self.simulator.target_satellites)
        '''
    
        # Execute actions in the simulator
        self.start_time_step = time.time()

        reward, done = self.simulator.step(action_vector, self.simulator_type)
        # Distribute reward among agents
        rewards = {agent: reward / len(self.agents) for agent in self.agents}
        self.step_timer = time.time() - self.start_time_step
        self.time_elapsed = time.time() - self.simulator.start_time
        average_time_per_step = self.time_elapsed / self.simulator.time_step_number
        remaining_steps = self.simulator.num_steps - self.simulator.time_step_number
        remaining_time_estimate = remaining_steps * average_time_per_step
        remaining_hours = int(remaining_time_estimate // 3600)  # Calculate remaining hours (3600 seconds in an hour)
        remaining_minutes = int((remaining_time_estimate % 3600) // 60)  # Calculate remaining minutes
        remaining_seconds = int(remaining_time_estimate % 60)  # Calculate remaining seconds

        # print("Beginning observations in step...")
        # observations = self.get_obs()
        observations = {
            agent: (
            self._generate_observation(agent)
            )
            for agent in self.agents
        }
        # print(f"Observations: {self.observations}")
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: done for agent in self.agents}
        infos = self.infos
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        
        return observations, rewards, terminated, truncated, infos
    
    @functools.lru_cache(maxsize=None)
    def action_space(self):
        # This method provides direct access to the action space for compatibility.
        # return self.get_action_space(self.agents[0])
        return spaces.Discrete(2 + self.num_targets)

    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        # This method provides direct access to the observation space for compatibility.
        # return self.get_observation_space(self.agents[0])
        return spaces.Dict({
                    'observer_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, len(self.orbital_params_order))),
                    'target_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, len(self.orbital_params_order_targets))),
                    'availability': spaces.MultiBinary(1),
                    'battery': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),
                    'storage': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),
                    'observation_status': spaces.Box(low=0, high=3, shape=(self.num_targets, 1)),
                    'pointing_accuracy': spaces.Box(low=0, high=np.inf, shape=(self.num_observers, self.num_targets)),
                    'communication_status': spaces.Box(low=0, high=1, shape=(self.num_observers, self.num_observers), dtype=np.int8),
                    'communication_ability': spaces.MultiBinary(self.num_observers)
                })


if __name__ == "__main__":
    ### Example of how to use the environment for a Monte Carlo simulation
    ############################ EDIT HERE ############################
    num_simulations = 100  # desired number of simulations
    num_targets = 10 # Number of target satellites
    num_observers = 10 # Number of observer satellites
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
        'global_observation_count_matrix', 'maximum_pointing_accuracy_average_matrix',
        'global_observation_status_matrix', 'batteries', 'storage', 'total_reward', 
        'total_duration', 'total_steps',
        ]
    ####################################################################
    env = SatelliteEnv(num_targets=10, num_observers=10, simulator_type='everyone', time_step=1, duration=24*60*60)
    parallel_api_test(env, num_cycles=1_000)
    for i in range(num_simulations):
        print(f"Starting simulation {i}...")
        # print("Creating environment...")
        # Run the simulation until timeout or agent failure
        env = SatelliteEnv(num_targets, num_observers, simulator_type, time_step, duration)
        total_reward = 0
        # print("Environment created. Resetting...")
        observation, infos = env.reset_main()
        # print("Resetting environment done. Starting simulation...")

        action_counts = {}
        start_time = time.time()

        while True:
            step_start_time = time.time()
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            for agent, action in actions.items():
                action_counts.setdefault(agent, []).append(action)
            #if env.simulator.time_step_number % steps_batch_size == 0:
                # print("Actions: ", actions)
                # print(f"Actions received. Executing next {steps_batch_size} steps...")
            observation, rewards, terminated, truncated, infos = env.step_main(actions)
            total_reward += sum(rewards.values())
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            # if env.simulator.time_step_number % steps_batch_size == 0:
            #    print(f"Step {env.simulator.time_step_number} finished, duration: {step_duration:.6f} seconds")
            #    print(f"Observations: {observation}")

            if any(terminated.values()) or any(truncated.values()):
                print("Episode finished")
                # file.write("Episode finished\n")
                break

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Total steps: {env.simulator.time_step_number}")
        print(f"Total duration of episode: {total_duration:.3f} seconds")
        print(f"Total reward: {total_reward}")

        # Prepare the data for plotting
        matrices = {
            'adjacency_matrix': env.simulator.adjacency_matrix_acc,
            'data_matrix': env.simulator.data_matrix_acc,
            'contacts_matrix': env.simulator.contacts_matrix_acc,
            'global_observation_count_matrix': env.simulator.global_observation_counts,
            'maximum_pointing_accuracy_average_matrix': env.simulator.max_pointing_accuracy_avg,
            'global_observation_status_matrix': env.simulator.global_observation_status_matrix,
            'batteries': env.simulator.batteries,
            'storage': env.simulator.storage,
            'total_reward': total_reward,
            'total_duration': total_duration,
            'total_steps': env.simulator.time_step_number,
        }
        np.array([matrices], dtype=np.float32)


        if write_to_csv_file_flag:
            # Log the full matrices as .npy files
            log_full_matrices(matrices, i, results_folder)

            # Log the summary statistics to a single CSV file
            data_summary = {
                'Simulation Index': i,
                'Total Reward': total_reward,
                'Total Duration': total_duration,
                # ... other summary statistics ...
            }
            log_summary_results(data_summary, i, results_folder)

        
        if plot_flag:
            plot(matrices, results_folder_plots, total_duration, total_reward )
    

    if write_to_csv_file_flag:
        # Compute statistics such as the mean of each attribute across all simulations and write to a CSV
        compute_statistics_from_npy(results_folder, relevant_attributes)
        print("Averages written to averages.csv")
        
