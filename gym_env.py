import numpy as np
from gymnasium import Env, spaces
from pettingzoo import ParallelEnv



import logging
from copy import deepcopy

from CommSubsystem import CommSubsystem
from OpticPayload import OpticPayload
from simulator import Simulator, CentralizedSimulator, MixedSimulator, DecentralizedSimulator


class SatelliteEnv(Env, ParallelEnv):
    def __init__(self, num_targets: int, 
                 num_observers: int, 
                 simulator_type: str = 'everyone', 
                 time_step: float = 1, 
                 duration: float = 24*60):
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

        # Action, observation and state spaces
        self.action_spaces = dict(
                zip(self.agents, [spaces.Discrete(2 + num_targets)] * self.num_observers)
            )
        self.observation_spaces = dict(
            zip(self.agents, [spaces.Dict({
            'observer_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, 14)),  # Orbital parameters for each observer satellite
            'target_satellites': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, 6)),  # Position and velocity for each target satellite
            'availability': spaces.Discrete(2, seed=42), # {0, 1} Availability of each observer satellite
            'battery': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),  # Battery level for each observer satellite
            'storage': spaces.Box(low=0, high=1, shape=(self.num_observers, 1)),  # Storage level for each observer satellite
            'observation_status': spaces.MultiDiscrete([4] * self.num_targets),  # Observation status for each target satellite
            'pointing_accuracy': spaces.Box(low=0, high=np.inf, shape=(self.num_observers, self.num_targets)),  # Pointing accuracy for each target satellite
            'communication_status': spaces.MultiBinary(self.num_observers),  # Communication status for each observer satellite
            'communication_ability': spaces.MultiBinary(self.num_observers, self.num_observers),  # Communication ability for each observer satellite
            })] * self.num_observers)
        )
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, 14)) # to be fixed

        # Initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(num_targets, num_observers, time_step, duration)
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(num_targets, num_observers, time_step, duration)
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(num_targets, num_observers, time_step, duration)
        else:
            raise ValueError("Invalid simulator type. Choose from 'centralized', 'decentralized', or 'everyone'.")
        
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return env.action_spaces[agent]
    

    def _generate_observation(self):
        """
        Generates observation for one agent
        Now all matrixes are synchronized for each timestep, so observation will depend on them
        Each observer has its own states as observation (at least)
        """
        # Generate observations based on the current state and communication history
        # Update observer's own state in the observation vector
        observation = {
            'observer_satellites': np.zeros((self.num_observers, 14)),
            'target_satellites': np.zeros((self.num_targets, 6)),
            'availability': None,
            'battery': np.zeros((self.num_observers, 1)),
            'storage': np.zeros((self.num_observers, 1)),
            'observation_status': np.zeros(self.num_targets, dtype=int),
            'pointing_accuracy': np.zeros((self.num_observers, self.num_targets)),
            'communication_status': np.zeros(self.num_observers, dtype=int),
            'communication_ability': np.zeros((self.num_observers, self.num_observers), dtype=int)
        }

        orbital_params_order = ['semimajoraxis', 'inclination', 'eccentricity',
                                'raan', 'arg_of_perigee', 'true_anomaly', 'mean_anomaly',
                                'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        orbital_params_order_targets = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for i, observer in enumerate(self.simulator.observer_satellites):
            observer_orbit_params = np.array([observer.orbit[param] for param in orbital_params_order])
            observation['observer_satellites'][i] = observer_orbit_params
            observation['availability'] = observer.availability
            observation['battery'][i] = observer.epsys['EnergyAvailable'] / observer.epsys['EnergyStorage']
            observation['storage'][i] = observer.DataHand['StorageAvailable'] / observer.DataHand['DataStorage']
            observation['observation_status'] = observer.observation_status_matrix
            observation['pointing_accuracy'][i] = observer.pointing_accuracy_matrix
            observation['communication_status'] = self.simulator.adjacency_matrix_acc[i]
            observation['communication_ability'][i] = observer.get_communication_ability(self.simulator.observer_satellites, self.simulator.time_step, self.simulator_type)

            for j, other_observer in enumerate(self.simulator.observer_satellites):
                if i != j:
                    if self.simulator.adjacency_matrix_acc[i, j] == 1:
                        other_observer_orbit_params = np.array([other_observer.orbit[param] for param in orbital_params_order])
                        observation['observer_satellites'][j] = other_observer_orbit_params
                    if self.simulator.adjacency_matrix[i, j] == 1:
                        observation['battery'][j] = other_observer.epsys['EnergyAvailable'] / other_observer.epsys['EnergyStorage']
                        observation['storage'][j] = other_observer.DataHand['StorageAvailable'] / other_observer.DataHand['DataStorage']
                        observation['communication_ability'][j] = other_observer.get_communication_ability(self.simulator.observer_satellites, self.simulator.time_step, self.simulator_type)
                        observation['pointing_accuracy'][j] = other_observer.pointing_accuracy_matrix

            for k, target in enumerate(self.simulator.target_satellites):
                if self.simulator.contacts_matrix[i, k] == 1:
                    target_orbit_params = np.array([target.orbit[param] for param in orbital_params_order_targets])
                    observation['target_satellites'][k] = target_orbit_params

        return observation
    
    def get_obs(self):
        """Format the observation per the PettingZoo Parallel API."""
        return {
            agent: self._generate_observation()
            for agent in self.possible_agents
        }
    
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

    def reset(self):
        self.delete_simulator()
        # Initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        else:
            raise ValueError("Invalid simulator type. Choose from 'centralized', 'decentralized', or 'everyone'.")
        
        self.agents = self.possible_agents[:]

        observation = self.get_obs()
        infos = {agent: {} for agent in self.agents}
        return observation, infos

    def shield_actions(self, actions):
        """
            Shield actions for each satellite
            if observer.is_processing:
                actions[i] = 0 # standby

            if observer.battery < 10:
                actions[i] = 0 # standby
        """
        pass

    def step(self, action_vector):
        # Implement the logic to execute one step in the environment
        self.simulator.reset_matrices_for_timestep()
        for observer in self.simulator.observer_satellites:
            observer.check_and_update_processing_state(self.time_step)
        self.simulator.get_global_targets(self.simulator.observer_satellites, self.simulator.target_satellites)

        # Execute actions in the simulator
        reward, done = self.simulator.step(action_vector, self.simulator_type)
        observation = self.get_obs()
        infos = {agent: {} for agent in self.agents}
        print(f"Step reward: {reward}")
        
        return observation, reward, done, infos

if __name__ == "__main__":
    print("Creating environment...")
    # Run the simulation until timeout or agent failure
    env = SatelliteEnv(num_targets=10, num_observers=100)
    total_reward = 0
    print("Environment created. Resetting...")
    observation, info = env.reset()
    print("Resetting environment done. Starting simulation...")


    while True:
        # action_vector = np.zeros(env.num_observers)e
        print("Getting actions...")
        #while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        print("Actions: ", actions)

            # observer_observation = observations[i]  # Get the current observer's observation
            # action = policy_model.predict(observer_observation)  # Predict action based on the observation
        
            # Shield actions
            # actions[i] = self.shield_actions(action, observer)
            # action = env.action_space.sample() in step already
        print("Actions received. Executing step...")
        observation, reward, done, info = env.step(actions)
        total_reward += reward
        print(f"\tReward: {reward:.3f} ({total_reward:.3f} cumulative)")
        if done:
            print("Episode finished")
            break

    print("Adjacency matrix:")
    print()  # Add a new line
    print(f"{env.simulator.adjacency_matrix_acc}")
    print()
    print("Data matrix:")
    print()  # Add a new line
    print(f"{env.simulator.data_matrix_acc}")
    print()
    print("Contacts matrix:")
    print()  # Add a new line
    print(f"{env.simulator.contacts_matrix_acc}")
    print()
    print("Global observation count matrix:")
    print()  # Add a new line
    print(f"{env.simulator.global_observation_counts}")
    print()
    print("Global observation status matrix:")
    print()  # Add a new line
    print(f"{env.simulator.global_observation_status_matrix}")
    
    print(f"Total time: {env.simulator.total_time} seconds")
    print(f"Total reward: {total_reward}")

    