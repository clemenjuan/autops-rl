import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from src.simulator import Simulator, CentralizedSimulator
from ray.rllib.connectors.env_to_module import FlattenObservations
import os
from datetime import datetime
import json
import random


class FSS_env(MultiAgentEnv):
    metadata = {
        "name": "FSS_env-v1",
    }
    
    def __init__(self, config=None):
        super().__init__()
        
        # Extract parameters from config
        if config is None:
            config = {}
            
        self.num_targets = config.get("num_targets", 20)
        self.num_observers = config.get("num_observers", 20)
        self.simulator_type = config.get("simulator_type", "everyone")
        self.time_step = config.get("time_step", 1)
        self.duration = config.get("duration", 24*60*60)
        self.seed = config.get("seed", 42)
        self.reward_type = config.get("reward_type", "case1")
        self.reward_config = config.get("reward_config", None)
        
        # Environment parameters
        self.num_satellites = self.num_targets + self.num_observers
        self.sim_time = 0
        self.latest_step_duration = 0
        assert self.num_observers > 0
        assert self.num_targets > 0
        self.special_events_count = 0
        self.special_event_observe = 0
        self.special_event_communicate = 0
        self.orbital_params_order = ['semimajoraxis', 'inclination', 'eccentricity',
                    'raan', 'arg_of_perigee', 'true_anomaly', 'mean_anomaly',
                    'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        self.orbital_params_order_targets = ['x', 'y', 'z', 'vx', 'vy', 'vz']

        # Initialize agents
        self.possible_agents = ["observer_" + str(r) for r in range(self.num_observers)]
        self.agents = self.possible_agents.copy()
        self.agent_name_mapping = dict(
            zip(self.agents, list(range(len(self.agents))))
        )  # Mapping of agent names to indices

        # Define the action and observation spaces for each agent
        self._action_space = spaces.Discrete(3)
        
        self._observation_space = spaces.Dict({
                "observer_satellites": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observers, len(self.orbital_params_order)), dtype=np.float32),
                "band": spaces.Box(low=1, high=5, shape=(1,), dtype=np.int8),
                "target_satellites": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_targets, len(self.orbital_params_order_targets)), dtype=np.float32),
                "availability": spaces.MultiBinary(1),
                "battery": spaces.Box(low=0, high=1, shape=(self.num_observers,), dtype=np.float32),
                "storage": spaces.Box(low=0, high=1, shape=(self.num_observers,), dtype=np.float32),
                "observation_status": spaces.Box(low=0, high=3, shape=(self.num_targets,), dtype=np.float32),
                "pointing_accuracy": spaces.Box(low=0, high=1, shape=(self.num_observers, self.num_targets), dtype=np.float32),
                "communication_status": spaces.Box(low=0, high=1, shape=(self.num_observers,), dtype=np.int8),
                "communication_ability": spaces.MultiBinary(self.num_observers)
            })
        
        self.action_spaces = {
            agent_id: self._action_space
            for agent_id in self.possible_agents
        }
        self.observation_spaces = {
            agent_id: self._observation_space
            for agent_id in self.possible_agents
        }
        
        # Initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(
                self.num_targets, self.num_observers, self.time_step, self.duration,
                reward_type=self.reward_type, reward_config=self.reward_config
            )
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(
                self.num_targets, self.num_observers, self.time_step, self.duration,
                reward_type=self.reward_type, reward_config=self.reward_config
            )
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(
                self.num_targets, self.num_observers, self.time_step, self.duration,
                reward_type=self.reward_type, reward_config=self.reward_config
            )
        else:
            raise ValueError("Invalid simulator type. Choose from 'centralized', 'decentralized', or 'everyone'.")

        # Initialize event tracking metrics
        self.observation_events = {f"observer_{i}": [] for i in range(self.num_observers)}
        self.communication_events = {f"observer_{i}": [] for i in range(self.num_observers)}
        self.observation_counts = {f"observer_{i}": 0 for i in range(self.num_observers)}
        self.communication_counts = {f"observer_{i}": 0 for i in range(self.num_observers)}

    #@override
    def get_observation_space(self, agent_id):
        # All observer agents have the same observation space in this implementation
        if agent_id.startswith("observer_"):
            return self._observation_space
        else:
            raise ValueError(f"Invalid agent id: {agent_id}!")

    #@override
    def get_action_space(self, agent_id):
        # All observer agents have the same action space in this implementation
        if agent_id.startswith("observer_"):
            return self._action_space
        else:
            raise ValueError(f"Invalid agent id: {agent_id}!")

    def reset(self, *, seed=None, options=None):
        # Reset the environment to an initial state
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)

        self.sim_time = 0
        self.latest_step_duration = 0
        self.special_events_count = 0
        self.special_event_observe = 0
        self.special_event_communicate = 0

        # Re-initialize simulator
        if self.simulator_type == 'centralized':
            self.simulator = CentralizedSimulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'decentralized':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        elif self.simulator_type == 'everyone':
            self.simulator = Simulator(self.num_targets, self.num_observers, self.time_step, self.duration)
        
        # Reset the list of agents
        self.agents = self.possible_agents.copy()
        
        # Reset event tracking metrics
        self.observation_events = {f"observer_{i}": [] for i in range(self.num_observers)}
        self.communication_events = {f"observer_{i}": [] for i in range(self.num_observers)}
        self.observation_counts = {f"observer_{i}": 0 for i in range(self.num_observers)}
        self.communication_counts = {f"observer_{i}": 0 for i in range(self.num_observers)}
        
        # Generate observations for each agent
        observations = {}
        infos = {}
        
        for agent in self.agents:
            observations[agent] = self._generate_observation(agent)
            infos[agent] = {}
        
        return observations, infos

    def step(self, action_dict):
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict

        # TODO: keep the agents that should act next
        #In general, the returned observations dict must contain those agents (and only those agents) that should act next. Agent IDs that should NOT act in the next step() call must NOT have their observations in the returned observations dict.
        #In summary, the exact order and synchronization of agent actions in your multi-agent episode is determined through the agent IDs contained in (or missing from) your observations dicts. Only those agent IDs that are expected to compute and send actions into the next step() call must be part of the returned observation dict.

        assert self.agents, "Cannot step an environment with no agents"
        assert set(action_dict.keys()) == set(self.agents), "Actions must be provided for all agents"

        # Calculate rewards, new observations, etc. as you already do
        observations = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}
        
        # Convert action_dict to the format expected by the simulator
        processed_actions = {}
        for agent_id, action in action_dict.items():
            # Extract the actual integer action value
            if isinstance(action, dict):
                # If action is a dict (from RLlib's new API)
                action_value = list(action.values())[0]  # Get the first value
            elif hasattr(action, 'item'):
                # If action is a tensor
                action_value = action.item()
            else:
                # If action is already an integer or similar
                action_value = action
            
            processed_actions[agent_id] = int(action_value)
        
        # Create zero actions for eventless steps
        zero_actions = {agent: 0 for agent in self.agents}

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
                break

        if special_event_detected:
            # Step the simulator with actual actions
            rewards, done = self.simulator.step(processed_actions, self.simulator_type, self.agents)

        # Generate observations for each agent
        for agent in self.agents:
            observations[agent] = self._generate_observation(agent)
            rewards[agent] = rewards.get(agent, 0.0)
            terminated[agent] = False
            truncated[agent] = False
            infos[agent] = {}
        
        # Always add the __all__ key to terminated and truncated
        terminated["__all__"] = False
        truncated["__all__"] = False
        
        if done:
            for agent in self.agents:
                terminated[agent] = True
                truncated[agent] = False
            terminated["__all__"] = True
            truncated["__all__"] = False
            self.agents = []

            # Save metrics to file for later analysis
            metrics_dir = os.path.join("collected_metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(metrics_dir, f"metrics_{timestamp}_{self.reward_type}_simulator_type_{self.simulator_type}.json")

            # Collect comprehensive metrics including event tracking
            metrics_data = self.collect_comprehensive_metrics()
            
            # Add termination information
            metrics_data["is_terminal"] = True
            metrics_data["step"] = self.simulator.time_step_number
            metrics_data["sim_time"] = self.sim_time
            
            # Save metrics to file
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)

            # Print summary metrics
            print(f"Special events detected: {self.special_events_count}")
            print(f"Special events for observing: {self.special_event_observe}")
            print(f"Special events for communicating: {self.special_event_communicate}")
            print(f"Observed targets: {metrics_data['observation_stats']['observed_targets']} out of {metrics_data['observation_stats']['total_targets']} ({metrics_data['observation_stats']['observation_percentage']:.2f}%)")
            print(f"Forced termination at step {self.simulator.time_step_number}")
                
        
        return observations, rewards, terminated, truncated, infos

    def _generate_observation(self, agent):
        """
        Generates observation for one agent
        Now all matrixes are synchronized for each timestep, so observation will depend on them
        Each observer has its own states as observation (at least)
        """
        # Generate observations based on the current state and communication history
        # Update observer's own state in the observation vector
        i = self.agent_name_mapping[agent]
        orbital_params_order = ['semimajoraxis', 'inclination', 'eccentricity',
                    'raan', 'arg_of_perigee', 'true_anomaly', 'mean_anomaly',
                    'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        orbital_params_order_targets = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        observer = self.simulator.observer_satellites[i]
        
        # Start by initializing the observation dictionary with proper shapes and default values
        # Use self.observation_spaces instead of self._observation_spaces
        observation = {key: np.zeros(shape=space.shape, dtype=space.dtype) 
                   for key, space in self.observation_spaces[agent].spaces.items()}
        
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

        # Ensure everything matches the expected types and shapes
        for key, value in observation.items():
            obs_space = self.observation_spaces[agent].spaces[key]
            assert observation[key].dtype == obs_space.dtype, f"Type mismatch for {key}: expected {obs_space.dtype}, got {observation[key].dtype}"
            assert observation[key].shape == obs_space.shape, f"Shape mismatch for {key}: expected {obs_space.shape}, got {observation[key].shape}"
            if not obs_space.contains(observation[key]):
                raise ValueError(f"{key} does not match the expected space")
            
        # Use np.clip to ensure all values are within the defined bounds for continuous spaces
        for key in observation:
            if isinstance(self.observation_spaces[agent].spaces[key], spaces.Box):
                observation[key] = np.clip(
                    observation[key],
                    self.observation_spaces[agent].spaces[key].low,
                    self.observation_spaces[agent].spaces[key].high
                )
        
        return observation
    
    def detect_special_event(self):
        can_observe = self.simulator.get_global_targets(self.simulator.observer_satellites, self.simulator.target_satellites)
        can_communicate = self.simulator.get_global_communication_ability(self.simulator.observer_satellites,self.simulator.time_step, self.simulator_type)

        if can_observe or can_communicate:
            self.special_events_count += 1
            if can_observe:
                self.special_event_observe += 1
            if can_communicate:
                self.special_event_communicate += 1
            return True

        return False


    def _calculate_avg_steps_between_events(self, events_dict):
        """Calculate average time steps between events for each agent"""
        result = {}
        for agent, events in events_dict.items():
            if len(events) <= 1:
                result[agent] = 0
            else:
                # Calculate time differences between consecutive events (in env steps)
                steps_diffs = [events[i] - events[i-1] for i in range(1, len(events))]
                result[agent] = sum(steps_diffs) / len(steps_diffs) if steps_diffs else 0
        
        # Add global average
        all_events = [event for agent_events in events_dict.values() for event in agent_events]
        all_events.sort()
        if len(all_events) <= 1:
            result["global_average"] = 0
        else:
            steps_diffs = [all_events[i] - all_events[i-1] for i in range(1, len(all_events))]
            result["global_average"] = sum(steps_diffs) / len(steps_diffs) if steps_diffs else 0
        
        return result

    def collect_comprehensive_metrics(self):
        """Collect all relevant metrics in one place"""
        
        # Calculate observation stats
        total_targets = self.num_targets
        observed_targets = np.sum(np.any(self.simulator.global_observation_status_matrix == 3, axis=0))
        observation_percentage = (observed_targets / total_targets) * 100 if total_targets > 0 else 0
        
        # Calculate connectivity stats
        total_possible_connections = self.num_observers * (self.num_observers - 1)
        active_connections = np.sum(self.simulator.adjacency_matrix_acc) - self.num_observers  # Remove self-connections
        connectivity_percentage = (active_connections / total_possible_connections) * 100 if total_possible_connections > 0 else 0
        
        # Battery and storage levels
        battery_levels = {f"observer_{i}": sat.epsys['EnergyAvailable'] / sat.epsys['EnergyStorage'] 
                         for i, sat in enumerate(self.simulator.observer_satellites)}
        storage_levels = {f"observer_{i}": sat.DataHand['StorageAvailable'] / sat.DataHand['DataStorage'] 
                         for i, sat in enumerate(self.simulator.observer_satellites)}
        
        # Collect baseline metrics from the existing function
        base_metrics = {
            "simulator_type": self.simulator_type,
            "num_observers": self.num_observers,
            "num_targets": self.num_targets,
            "time_step": self.time_step,
            "duration": self.duration,
            "current_time": self.sim_time,
            "reward_type": self.reward_type,
            "seed": self.seed,
            
            # Add observation stats
            "observation_stats": {
                "total_targets": int(total_targets),
                "observed_targets": int(observed_targets),
                "observation_percentage": float(observation_percentage)
            },
            
            # Add connectivity stats
            "connectivity_stats": {
                "total_possible_connections": int(total_possible_connections),
                "active_connections": int(active_connections),
                "connectivity_percentage": float(connectivity_percentage)
            },
            
            # Add resource stats
            "resource_stats": {
                "battery_levels": battery_levels,
                "storage_levels": storage_levels,
                "average_battery": float(np.mean(list(battery_levels.values()))),
                "average_storage": float(np.mean(list(storage_levels.values())))
            },
            
            # Add event tracking metrics
            "event_stats": {
                "special_events_count": self.special_events_count,
                "special_event_observe": self.special_event_observe,
                "special_event_communicate": self.special_event_communicate
            },
            
            # Add summary matrices (as compact statistics)
            "matrix_stats": {
                "global_observation_status_avg": float(np.mean(self.simulator.global_observation_status_matrix)),
                "adjacency_matrix_acc_avg": float(np.mean(self.simulator.adjacency_matrix_acc)),
                "global_observation_counts": float(np.mean(self.simulator.global_observation_counts)),
                "global_communication_counts": float(np.mean(self.simulator.global_communication_counts))
            }
        }
        
        return base_metrics

def _env_to_module_pipeline(env):
    return FlattenObservations(
        input_observation_space=env.observation_space,
        input_action_space=env.action_space,
        multi_agent=True
    )