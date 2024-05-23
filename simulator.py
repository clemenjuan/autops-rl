import numpy as np
import math
import random
import time


class Simulator():
    """
    Base class for the rest of simulators. It contains common methods and attributes.
    """
    def __init__(self, 
                num_targets: int,
                num_observers: int,
                time_step: float = 1,
                duration: float = 24*60*60,
                )-> None:
        super().__init__()
        from satellites import TargetSatellite, ObserverSatellite
        self.num_satellites = num_targets + num_observers
        self.data_matrix = np.zeros((num_observers, num_observers), dtype=float)  # Current timestep data exchange
        self.data_matrix_acc = np.zeros((num_observers, num_observers), dtype=float)  # Accumulated data exchange
        self.adjacency_matrix = np.zeros((num_observers, num_observers), dtype=int)  # Current timestep adjacency matrix
        self.adjacency_matrix_acc = np.zeros((num_observers, num_observers), dtype=int)  # Accumulated adjacency matrix
        self.contacts_matrix = np.zeros((num_observers, num_targets), dtype=int)  # Current timestep contacted targets matrix
        self.contacts_matrix_acc = np.zeros((num_observers, num_targets), dtype=int)  # Accumulated contacted targets matrix
        self.target_satellites = [TargetSatellite(name=f"Target-{i+1}") for i in range(num_targets)]
        self.observer_satellites = [ObserverSatellite(name=f"Observer-{i+1}", num_targets=num_targets, num_observers=num_observers) for i in range(num_observers)]
        self.time_step = time_step
        self.time_step_number = 0
        self.duration = duration
        self.num_steps = int(duration / time_step)
        self.start_time = time.time()
        self.total_time = None
        self.breaker = False #our mighty loop exiter!
        self.global_observation_counts = np.zeros((num_observers, num_targets), dtype=int)  # Global matrix to track the number of observations for each target
        self.global_observation_status_matrix = np.zeros((num_observers,num_targets), dtype=int)  # Matrix to track observation status
        self.reward_matrix = np.zeros((self.num_steps, num_observers))
        self.max_pointing_accuracy_avg = np.zeros(num_targets, dtype=float)
        self.batteries = np.zeros(num_observers, dtype=float)
        self.storage = np.zeros(num_observers, dtype=float)

    def reset_matrices_for_timestep(self):
        """Resets matrices to only track the latest timestep's observations and connections."""
        self.data_matrix.fill(0)
        self.adjacency_matrix.fill(0)
        self.contacts_matrix.fill(0)

    def process_actions(self, actions, type_of_communication, agents):
        reward_step = {agent: 0 for agent in agents}
        for i, (observer, action) in enumerate(actions.items()):
            agent = agents[i]
            observer = self.observer_satellites[i]
            # print(f"Processing action: {action} for {observer.name}, step: {self.time_step_number}")

            # Update energy consumption and storage consumption based on the action
            if action == 0: # Standby
                power_consumption = observer.power_consumption_rates["standby"]
                # print(f"Observer {i} is on standby")
                observer.stand_by()
                # Deduct the power consumption from the available energy
                if not observer.is_processing:
                    observer.epsys['EnergyAvailable'] -= power_consumption * self.time_step
                # only reduce energy if the satellite is not processing (already deducted in the processing function)
            elif action == 1: # Communication
                communication_done = False
                max_steps = 0
                # Calculate the data size
                data_size = observer.DataHand['DataSize']*8 # in bits
                # Calculate the sum of contacts
                sum_of_contacts = np.sum(self.contacts_matrix[i]) * 28 * 8  # in bits
                # Calculate the sum of adjacency
                sum_of_adjacency = np.sum(self.adjacency_matrix[i]) * observer.DataHand['DataSize'] * 8  # in bits
                # Calculate the sum of accumulated contacts
                sum_of_contacts_acc = np.sum(self.contacts_matrix_acc[i]) * 28 * 8  # in bits
                # Calculate the sum of accumulated adjacency
                sum_of_adjacency_acc = np.sum(self.adjacency_matrix_acc[i]) * observer.DataHand['DataSize'] * 8  # in bits
                # Calculate the total data to transmit
                data_to_transmit = data_size + sum_of_contacts + sum_of_adjacency # - sum_of_contacts_acc - sum_of_adjacency_acc
                data_to_transmit = max(data_to_transmit, 0)
                # print(f"Data to transmit: {data_to_transmit} bits")

                # print(f"Starting communication checks for observer {i}")
                for j, other_observer in enumerate(self.observer_satellites):
                    if i == j:
                        observer.update_adjacency_matrix(i, j) # Update adjacency matrix for self communication
                        continue  # Skip self communication

                    communication_done = False
                    steps = 0
                    data_transmitted = 0
                    total_data_transmitted = 0
                    while not communication_done and total_data_transmitted < data_to_transmit:
                        reward_step[agent], communication_done, steps, contacts_matrix, contacts_matrix_acc, adjacency_matrix, adjacency_matrix_acc, data_matrix, data_matrix_acc, global_observation_counts, max_pointing_accuracy_avg, data_transmitted = observer.propagate_information(i, other_observer, j, self.time_step + steps * self.time_step, type_of_communication, reward_step[agent], steps, communication_done, total_data_transmitted, data_to_transmit)
                        
                        total_data_transmitted += data_transmitted  # Accumulate data transmitted
                        if communication_done or data_transmitted == 0:
                            break  # Exit if communication is done or no data was transmitted
                    
                    max_steps = max(steps, max_steps)
                    

                self.contacts_matrix = np.maximum(contacts_matrix, self.contacts_matrix)
                self.contacts_matrix_acc = np.maximum(contacts_matrix_acc, self.contacts_matrix_acc)
                self.adjacency_matrix = np.maximum(adjacency_matrix, self.adjacency_matrix)
                self.adjacency_matrix_acc = np.maximum(adjacency_matrix_acc, self.adjacency_matrix_acc)
                self.data_matrix = np.maximum(data_matrix, self.data_matrix)
                self.data_matrix_acc = np.maximum(data_matrix_acc, self.data_matrix_acc)
                self.global_observation_counts = np.maximum(global_observation_counts, self.global_observation_counts)
                self.max_pointing_accuracy_avg = np.maximum(max_pointing_accuracy_avg, self.max_pointing_accuracy_avg)
                
                observer.processing_time += max_steps * self.time_step
                power_consumption = observer.power_consumption_rates["communication"]
                storage_consumption = observer.storage_consumption_rates["communication"] # needs fixing 
                observer.epsys['EnergyAvailable'] -= power_consumption * self.time_step * max_steps
                observer.DataHand['StorageAvailable'] -= total_data_transmitted # in bits
                # print(f"Data transmitted: {total_data_transmitted}, Storage available: {observer.DataHand['StorageAvailable']}")
            else: # Observation
                observation_done = False
                steps = 0
                power_consumption = observer.power_consumption_rates["observation"]
                storage_consumption = observer.storage_consumption_rates["observation"]
                # print(f"Observer {i} is observing target {action - 2}")
                while not observation_done:
                    reward_step[agent], observation_done, steps, contacts_matrix, contacts_matrix_acc, adjacency_matrix, adjacency_matrix_acc, data_matrix, data_matrix_acc, global_observation_counts, max_pointing_accuracy_avg = observer.observe_target(i, self.target_satellites[action - 2], action - 2, self.time_step + steps*self.time_step, reward_step[agent], steps, observation_done)
                # print(f"Observer {i} has finished observing target {action - 2}")
                self.contacts_matrix = np.maximum(contacts_matrix, self.contacts_matrix)
                self.contacts_matrix_acc = np.maximum(contacts_matrix_acc, self.contacts_matrix_acc)
                self.adjacency_matrix = np.maximum(adjacency_matrix, self.adjacency_matrix)
                self.adjacency_matrix_acc = np.maximum(adjacency_matrix_acc, self.adjacency_matrix_acc)
                self.data_matrix = np.maximum(data_matrix, self.data_matrix)
                self.data_matrix_acc = np.maximum(data_matrix_acc, self.data_matrix_acc)
                self.global_observation_counts = np.maximum(global_observation_counts, self.global_observation_counts)
                self.max_pointing_accuracy_avg = np.maximum(max_pointing_accuracy_avg, self.max_pointing_accuracy_avg)
                # if self.max_pointing_accuracy_avg.any() > 0:
                #    print(f"ID of max_pointing_accuracy_avg in process_actions: {id(self.max_pointing_accuracy_avg)}")
                #    print(f"Max pointing accuracy average in process actions: {self.max_pointing_accuracy_avg}")

                
                observer.epsys['EnergyAvailable'] -= power_consumption * self.time_step * steps
                observer.DataHand['StorageAvailable'] -= storage_consumption * self.time_step * (steps-1)
                observer.processing_time += steps * self.time_step

            if observer.epsys['EnergyAvailable'] < 0 or observer.DataHand['StorageAvailable'] < 0:
                    print(f"Satellite energy or storage depleted ({observer.name}). Terminating simulation.")
                    reward_step[agent] -= 10000
                    self.breaker = True

            self.batteries[i] = observer.epsys['EnergyAvailable'] / observer.epsys['EnergyStorage']
            self.storage[i] = observer.DataHand['StorageAvailable'] / observer.DataHand['DataStorage']
            # print(f"Storage of observer {i}: {self.storage[i]}")

        return reward_step



    def analyze_and_correct_duplications(self,reward): # needs to be implemented correctly
        # Assuming observation records have been synchronized among satellites
        duplicated_observation_indices = np.argwhere(self.global_observation_counts > 1)
        for obs_index in duplicated_observation_indices:
            # Implement your chosen strategy for handling duplicates here
            # E.g., keep the observation with the highest pointing accuracy, average them, etc.
            reward -= 10  # Penalize for duplicates
        return reward

    def get_global_targets(self, observer_satellites, target_satellites):
        # Each observer satellite detects targets within its view
        for i, observer in enumerate(observer_satellites):
            observer.get_targets(i,target_satellites, self.time_step)

    def propagate_orbits(self):
        # Simulate one time step for all satellites
        for satellite in self.target_satellites:
            satellite.propagate_orbit(self.time_step)
        for i, observer in enumerate(self.observer_satellites):
            observer.propagate_orbit(self.time_step)

    def propagate_attitudes(self):
        # Simulate one time step for all satellites
        for satellite in self.target_satellites:
            satellite.propagate_attitude(self.time_step)
        for observer in self.observer_satellites:
            observer.propagate_attitude(self.time_step)


    
    def update_communication_timeline(self):
        # First, update the timeline for all observer-to-observer and observer-to-target communications
        for observer_index, observer in enumerate(self.observer_satellites):
            # Increment the timeline for each communication link the observer has
            observer.communication_timeline_matrix += 1

            # Use the adjacency matrix to reset the timeline for the latest communicated satellites and targets
            for target_index, was_communicated in enumerate(self.contacts_matrix[observer_index]):
                if was_communicated == 1:  # If there was recent communication or observation
                    observer.communication_timeline_matrix[target_index] = 1  # Reset the timeline to 1 for recent communication

    def update_global_observation_status_matrix(self, observer, target):
        for i, observer_satellite in enumerate(self.observer_satellites):
            for j, target_satellite in enumerate(self.target_satellites):
                self.global_observation_status_matrix[i, j] = observer_satellite.observation_status_matrix[j]



    def step(self, actions, simulator_type, agents): # choose type of communication: centralized, decentralized, everyone
        # Simulate one time step for all satellites
        rewards = {agent: 0 for agent in agents}  # Initialize rewards for all agents
        self.reset_matrices_for_timestep()

        for observer in self.observer_satellites:
            observer.check_and_update_processing_state(self.time_step)
            # Update energy with solar panel charging
            sunlight_exposure = observer.get_sunlight_exposure()
            observer.charge_battery(sunlight_exposure, self.time_step)

        self.get_global_targets(self.observer_satellites, self.target_satellites)

        rewards = self.process_actions(actions, simulator_type, agents) # careful with action/actions

        self.update_communication_timeline()
        self.update_global_observation_status_matrix(self.observer_satellites, self.target_satellites)

        # Move to the next timestep
        self.propagate_orbits()
        self.propagate_attitudes()
        self.time_step_number += 1

        # print(f"Time step number: {self.time_step_number}")
        done = self.is_terminated()
        
        return rewards, done



    def is_terminated(self):
        if self.global_observation_status_matrix.all() == 3:
            print(f"Simulation terminated because all observations were made at step {self.time_step_number}.")
            self.breaker = True

        if self.time_step_number >= self.num_steps:
            # print(f"Simulation reaches its duration limit at step {self.time_step_number}. Total duration: {self.duration} seconds.")
            self.breaker = True

        if self.breaker:
            self.total_time = time.time() - self.start_time
            # print(f"Simulation terminated after {self.total_time} seconds.")

        return self.breaker










############################################################################################################
# Different types of simulators
class CentralizedSimulator(Simulator):
    """
    Centralized simulator for the Gymnasium environment.
    Only 1 observer satellite (relay satellite) has communication with the rest of the observation satellites. 
    The observer satellites can only communicate with the relay.
    """
    def __init__(self, num_targets: int = 100, num_observers: int = 5, time_step: float = 1, duration: float = 24 * 60) -> None:
        super().__init__(num_targets, num_observers, time_step, duration)
        # set one random satellite to act as relay - it correspond to assign to the band the value of 5
        self.observer_satellites[random.randint(0, num_observers - 1)].commsys['band'] = 5




class DecentralizedSimulator(Simulator):
    """
    Everyone can talk with everyone
    Decentralized simulator for the Gymnasium environment.
    Each observer satellite can communicate with the rest of the observation satellites that share the same type of band communication.
    """
    def __init__(self, num_targets: int = 100, num_observers: int = 5, time_step: float = 1, duration: float = 24 * 60) -> None:
        super().__init__(num_targets, num_observers, time_step, duration)

        # self.observer_satellites[random.randint(0, num_observers - 1)].commsys['band'] = 1

        # All the bands are defined randomly in the Class Satellite in the satellites.py



class MixedSimulator(CentralizedSimulator, DecentralizedSimulator):
    """
    Mixed simulator for the Gymnasium environment.
    There are multiple observer satellites (relays) that can communicate with all the rest of the observation satellites.
    Each observer satellite can communicate with the rest of the observation satellites that share the same type of band communication.

    Different cases: decentralized with random band allocation, some 5 and some random, cnetralized with more 5 bands.
    """
    def __init__(self, num_targets: int = 100, num_observers: int = 5, time_step: float = 1, duration: float = 24 * 60) -> None:
        super().__init__(num_targets, num_observers, time_step, duration)

        # set one random number of observer satellites to act as relay - it correspond to assign to the band the value of 5
        # All the other band are defined in the Class Satellite in the satellites.py
        for i in random.randint(0, num_observers - 1):
            self.observer_satellites[random.randint(0, num_observers - 1)].commsys['band'] = 5

    """
    
    observer_satellite_1.commsys['band']==observer_satellite_2.commsys['band'] or 
    """