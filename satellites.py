import math
import random
import numpy as np
from typing import Any

from gymnasium import spaces
from abc import ABC, abstractmethod

from OpticPayload import OpticPayload
from CommSubsystem import CommSubsystem
from simulator import Simulator, CentralizedSimulator, MixedSimulator, DecentralizedSimulator



class Satellite(ABC):
    """
    Represents a satellite object.

    Attributes:
        orbit (dict): Dictionary containing orbital parameters of the satellite.
        epsys (dict): Dictionary containing electric power subsystem parameters of the satellite.
        commsys (dict): Dictionary containing communication subsystem parameters of the satellite.
        DataHand (dict): Dictionary containing data handling parameters of the satellite.
        PropSys (dict): Dictionary containing propulsion system parameters of the satellite.
        Optic (dict): Dictionary containing optical payload parameters of the satellite.
        category (dict): Dictionary containing category parameters of the satellite.
        name (dict): Dictionary containing name parameters of the satellite.
        attitude (dict): Dictionary containing attitude parameters of the satellite.
        availability (dict): Dictionary containing availability parameters of the satellite.
        instrumentation (dict): Dictionary containing instrumentation parameters of the satellite.
        RewardFunction (dict): Dictionary containing reward function parameters of the satellite.
        data_comms (CommSubsystem): Communication subsystem object.
        processing_time (int): Processing time for the satellite after receiving information (in seconds).
    """
    def __init__(self, orbit=None, epsys=None, commsys=None, DataHand=None, PropSys=None, Optic=None, category=None, name=None, attitude=None, availability=None, instrumentation=None, RewardFunction=None):
        # Sun-synchronous parameters

        semimajoraxis_sso = random.uniform(7070, 7170)  # Corresponding to 700-800 km altitudes
        J2 = 1.08263e-3  # Second zonal harmonic of Earth's gravitational potential
        omega_E = 7.2921159e-5  # rad/s, Earth's rotation rate
        R_Earth = 6378.137  # km, Mean Earth radius
        # Calculate the mean motion for an equatorial orbit
        #n_eq = math.sqrt(398600.5 / semimajoraxis_sso**3)  # sqrt(mu/a^3), where mu is Earth's gravitational parameter
        RAAN_ChangeSSO=0.986

        # Calculate inclination for SSO
        #cos_i = (-2 * math.pi * semimajoraxis_sso**3 * n_eq / (J2 * omega_E * R_Earth**2))**(1/7)
        cos_i = RAAN_ChangeSSO/(-9.96*(R_Earth/semimajoraxis_sso)**(7/2)) 
        inclination = math.acos(cos_i) * 180 / math.pi  # Convert from radians to degrees
        #inclination=98.2
        #print(inclination)
        
        if orbit is None:
            orbit = {
                'x': 0,
                'y': 0,
                'z': 0,
                'vx': 0,
                'vy': 0,
                'vz': 0,
                'radius': 0,
                'mean_anomaly': 0,
                'semimajoraxis': semimajoraxis_sso,  # km
                'inclination': inclination,  # degrees
                'eccentricity': random.uniform(0, 0.001),
                'raan': 0,
                'arg_of_perigee':0,
                'true_anomaly':random.uniform(0, 360),
                # add more orbital parameters as needed
            }

        #Electric Power Subsystem
        
        if epsys is None: 
            epsys = {
                'EnergyStorage': 84*3600,    #From Endurosat battery pack [W]*[s]=[J]
                'SolarPanelSize': 0.4*0.3,
                'EnergyAvailable': 10*3600,
                # add more subsystems as needed

            }

        # Comm Subsystem

        if commsys is None:
            commsys= {
                #Random communication system on board
                # 1: UHF
                # 2: VHF
                # 3: S-band
                # 4: X-band
                # 5: All bands
                'band': random.randint(1,4)
            }

        #Data Handling

        if DataHand is None:
            DataHand = {
                'DataStorage': 8*64e9, # Maximum storage onboard. 8*[bytes]=[bites], from ISISpace bus
                'StorageAvailable': 1*64e9, # Storage available for observation
                'DataSize': 52, # Data package size per satellite in bytes
            }
        
        # Propulsion System

        if PropSys is None:
            PropSys = {
                'PropellantMass': 1, # Maximum propellant onboard. [kg]
                'PropulsionType': 0,  #This value is a flag to define if we have chemical or electrical propulsion
                'SpecificImpulse': 250, #Specific impulse of the propulsion system in [s]
                'Thrust': 1, # Thrust of the propulsion system in [N]
            }

        # Optical Payload 
        
        if Optic is None:
            Optic = {
                'ApertureDiameter': 0.09, # Aperture diameter [m]
                'Wavelength': 700e-9, #Max wavelength of observation
            }
        
        # Index for define if the satellite is an observer or a target or something else
        if category is None:
            category = {
                'Target': 0, # Index for target to be identified
                'Observation': 1, #Index for observation satellite
            }

        # Index for define if the satellite is an observer or a target or something else
        if name is None:
            name = {
                'Name':0,
                
            }
    	# Attitude parameters
        if attitude is None:
            attitude= {
            'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),  # Initial quaternion [scalar, i, j, k]
            'angular_velocity': np.array([0.0, 0.0, 0.0]),  # Initial angular velocity [rad/s]
            } 

        #Define availability of the satellite
        if availability is None:
            availability= {
            'availability': 1,   # use 1 for available and 0 for not available
            } 
        
        if instrumentation is None:
            instrumentation= {
            'cameraVIS': 1,
            'cameraIR': 1,
            'radar': 1,
            'GNSS': 1,
            } 

        #Reward function RewardFunction

        if RewardFunction is None:
            RewardFunction= {
            'RFPointing': 0,  # Reward function related to pointing 
            'RFPower': 0,  # Reward Function related to the power available
            'RFAvailability': 0, # Reward Function related to the availability of the satellite 
            'RFInstrumentation': 0, # Reward Function related to the Instrumentation on board 
            'RFTimeObservation': 0, # Reward Function related to the time observation 
            'RFNumberObject': 0, # Reward Function related to the number of objects that can be observed
            'RFComparison': 0, # Reward Function related to the number of satellite with which the RF has been compared 
            }
   
# =============================================================================
#         if flag is None:
#             flag = random.choice([True, False])
# =============================================================================
        self.orbit = orbit
        self.orbit = self.propagate_orbit(0)[0] # Propagate the orbit to get the initial position and velocity
        self.epsys = epsys
        self.commsys = commsys
        self.DataHand = DataHand
        self.PropSys = PropSys
        self.Optic = Optic
        self.category = category
        self.name = name
        self.attitude = attitude
        self.availability = availability
        self.instrumentation = instrumentation
        self.RewardFunction = RewardFunction
        self.data_comms = CommSubsystem()
        # Introduce a processing time for a satellite after it receives information
        self.processing_time = 0 # You can adjust this value based on your requirements (in seconds)
        # self.sim_instance = Simulator()

    def propagate_orbit(self, time_step):
        """
        Propagates the orbit of the satellite using the Euler method.

        Args:
            time_step (float): The time step in seconds.

        Returns:
            tuple: A tuple containing the position and velocity vectors of the satellite.

        """
        if not isinstance(self.orbit, dict):
            raise TypeError(f"Expected orbit to be a dict, found {type(self.orbit)} instead.")

        a = self.orbit['semimajoraxis'] * 1000  # the semimajor axis in meters
        e = self.orbit['eccentricity']
        i = math.radians(self.orbit['inclination'])
        omega = math.radians(self.orbit['arg_of_perigee'])
        Omega = math.radians(self.orbit['raan'])
        theta = math.radians(self.orbit['true_anomaly'])

        # calculate the mean motion of the satellite in radians per second
        n = math.sqrt(3.986004418e14 / a ** 3)  # the mean motion in radians per second

        # calculate the eccentric anomaly using Kepler's equation
        E = math.atan2(math.sqrt(1 - e ** 2) * math.sin(theta), e + math.cos(theta))
        M = E - e * math.sin(E)

        # calculate the true anomaly and the radius vector
        E_dot = n / (1 - e * math.cos(E))
        theta_dot = math.sqrt(1 - e ** 2) * E_dot
        theta += theta_dot * time_step
        r = a * (1 - e ** 2) / (1 + e * math.cos(theta))

        # calculate the position and velocity vectors of the satellite in the inertial frame
        x = r * (math.cos(Omega) * math.cos(omega + theta) - math.sin(Omega) * math.sin(omega + theta) * math.cos(i))
        y = r * (math.sin(Omega) * math.cos(omega + theta) + math.cos(Omega) * math.sin(omega + theta) * math.cos(i))
        z = r * math.sin(omega + theta) * math.sin(i)
        v = (-n * r / math.sqrt(1 - e ** 2) * (
                    math.cos(Omega) * math.sin(omega + theta) + math.sin(Omega) * math.cos(omega + theta) * math.cos(
                i)),
             n * r / math.sqrt(1 - e ** 2) * (math.sin(Omega) * math.sin(omega + theta) - math.cos(Omega) * math.cos(
                 omega + theta) * math.cos(i)),
             n * r / math.sqrt(1 - e ** 2) * math.sin(i))

        # update the orbit parameters
        self.orbit.update({
        'x': x,
        'y': y,
        'z': z,
        'vx': v[0],
        'vy': v[1],
        'vz': v[2],
        'radius': r / 1000,
        'mean_anomaly': math.degrees(M),
        'true_anomaly': math.degrees(theta),
        })

        return self.orbit,(x, y, z), v  # return the position and velocity vectors of the satellite

        
    def propagate_attitude(self, time_step):
        """
        Propagates the attitude quaternion of the satellite using quaternion kinematics.

        Parameters:
        - time_step (float): The time step for the propagation.

        Returns:
        None
        """

        # Propagate the attitude quaternion using quaternion kinematics
        quaternion = self.attitude['quaternion']
        angular_velocity = self.attitude['angular_velocity']

        # Update quaternion using quaternion kinematics (Euler integration)
        q_dot = 0.5 * np.array([
            -quaternion[1] * angular_velocity[0] - quaternion[2] * angular_velocity[1] - quaternion[3] * angular_velocity[2],
             quaternion[0] * angular_velocity[0] + quaternion[2] * angular_velocity[2] - quaternion[3] * angular_velocity[1],
             quaternion[0] * angular_velocity[1] - quaternion[1] * angular_velocity[2] + quaternion[3] * angular_velocity[0],
             quaternion[0] * angular_velocity[2] + quaternion[1] * angular_velocity[1] - quaternion[2] * angular_velocity[0],
        ])
        quaternion += q_dot * time_step

        # Normalize quaternion to maintain unit length
        quaternion /= np.linalg.norm(quaternion)

        # Update attitude parameters
        self.attitude['quaternion'] = quaternion

    def distance_between(sat1, sat2, time_step):
        # propagate both satellites to the same time
        orbit1,pos1, vel1 = sat1.propagate_orbit(time_step)
        orbit2,pos2, vel2 = sat2.propagate_orbit(time_step)

        # calculate the distance between the two satellites
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # print(distance)
        return distance
    

    def calculate_pointing_direction(self):
        # Calculate pointing direction from the attitude quaternion
        quaternion = self.attitude['quaternion']

        # Convert quaternion to rotation matrix
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)

        # Extract pointing direction (third column of rotation matrix)
        pointing_direction = rotation_matrix[:, 2]

        return pointing_direction

    @staticmethod    
    def quaternion_to_rotation_matrix(quaternion):
        q = quaternion
        return np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
            [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
            [-2*q[0]*q[2] + 2*q[1]*q[3], 2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])



class TargetSatellite(Satellite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = "target"
        # Any other specific attributes/methods for the target satellite

class ObserverSatellite(Satellite):
    """
    Represents an observer satellite that can detect and observe target satellites.

    Attributes:
        name (str): The name of the observer satellite.
        num_targets (int): The number of target satellites.
        num_observers (int): The number of observer satellites.
        commsys (dict): The communication system configuration.
        num_observers (int): The number of observer satellites.
        observation_status_matrix (ndarray): The matrix to track the observation status of each target satellite.
        pointing_accuracy_matrix (ndarray): The matrix to track the pointing accuracy for each target satellite.
        observation_time_matrix (ndarray): The matrix to track the total observation time for each target satellite.
        optic_payload (OpticPayload): The optic payload of the observer satellite.
        max_distance (float): The maximum distance for detection based on the optic payload.
        has_new_data (ndarray): The flag to track new data status for each observer satellite.
        communication_timeline_matrix (ndarray): The matrix to track the communication timeline.
        is_processing (bool): Indicates if the satellite is currently processing.
        processing_time (int): The time required for processing new data.
        observation_counts (ndarray): The matrix to track the number of observations (timesteps) for each target satellite.
        cumulative_pointing_accuracy (ndarray): The matrix to track the cumulative pointing accuracy for each target satellite.
        power_consumption_rates (dict): The power consumption rates for different modes.
        current_power_consumption (int): The current power consumption.

    Methods:
        evaluate_pointing_accuracy(target_satellite, time_step): Evaluates the pointing accuracy for a target satellite.
        get_targets(target_satellites, time_step): Gets the target satellites and updates the observation status.
        check_and_update_processing_state(time_step): Checks and updates the processing state of the satellite.
        can_communicate(): Checks if the satellite can communicate.
        can_communicate_with_centralized(other, time_step): Checks if the satellite can communicate with another satellite in a centralized communication model.
        can_communicate_with_decentralized(other, time_step): Checks if the satellite can communicate with another satellite in a decentralized communication model.
        can_communicate_with_everyone(other, time_step): Checks if the satellite can communicate with another satellite in a fully decentralized communication model.
        update_processing_status(observation_status_matrix): Updates the observation status based on received information.
        calculate_data_volume(other, time_step): Calculates the data volume to be transmitted to another satellite.
        stand_by(): Performs the stand-by action.
        propagate_information(other_satellite, time_step, communication_type, reward, steps, communication_done, data_transmitted, data_to_transmit): Propagates information to other satellites based on communication capabilities and new data availability.
    """
    def __init__(self, num_targets, num_observers, name="observer"):
        super().__init__(name = name)
        self.num_observers = num_observers
        self.observation_status_matrix = np.zeros(num_targets, dtype=int)  # 0: undetected, 1: detected, 2: being observed, 3: observed
        self.pointing_accuracy_matrix = np.zeros(num_targets, dtype=float)  # pointing accuracy for each target
        self.communication_ability = np.zeros(num_observers, dtype=int)  # Track communication status for each observer satellite
        self.observation_time_matrix = np.zeros(num_targets, dtype=float)  # Total observation time for each target
        self.optic_payload = OpticPayload()
        self.max_distance = self.optic_payload.dist_detect() / 1000  # Assuming optic_payload is defined in Satellite
        self.has_new_data = np.zeros(num_observers, dtype=bool)  # Track new data status for each observer satellite
        self.communication_timeline_matrix = np.zeros(num_targets, dtype=int)  # Track communication timeline
        self.is_processing = False  # Indicates if the satellite is currently processing
        self.processing_time = 0  # Time required for processing new data
        self.observation_counts = np.zeros(num_targets, dtype=int)  # Track the number of observations (timesteps) for each target
        self.cumulative_pointing_accuracy = np.zeros(num_targets, dtype=float)  # Track cumulative pointing accuracy for each target
        # Add power consumption rates (in Watts)
        self.power_consumption_rates = {
            "standby": 7,  # Standby mode
            "communication": 10,  # During communication
            "observation": 10,  # During observation
        }
        self.storage_consumption_rates = {
            "observation": 0.1,  # Storage consumption rate during observation
            "communication": 0.1,  # Storage consumption rate during communication
        }
        self.current_power_consumption = 0  # Current power consumption

        self.contacts_matrix = np.zeros((num_observers, num_targets), dtype=int)  # Current timestep contacted targets matrix
        self.contacts_matrix_acc = np.zeros((num_observers, num_targets), dtype=int)  # Accumulated contacted targets matrix
        self.data_matrix = np.zeros((num_observers, num_observers), dtype=float)  # Current timestep data exchange
        self.data_matrix_acc = np.zeros((num_observers, num_observers), dtype=float)  # Accumulated data exchange
        self.adjacency_matrix = np.zeros((num_observers, num_observers), dtype=int)  # Current timestep adjacency matrix
        self.adjacency_matrix_acc = np.zeros((num_observers, num_observers), dtype=int)  # Accumulated adjacency matrix
        self.global_observation_counts = np.zeros(num_targets, dtype=int)  # Global matrix to track the number of observations for each target

    def evaluate_pointing_accuracy(self, target_satellite,time_step):
        # Evaluate pointing accuracy for each observer satellite with respect to each target satellite if they are in range. Otherwise returne None
        distance = self.distance_between(target_satellite, time_step)

        if distance < self.max_distance:

            # Calculate the pointing direction of the observer satellite
            pointing_direction = self.calculate_pointing_direction()

            # Get the position vector of the target satellite
            target_position = np.array([
                target_satellite.orbit['x'],
                target_satellite.orbit['y'],
                target_satellite.orbit['z']
            ])

            # Calculate the vector pointing from observer to target
            observer_to_target_vector = target_position - np.array([
                self.orbit['x'],
                self.orbit['y'],
                self.orbit['z']
            ])

            # Normalize the pointing direction vector and observer-to-target vector
            pointing_direction_norm = pointing_direction / np.linalg.norm(pointing_direction)
            observer_to_target_norm = observer_to_target_vector / np.linalg.norm(observer_to_target_vector)

            # Calculate the cosine of the angle between the pointing direction and observer-to-target vector
            cos_angle = np.dot(pointing_direction_norm, observer_to_target_norm)

            # Calculate the angular distance (in radians) using the arccosine of the cosine of the angle
            angular_distance = np.arccos(cos_angle)

            # Convert angular distance from radians to degrees
            angular_distance_deg = np.degrees(angular_distance)

            # Print the angular distance as the pointing accuracy for the observer satellite
            # print(f"Observer satellite {observer_satellite.name} sees target satellite {target_satellite.name} with a "
            #    f"Pointing accuracy (angular distance) to target: {angular_distance_deg:.2f} degrees")
            
            return angular_distance_deg
        else:
            # print(f"Target {target_satellite.name} is out of range for observer {observer_satellite.name}")
            return None
        

    def get_targets(self, observer_index, target_satellites, time_step):
        for target_index, target_satellite in enumerate(target_satellites):
            pointing_accuracy = self.evaluate_pointing_accuracy(target_satellite, time_step) #, self.max_distance)
            self.pointing_accuracy_matrix[target_index] = pointing_accuracy
            if pointing_accuracy is not None:
                if self.observation_status_matrix[target_index] in [1, 2, 3]: # already detected, being observed or observed
                    self.update_contacts_matrix(observer_index, target_index) # Only mark as contacted this timestep
                else:
                    self.observation_status_matrix[target_index] = 1  # Mark as detected
                    self.has_new_data[:] = True  # Set flag to indicate new data
                    self.update_contacts_matrix(observer_index, target_index) # Mark as contacted this timestep
        return self.contacts_matrix, self.contacts_matrix_acc

            
    def update_data_matrix(self, observer_index, other_observer_index, data_size):
        # Update data matrix
        self.data_matrix[observer_index][other_observer_index] += data_size
        self.data_matrix[other_observer_index][observer_index] += data_size
        self.data_matrix_acc[observer_index][other_observer_index] += data_size
        self.data_matrix_acc[other_observer_index][observer_index] += data_size
        # return self.data_matrix, self.data_matrix_acc

    def update_adjacency_matrix(self, observer_index, other_observer_index):
        # Update adjacency matrix
        self.adjacency_matrix[observer_index][other_observer_index] = 1
        self.adjacency_matrix[other_observer_index][observer_index] = 1
        self.adjacency_matrix_acc[observer_index][other_observer_index] = 1
        self.adjacency_matrix_acc[other_observer_index][observer_index] = 1
        # return self.adjacency_matrix, self.adjacency_matrix_acc

    def update_contacts_matrix(self, observer_index, target_index):
        # Mark communication
        self.contacts_matrix[observer_index][target_index] = 1
        self.contacts_matrix_acc[observer_index][target_index] = 1
        # return self.contacts_matrix, self.contacts_matrix_acc


    def synchronize_contacts_matrix(self, index1, index2):
        self.contacts_matrix[index1] = np.maximum(self.contacts_matrix[index1], self.contacts_matrix[index2])
        self.contacts_matrix[index2] = self.contacts_matrix[index1]  # Both rows now reflect the union of connections
        self.contacts_matrix_acc[index1] = np.maximum(self.contacts_matrix_acc[index1], self.contacts_matrix_acc[index2])
        self.contacts_matrix_acc[index2] = self.contacts_matrix_acc[index1]  # Both rows now reflect the union of connections
        # return self.contacts_matrix, self.contacts_matrix_acc


    def check_and_update_processing_state(self, time_step):
        if self.is_processing or self.processing_time > 0:
            self.processing_time -= time_step
            self.availability = 0  # Set availability to 0 while processing
        if self.processing_time <= 0:
            self.is_processing = False
            self.availability = 1  # Set availability to 1 after processing is complete
            self.processing_time = 0

    def can_communicate(self,other_satellite_index):
        return not self.is_processing and self.has_new_data[other_satellite_index]

    def can_communicate_with_centralized(self, other, time_step):
        dist = self.distance_between(other, time_step)
        eff_datarate = self.data_comms.calculateEffectiveDataRate(dist)
        
        # Check for communication capability based on effective data rate and whether any satellite acts as a relay
        if eff_datarate > 0 and not self.is_processing and not other.is_processing and (self.commsys['band'] == 5 or other.commsys['band'] == 5):
            return True
        return False
    
    def can_communicate_with_decentralized(self, other, time_step):
        dist = self.distance_between(other, time_step)
        eff_datarate = self.data_comms.calculateEffectiveDataRate(dist)
        
        # Check for communication capability based on effective data rate and matching communication bands
        # In a decentralized communication model, satellites can communicate if they share the same band
        # and are not currently processing other communications.
        if eff_datarate > 0 and not self.is_processing and not other.is_processing and self.commsys['band'] == other.commsys['band']:
            return True
        return False
    
    def can_communicate_with_everyone(self, other, time_step):
        dist = self.distance_between(other, time_step)
        eff_datarate = self.data_comms.calculateEffectiveDataRate(dist)

        # Fully decentralized communication model where all satellites can communicate with each other
        if eff_datarate > 0 and not self.is_processing and not other.is_processing:
            return True
        return False

    def get_communication_ability(self, observer_satellites, time_step, communication_type):
        """
        Get the communication vector for the observer satellite based on the communication model.
        The communication vector indicates which observer satellites can communicate with the current satellite.
        """
        for i, other_observer in enumerate(observer_satellites):
            if self != other_observer and self.can_communicate(i):
                can_communicate = False

                # Determine communication capability based on the specified type
                if communication_type == 'centralized':
                    can_communicate = self.can_communicate_with_centralized(other_observer, time_step)
                elif communication_type == 'decentralized':
                    can_communicate = self.can_communicate_with_decentralized(other_observer, time_step)
                elif communication_type == 'everyone':
                    can_communicate = self.can_communicate_with_everyone(other_observer, time_step)
                else :
                    raise ValueError("Invalid communication type. Choose from 'centralized', 'decentralized', or 'everyone'.")

                if can_communicate:
                    self.communication_ability[i] = 1
         
        return self.communication_ability


    def update_processing_status(self, observation_status_matrix):
        """
        Update the satellite's observation data based on received information,
        assuming data comes in a structured form.
        """
        self.observation_status_matrix = np.maximum(self.observation_status_matrix, observation_status_matrix)
        # self.observation_time_matrix = np.maximum(self.observation_time_matrix, observation_time_matrix)
        # self.pointing_accuracy_matrix = np.maximum(self.pointing_accuracy_matrix, pointing_accuracy_matrix)
        # Process new data
        self.is_processing = True
        self.processing_time += 20  # Set processing time to 20 seconds

    def calculate_data_volume(self, other, time_step):
        distance = self.distance_between(other, time_step)
        eff_datarate = self.data_comms.calculateEffectiveDataRate(distance)
        return eff_datarate * time_step

    # action = 0: Stand-by
    def stand_by(self):
        # Logic for standing by, e.g., energy recovery or simply passing time
        pass

    # action = 1: Propagate information
    def propagate_information(self,index, other_satellite, other_satellite_index, time_step, communication_type, reward_step, steps, communication_done, data_transmitted, data_to_transmit):
        """
        Propagate information to other satellites based on communication capabilities and new data availability.
        This method combines the logic of sharing data across centralized, decentralized, and fully open communication models.
        """
        if self != other_satellite and self.can_communicate(other_satellite_index):
            can_communicate = False
            volume_of_data = 0

            # Determine communication capability based on the specified type
            if communication_type == 'centralized':
                can_communicate = self.can_communicate_with_centralized(other_satellite, time_step)
            elif communication_type == 'decentralized':
                can_communicate = self.can_communicate_with_decentralized(other_satellite, time_step)
            elif communication_type == 'everyone':
                can_communicate = self.can_communicate_with_everyone(other_satellite, time_step)
            else :
                raise ValueError("Invalid communication type. Choose from 'centralized', 'decentralized', or 'everyone'.")
            
            if can_communicate:
                # Calculate the volume of data exchanged based on effective data rate and time step
                volume_of_data = self.calculate_data_volume(other_satellite, time_step)
                data_transmitted += volume_of_data
                # Update data matrices in the Simulator
                self.update_data_matrix(index, other_satellite_index, volume_of_data)
                reward_step += 1 # Reward for successful communication
                
                # if data transmitted is already enough, then information has been correctly propagated
                if data_transmitted >= data_to_transmit:
                    self.update_adjacency_matrix(index, other_satellite_index)

                    # Fusion both contacts_matrix from both satellites
                    self.synchronize_contacts_matrix(index, other_satellite_index)

                    # Fusion global observation counts
                    self.global_observation_counts = np.maximum(self.global_observation_counts, other_satellite.global_observation_counts)

                    # Share observation data
                    other_satellite.update_processing_status(self.observation_status_matrix)
                    
                    # Mark data as successfully shared and reset the flag
                    self.has_new_data[other_satellite_index] = False
                    reward_step += 1000  # Reward for successful complete communication
                    communication_done = True
            else:
                reward_step -= 100 # Penalty for failed (other not available) or incomplete communication
                communication_done = True
        else:
            reward_step -= 1000  # Penalty for failed communication (not available or same satellite)
            communication_done = True
        steps += 1
        return reward_step,communication_done, steps, self.contacts_matrix, self.contacts_matrix_acc, self.adjacency_matrix, self.adjacency_matrix_acc, self.data_matrix, self.data_matrix_acc, self.global_observation_counts

    # action >= 2: Observe target
    def observe_target(self, index, target, target_index, time_step, reward_step, steps=0, observation_done=False):
        if target_index <= len(self.observation_status_matrix):
            if self.is_processing:
                # Implement penalty for trying to observe while processing
                reward_step -= 1000
                observation_done = True
            else:
                pointing_accuracy = self.evaluate_pointing_accuracy(target, time_step)
                if pointing_accuracy is not None:
                    if self.observation_status_matrix[target_index] == 3:
                        # implement penalty for trying to observe an already observed target
                        reward_step -= 100
                        observation_done = True
                    # Update cumulative pointing accuracy and counts
                    self.cumulative_pointing_accuracy[target_index] += pointing_accuracy
                    self.observation_counts[target_index] += 1
                    self.observation_status_matrix[target_index] = 2  # Mark as being observed
                    self.has_new_data[:] = True  # Set flag to indicate new data
                    self.update_contacts_matrix(index, target_index)
                    # Update observation time
                    self.observation_time_matrix[target_index] += time_step  # Assuming time_step is in seconds
                else:
                    if self.observation_counts[target_index] > 0:
                        # just finished observing the target
                        self.observation_status_matrix[target_index] = 3  # Mark as observed
                        self.has_new_data[:] = True
                        self.global_observation_counts[target_index] += 1  # Update global observation matrix
                        observation_done = True
                        reward_step += 1000*self.cumulative_pointing_accuracy[target_index]  # Reward for successful observation
                    else:
                        # implement penalty for not observing the target (out of range)
                        reward_step -= 100
                        observation_done = True
        steps += 1
        return reward_step, observation_done, steps, self.contacts_matrix, self.contacts_matrix_acc, self.adjacency_matrix, self.adjacency_matrix_acc, self.data_matrix, self.data_matrix_acc, self.global_observation_counts
                    
