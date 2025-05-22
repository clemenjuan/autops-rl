import math
import random
import numpy as np
from abc import ABC

from src.subsystems.OpticPayload import OpticPayload
from src.subsystems.CommSubsystem import CommSubsystem



class Satellite(ABC):
    """
    Represents a satellite in Earth's orbit.
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
                'raan': random.uniform(0,180),
                'arg_of_perigee':0,
                'true_anomaly':random.uniform(0, 360),
                # add more orbital parameters as needed
            }

        #Electric Power Subsystem
        
        if epsys is None: 
            epsys = {
                'EnergyStorage': 84*3600,    #From Endurosat battery pack [W]*[s]=[J]
                'SolarPanelSize': 0.4*0.3, # deployable solar panels 12U solar panel area [m2]
                'EnergyAvailable': random.randint(20,70)*3600,
                'Efficiency': 0.3,  # Efficiency of the solar panels
                'SolarConstant': 1370 # W/m2, Solar constant
                # prod = area * efficiency * solarconstant
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
                'DataStorage': 8*64e9, # Maximum storage onboard. 8*64G[bytes]=[bites], from ISISpace bus
                'StorageAvailable': random.uniform(2,7) * 64e9, # Storage available for observation
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
            'angular_velocity': np.array([0.01, 0.01, 0.01]) # Initial angular velocity [rad/s]
            } 

        #Define availability of the satellite
        if availability is None:
            availability= 1
        
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
        Propagates the orbit of the satellite using the Kleper method.

        Args:
            time_step (float): The time step in seconds.

        Returns:
            tuple: A tuple containing the position and velocity vectors of the satellite.

        """
        if not isinstance(self.orbit, dict):
            raise TypeError(f"Expected orbit to be a dict, found {type(self.orbit)} instead.")

        a = self.orbit['semimajoraxis']  # the semimajor axis in meters
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
        v = (
            -n * r / math.sqrt(1 - e ** 2) * (math.cos(Omega) * math.sin(omega + theta) + math.sin(Omega) * math.cos(omega + theta) * math.cos(i)),
            -n * r / math.sqrt(1 - e ** 2) * (math.sin(Omega) * math.sin(omega + theta) - math.cos(Omega) * math.cos(omega + theta) * math.cos(i)), # sign changed
            n * r / math.sqrt(1 - e ** 2) * math.sin(i)
        )

        # update the orbit parameters
        self.orbit.update({
        'x': x,
        'y': y,
        'z': z,
        'vx': v[0],
        'vy': v[1],
        'vz': v[2],
        'radius': r,
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
        # print(f"Attitude quaternion: {quaternion}, angular velocity: {angular_velocity}")

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


    def distance_between(self, sat2, time_step):
        pos1 = np.array([self.orbit['x'], self.orbit['y'], self.orbit['z']])
        pos2 = np.array([sat2.orbit['x'], sat2.orbit['y'], sat2.orbit['z']])

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
        # print(f"Attitude quaternion: {quaternion}")

        # Convert quaternion to rotation matrix
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        # print(f"Rotation matrix:\n{rotation_matrix}")

        # Extract pointing direction (first column of rotation matrix if the default pointing is along x-axis)
        pointing_direction = rotation_matrix[:, 0]
        # print(f"Pointing direction: {pointing_direction}")

        return pointing_direction

    @staticmethod    
    def quaternion_to_rotation_matrix(quaternion):
        q = quaternion
        return np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
            [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
            [-2*q[0]*q[2] + 2*q[1]*q[3], 2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])
    
    def charge_battery(self, sunlight_exposure, time_step):
        # Charge the battery based on the solar panel efficiency and sunlight exposure
        # print(f"Solar panel size: {self.epsys['SolarPanelSize']:.2f} m2, Solar panel efficiency: {self.epsys['Efficiency']:.2f}, Solar constant: {self.epsys['SolarConstant']:.2f} W/m2, Sunlight exposure: {sunlight_exposure:.2f} W/m2, Time step: {time_step:.2f} s")
        energy_produced = self.epsys['SolarPanelSize'] * self.epsys['Efficiency'] * self.epsys['SolarConstant'] * sunlight_exposure * time_step
        self.epsys['EnergyAvailable'] = min(self.epsys['EnergyAvailable'] + energy_produced, self.epsys['EnergyStorage']) # Charge the battery up to the maximum capacity
        self.epsys['EnergyAvailable'] = max(self.epsys['EnergyAvailable'], 0)  # Ensure energy is within bounds
        assert self.epsys['EnergyAvailable'] <= self.epsys['EnergyStorage'], "Energy storage exceeded the maximum capacity."
        assert self.epsys['EnergyAvailable'] >= 0, "Energy storage cannot be negative."
        # print(f"Energy produced: {energy_produced:.2f} J, Energy available: {self.epsys['EnergyAvailable']:.2f} J")

    def get_sunlight_exposure(self):
        # Calculate the sunlight exposure
        # Get the Sun vector
        sun_vector = self.get_sun_vector()

        # Calculate the angle between the Sun vector and the satellite's pointing direction
        pointing_direction = self.calculate_pointing_direction()
        # print(f"Pointing direction: {pointing_direction}")
        cos_angle = np.dot(pointing_direction, sun_vector)
        angle = math.acos(cos_angle)

        # Check if the satellite is in the Earth's shadow
        if self.is_in_eclipse():
            return 0  # No sunlight exposure if in eclipse

        # Calculate sunlight exposure based on the angle
        # Assuming solar panels are perfectly aligned with the satellite's pointing direction
        exposure = max(0, cos_angle)  # Max ensures no negative values
        
        # print(f"Angle between Sun vector and pointing direction: {math.degrees(angle):.2f} degrees, Sunlight exposure: {exposure:.2f}")
        return exposure
    
    def is_in_eclipse(self):
        # Get the satellite's position
        sat_position = np.array([self.orbit['x'], self.orbit['y'], self.orbit['z']])
        # Calculate the Earth's shadow radius at the satellite's altitude
        earth_radius = 6371e3  # Earth radius in meters
        shadow_radius = earth_radius * (1 + self.orbit['semimajoraxis'] / (self.orbit['semimajoraxis'] + earth_radius))
        
        # Calculate the distance from the satellite to the center of the Earth
        distance_to_earth = np.linalg.norm(sat_position)

        # Check if the satellite is within the Earth's shadow
        return distance_to_earth < shadow_radius
    
    def get_sun_vector(self):
        # Get the satellite's position
        sat_position = np.array([self.orbit['x'], self.orbit['y'], self.orbit['z']])
        # Get the Sun's position
        sun_position = self.get_sun_position()
        # Calculate the vector from the satellite to the Sun
        sun_vector = sun_position - sat_position
        return sun_vector / np.linalg.norm(sun_vector)  # Normalize the vector
    
    def get_sun_position(self):
        # For simplicity, assume a fixed position in the inertial frame
        # Realistically, this should be updated based on the current date and time
        return np.array([1.496e+11, 0, 0])  # Example: Sun at 1 AU along the x-axis in meters
    


class TargetSatellite(Satellite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = "target"
        # Any other specific attributes/methods for the target satellite

class ObserverSatellite(Satellite):
    """
    Represents an observer satellite that can detect and observe target satellites.
    """
    def __init__(self, num_targets, num_observers, name="observer", *args, **kwargs):
        
        # Extract the observer index from the name (format: "Observer-{index}")
        try:
            # Parse the observer index from the name string
            if '-' in name:
                self.observer_index = int(name.split('-')[1]) - 1  # Subtract 1 to make it 0-indexed
            else:
                self.observer_index = 0  # Default index if name doesn't follow expected format
        except (IndexError, ValueError):
            self.observer_index = None  # Handle any parsing errors

        # Apply Walker delta formation if not explicitly provided        
        if 'orbit' not in kwargs or kwargs['orbit'] is None:
            # Set up Walker delta formation
            total_sats = num_observers
            num_planes = min(5, num_observers)  # Use at most 5 orbital planes
            sats_per_plane = math.ceil(total_sats / num_planes)
            
            # Calculate satellite index in the formation
            observer_index = kwargs.get('observer_index', 0)
            plane_idx = observer_index % num_planes
            position_in_plane = observer_index // num_planes
            
            # Base parameters for all satellites in formation
            semimajoraxis = random.uniform(7070, 7170)  # 700-800 km altitude
            inclination = 98.0  # Common inclination for the constellation
            eccentricity = 0.001  # Near-circular orbits
            
            # Calculate RAAN for each plane (evenly distributed)
            raan = (plane_idx * 360.0 / num_planes) % 360
            
            # Calculate argument of perigee
            arg_perigee = 0.0
            
            # Calculate mean anomaly with phase shift between planes
            phase_shift = 360.0 / (sats_per_plane * num_planes)
            relative_spacing = 1  # F parameter in Walker notation
            mean_anomaly = (position_in_plane * 360.0 / sats_per_plane + 
                            plane_idx * relative_spacing * phase_shift) % 360
            
            # Create orbit dictionary
            kwargs['orbit'] = {
                'x': 0, 'y': 0, 'z': 0, 'vx': 0, 'vy': 0, 'vz': 0, 'radius': 0,
                'mean_anomaly': mean_anomaly,
                'semimajoraxis': semimajoraxis,
                'inclination': inclination,
                'eccentricity': eccentricity,
                'raan': raan,
                'arg_of_perigee': arg_perigee,
                'true_anomaly': mean_anomaly  # Initialize true anomaly to mean anomaly
            }
            
        super().__init__(name=name, *args, **kwargs)
        self.num_observers = num_observers
        self.observation_status_matrix = np.zeros(num_targets, dtype=np.int32)  # 0: undetected, 1: detected, 2: being observed, 3: observed
        self.pointing_accuracy_matrix = np.zeros(num_targets, dtype=np.float32)  # pointing accuracy for each target
        self.communication_ability = np.zeros(num_observers, dtype=np.int32)  # Track communication status for each observer satellite
        self.observation_time_matrix = np.zeros(num_targets, dtype=np.float32)  # Total observation time for each target
        self.optic_payload = OpticPayload()
        self.max_distance = self.optic_payload.dist_detect()  # Assuming optic_payload is defined in Satellite
        self.has_new_data = np.zeros(num_observers, dtype=bool)  # Track new data status for each observer satellite
        self.communication_timeline_matrix = np.zeros(num_targets, dtype=np.int32)  # Track communication timeline
        self.is_processing = False  # Indicates if the satellite is currently processing
        self.processing_time = 0  # Time required for processing new data
        self.observation_counts = np.zeros(num_targets, dtype=np.int32)  # Track the number of observations (timesteps) for each target
        self.pointing_accuracy_avg = np.zeros((num_observers, num_targets), dtype=np.float32)  # Track average pointing accuracy for each target
        self.cumulative_pointing_accuracy = np.zeros((num_observers,num_targets), dtype=np.float32)  # Track cumulative pointing accuracy for each target
        self.max_pointing_accuracy_avg_sat = np.zeros(num_targets, dtype=np.float32)  # Track maximum pointing accuracy for each target
        # Add power consumption rates (in Watts)
        self.power_consumption_rates = { # Power consumption rates in Watts
            "standby": 7.5,  # Standby mode
            "communication": 9.3,  # During communication
            "observation": 18.806,  # During observation
        }
        self.storage_consumption_rates = {
            "observation": 1024*1024*8,  # Storage consumption rate during observation - 1 Mbits/s
            "communication": 0,  # Storage consumption rate during communication
        }
        self.current_power_consumption = 0  # Current power consumption

        self.contacts_matrix = np.zeros((num_observers, num_targets), dtype=np.int32)  # Current timestep contacted targets matrix
        self.contacts_matrix_acc = np.zeros((num_observers, num_targets), dtype=np.int32)  # Accumulated contacted targets matrix
        self.data_matrix = np.zeros((num_observers, num_observers), dtype=np.float32)  # Current timestep data exchange
        self.data_matrix_acc = np.zeros((num_observers, num_observers), dtype=np.float32)  # Accumulated data exchange
        self.adjacency_matrix = np.zeros((num_observers, num_observers), dtype=np.int32)  # Current timestep adjacency matrix
        self.adjacency_matrix_acc = np.eye((num_observers), dtype=np.int32)  # Accumulated adjacency matrix
        self.global_communication_counts = np.zeros((num_observers,num_observers), dtype=int)  # Global matrix to track the number of communications between each pair of satellites
        self.global_observation_counts = np.zeros((num_observers,num_targets), dtype=int)  # Global matrix to track the number of observations for each target

    def evaluate_pointing_accuracy(self, target_satellite, time_step):
        # Field of view limited to 10º
        # Evaluate pointing accuracy for each observer satellite with respect to each target satellite if they are in range. Otherwise return 0.
        distance = self.distance_between(target_satellite, time_step)
        # print(f"Distance between {self.name} and {target_satellite.name}: {distance:.2f} km")
        # print(f"Max distance: {self.max_distance:.2f} km")
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

            observer_to_target_vector = observer_to_target_vector / np.linalg.norm(observer_to_target_vector)
            # print(f"Observer-to-target vector: {observer_to_target_vector}")

            pointing_direction_norm = pointing_direction / np.linalg.norm(pointing_direction)
            # print(f"Pointing direction: {pointing_direction_norm}")

            cos_angle = np.dot(pointing_direction_norm, observer_to_target_vector)
            # print(f"Cosine of the angle: {cos_angle:.2f}")

            angular_distance = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            # print(f"Angular distance: {angular_distance:.2f} radians")

            angular_distance_deg = np.degrees(angular_distance)
            # print(f"Angular distance: {angular_distance_deg:.2f} degrees")

            if angular_distance_deg <= 5:
                normalized_pointing_accuracy = max(0, (5 - angular_distance_deg) / 5)
                return normalized_pointing_accuracy
            else:
                return 0
        else:
            return 0
        

    def get_targets(self, observer_index, target_satellites, time_step):
        for target_index, target_satellite in enumerate(target_satellites):
            pointing_accuracy = self.evaluate_pointing_accuracy(target_satellite, time_step) #, self.max_distance)
            self.pointing_accuracy_matrix[target_index] = pointing_accuracy
            if pointing_accuracy > 0:
                if self.observation_status_matrix[target_index] in [1, 2, 3]: # already detected, being observed or observed
                    self.update_contacts_matrix(observer_index, target_index) # Only mark as contacted this timestep
                else:
                    self.observation_status_matrix[target_index] = 1  # Mark as detected
                    # print(f"Observer {observer_index} detected target {target_index}")
                    self.has_new_data[:] = True  # Set flag to indicate new data
                    self.update_contacts_matrix(observer_index, target_index) # Mark as contacted this timestep
                
                # print(f"{self.name} has detected {target_satellite.name}")
        return self.contacts_matrix, self.contacts_matrix_acc

    def update_max_pointing_accuracy_avg_sat(self,index, target_index):
        self.max_pointing_accuracy_avg_sat[target_index] = np.maximum(self.pointing_accuracy_avg[index,target_index], self.max_pointing_accuracy_avg_sat[target_index])
        return self.max_pointing_accuracy_avg_sat
            
    def update_data_matrix(self, observer_index, other_observer_index, data_size):
        # Update data matrix
        self.data_matrix[observer_index,other_observer_index] += data_size
        self.data_matrix[other_observer_index,observer_index] += data_size
        self.data_matrix_acc[observer_index,other_observer_index] += data_size
        self.data_matrix_acc[other_observer_index,observer_index] += data_size

    def update_adjacency_matrix(self, observer_index, other_observer_index):
        # Update adjacency matrix
        # print(f"Updating adjacency between {observer_index} and {other_observer_index}")
        self.adjacency_matrix[observer_index, other_observer_index] = 1
        self.adjacency_matrix[other_observer_index, observer_index] = 1
        self.adjacency_matrix_acc[observer_index, other_observer_index] = 1
        self.adjacency_matrix_acc[other_observer_index, observer_index] = 1

    def update_contacts_matrix(self, observer_index, target_index):
        # Mark communication
        self.contacts_matrix[observer_index,target_index] = 1
        self.contacts_matrix_acc[observer_index,target_index] = 1

    def synchronize_contacts_matrix(self, index1, index2):
        self.contacts_matrix[index1] = np.maximum(self.contacts_matrix[index1], self.contacts_matrix[index2])
        self.contacts_matrix[index2] = self.contacts_matrix[index1]
        
        self.contacts_matrix_acc[index1] = np.maximum(self.contacts_matrix_acc[index1], self.contacts_matrix_acc[index2])
        self.contacts_matrix_acc[index2] = self.contacts_matrix_acc[index1]

    def check_and_update_processing_state(self, time_step):
        if self.is_processing or self.processing_time > 0:
            self.processing_time -= time_step
            self.availability = 0  # Set availability to 0 while processing
        if self.processing_time <= 0:
            self.is_processing = False
            self.availability = 1  # Set availability to 1 after processing is complete
            self.processing_time = 0

    def can_communicate(self,other_satellite_index):
        if self.is_processing:
            return False
        if not self.has_new_data[other_satellite_index]:
            return False
        return True

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
        self.communication_ability = np.zeros((self.num_observers, ), dtype=np.int8)
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
    def propagate_information(self, index, other_satellite, other_satellite_index, time_step, communication_type, reward_step, steps, communication_done, data_transmitted, data_to_transmit):
        result = {
            "successful": False,
            "data_transmitted": 0,
            "communication_possible": False,
            "can_communicate": False
        }
        
        if self.can_communicate(other_satellite_index) and data_to_transmit > 0:
            result["communication_possible"] = True
            
            # Determine communication capability based on the specified type
            can_communicate = False
            if communication_type == 'centralized':
                can_communicate = self.can_communicate_with_centralized(other_satellite, time_step)
            elif communication_type == 'decentralized':
                can_communicate = self.can_communicate_with_decentralized(other_satellite, time_step)
            elif communication_type == 'everyone':
                can_communicate = self.can_communicate_with_everyone(other_satellite, time_step)
            else:
                raise ValueError("Invalid communication type. Choose from 'centralized', 'decentralized', or 'everyone'.")
            
            result["can_communicate"] = can_communicate
            
            if can_communicate:
                effective_data_rate = self.calculate_data_volume(other_satellite, time_step)
                if effective_data_rate <= 0:
                    communication_done = True
                    steps += 1
                    return result, communication_done, steps, self.contacts_matrix, self.contacts_matrix_acc, self.adjacency_matrix, self.adjacency_matrix_acc, self.data_matrix, self.data_matrix_acc, self.global_communication_counts, self.global_observation_counts, self.max_pointing_accuracy_avg_sat, data_transmitted
                
                volume_of_data = min(effective_data_rate, data_to_transmit)
                data_transmitted += volume_of_data
                result["data_transmitted"] = volume_of_data
                
                self.update_data_matrix(index, other_satellite_index, volume_of_data)
                
                if data_transmitted >= data_to_transmit:
                    self.update_adjacency_matrix(index, other_satellite_index)
                    self.synchronize_contacts_matrix(index, other_satellite_index)
                    self.global_communication_counts[index, other_satellite_index] += 1
                    self.global_observation_counts = np.maximum(self.global_observation_counts, other_satellite.global_observation_counts)
                    other_satellite.update_processing_status(self.observation_status_matrix)
                    
                    for i in range(len(self.has_new_data)):
                        if self.has_new_data[i] and not other_satellite.has_new_data[i]:
                            other_satellite.has_new_data[i] = True
                        if not self.has_new_data[i] and other_satellite.has_new_data[i]:
                            self.has_new_data[i] = True
                    
                    self.has_new_data[other_satellite_index] = False
                    other_satellite.has_new_data[index] = False
                    
                    result["successful"] = True
                    communication_done = True
            else:
                communication_done = True
        else:
            communication_done = True
        
        steps += 1
        return result, communication_done, steps, self.contacts_matrix, self.contacts_matrix_acc, self.adjacency_matrix, self.adjacency_matrix_acc, self.data_matrix, self.data_matrix_acc, self.global_communication_counts, self.global_observation_counts, self.max_pointing_accuracy_avg_sat, data_transmitted
    
    # action >= 2: Observe target
    def observe_target(self, index, target, target_index, time_step, reward_step, steps=0):
        result = {
            "successful": False,
            "pointing_accuracy": 0,
            "is_processing": self.is_processing,
            "already_observed": False,
            "out_of_range": False
        }
        
        if target_index < len(self.observation_status_matrix):
            if self.is_processing:
                # Just set status, reward will be calculated by simulator
                pass
            else:
                distance = self.distance_between(target, time_step)
                pointing_accuracy = self.evaluate_pointing_accuracy(target, time_step)
                
                if distance <= self.max_distance and pointing_accuracy > 0:
                    if self.observation_status_matrix[target_index] == 3:
                        result["already_observed"] = True
                    else:
                        self.cumulative_pointing_accuracy[index, target_index] += pointing_accuracy
                        self.observation_counts[target_index] += 1
                        self.observation_status_matrix[target_index] = 2  # Mark as being observed
                        self.has_new_data[:] = True  # Set flag to indicate new data
                        self.update_contacts_matrix(index, target_index)
                        self.observation_time_matrix[target_index] += time_step  # Assuming time_step is in seconds
                        
                        result["pointing_accuracy"] = pointing_accuracy
                        self.DataHand['StorageAvailable'] -= self.storage_consumption_rates["observation"] * time_step
                else:
                    result["out_of_range"] = True
                    
                # Check if observation is complete
                if (self.observation_counts[target_index] > 0 and 
                    self.observation_status_matrix[target_index] == 2 and 
                    self.cumulative_pointing_accuracy[index, target_index] > 0):
                    
                    self.pointing_accuracy_avg[index, target_index] = self.cumulative_pointing_accuracy[index, target_index] / self.observation_counts[target_index]
                    if self.pointing_accuracy_avg[index, target_index] > 0:
                        self.global_observation_counts[index, target_index] += 1
                        self.update_max_pointing_accuracy_avg_sat(index, target_index)
                        self.observation_status_matrix[target_index] = 3  # Mark as observed
                        self.has_new_data[:] = True
                        
                        # Set success flag
                        result["successful"] = True
                    else:
                        raise ValueError("Error: Pointing accuracy average is zero")
        
        steps += 1
        return result, steps, self.contacts_matrix, self.contacts_matrix_acc, self.adjacency_matrix, self.adjacency_matrix_acc, self.data_matrix, self.data_matrix_acc, self.global_observation_counts, self.max_pointing_accuracy_avg_sat

# Example usage of the updated class (main function or simulation setup)
if __name__ == "__main__":
    satellite = Satellite()
    for _ in range(10):  # Simulate 10 time steps
        satellite.propagate_orbit(1)
        satellite.propagate_attitude(1)
        sunlight_exposure = satellite.get_sunlight_exposure()
        satellite.charge_battery(sunlight_exposure, 1)