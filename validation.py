import unittest
import os
import time
import numpy as np
from simulator import Simulator
from satellites import ObserverSatellite, TargetSatellite
from astropy import units # as u
from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import random
from FSS_env import FSS_env



class TestSatelliteSimulation(unittest.TestCase):
    
    def setUp(self):
        # Use consistent initial conditions for all satellites
        initial_orbit = {
            'semimajoraxis': 7150000,  # m
            'eccentricity': 0.001,
            'inclination': 98.7,  # degrees
            'raan': 0,
            'arg_of_perigee': 0,
            'true_anomaly': 0,
        }

        initial_orbit1 = {
            'semimajoraxis': 7150000,  # m
            'eccentricity': 0.001,
            'inclination': 98.7,  # degrees
            'raan': 0,
            'arg_of_perigee': 0,
            'true_anomaly': 0,
            'x': 0,
            'y': 0,
            'z': 0,
            'vx': 0,
            'vy': 0,
            'vz': 0,
        }

        initial_orbit2 = {
            'semimajoraxis': 7150000,  # m
            'eccentricity': 0.001,
            'inclination': 98.7,  # degrees
            'raan': 0,
            'arg_of_perigee': 0,
            'true_anomaly': 30,
            'x': 0,
            'y': 0,
            'z': 0,
            'vx': 0,
            'vy': 0,
            'vz': 0,
        }

        initial_epsys = {
            'EnergyStorage': 84 * 3600,  # J
            'SolarPanelSize': 0.4 * 0.3,  # m^2
            'EnergyAvailable': 50 * 3600,  # J
            'Efficiency': 0.3,
            'SolarConstant': 1370  # W/m^2
        }

        # Configuration for the test
        self.time_step = 1 # seconds
        self.duration = 24*60*60  # seconds
        self.num_targets = 1
        self.num_observers = 2
        self.simulator_type = 'everyone'
        
        self.env = FSS_env(self.num_targets, self.num_observers, self.simulator_type, self.time_step, self.duration)

        self.observer1 = ObserverSatellite(num_targets=1, num_observers=2, orbit=initial_orbit.copy(), epsys=initial_epsys.copy())
        self.observer2 = ObserverSatellite(num_targets=1, num_observers=2, orbit=initial_orbit1.copy())
        self.observer3 = ObserverSatellite(num_targets=1, num_observers=2, orbit=initial_orbit2.copy())
        self.target = TargetSatellite(orbit=initial_orbit.copy())
        self.simulator = Simulator(num_targets=1, num_observers=2, time_step=self.time_step, duration=self.duration)
        
        

    def test_orbit_propagation(self):
        print("\n######### Orbit propagation test ##########################################\n")
        # Propagate once with custom propagator to get initial state
        self.observer1.propagate_orbit(self.time_step)
        initial_orbit = self.observer1.orbit.copy()
        print("Initial orbit parameters:")
        print('x:', initial_orbit['x'], 'y:', initial_orbit['y'], 'z:', initial_orbit['z'], 'vx:', initial_orbit['vx'], 'vy:', initial_orbit['vy'], 'vz:', initial_orbit['vz'])

        a = initial_orbit['semimajoraxis'] * units.m
        ecc = initial_orbit['eccentricity'] * units.one
        inc = initial_orbit['inclination'] * units.deg
        raan = initial_orbit['raan'] * units.deg
        argp = initial_orbit['arg_of_perigee'] * units.deg
        nu = initial_orbit['true_anomaly'] * units.deg
        
        orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)

        # Store trajectories
        custom_trajectory = {'x': [], 'y': [], 'z': []}
        poliastro_trajectory = {'x': [], 'y': [], 'z': []}

        custom_trajectory['x'].append(initial_orbit['x'])
        custom_trajectory['y'].append(initial_orbit['y'])
        custom_trajectory['z'].append(initial_orbit['z'])
        poliastro_trajectory['x'].append(orb.r[0].to(units.m).value)
        poliastro_trajectory['y'].append(orb.r[1].to(units.m).value)
        poliastro_trajectory['z'].append(orb.r[2].to(units.m).value)

        # Propagate once more with custom propagator
        self.observer1.propagate_orbit(self.time_step)
        custom_orbit = self.observer1.orbit
        custom_trajectory['x'].append(custom_orbit['x'])
        custom_trajectory['y'].append(custom_orbit['y'])
        custom_trajectory['z'].append(custom_orbit['z'])

        # Propagate with poliastro's propagator
        orb = orb.propagate(self.time_step * units.second)

        poliastro_orbit = {
            'x': orb.r[0].to(units.m).value,  # m
            'y': orb.r[1].to(units.m).value,  # m
            'z': orb.r[2].to(units.m).value,  # m
            'vx': orb.v[0].to(units.m/units.s).value,  # m/s
            'vy': orb.v[1].to(units.m/units.s).value,  # m/s
            'vz': orb.v[2].to(units.m/units.s).value   # m/s
        }
        poliastro_trajectory['x'].append(poliastro_orbit['x'])
        poliastro_trajectory['y'].append(poliastro_orbit['y'])
        poliastro_trajectory['z'].append(poliastro_orbit['z'])

        print("\nCustom Kleperian orbit propagator results:")
        print(f"'x': {custom_orbit['x']}, 'y': {custom_orbit['y']}, 'z': {custom_orbit['z']}, 'vx': {custom_orbit['vx']}, 'vy': {custom_orbit['vy']}, 'vz': {custom_orbit['vz']}")

        print("\nPoliastro orbit propagator results:")
        print(poliastro_orbit)

        # Calculate position error
        pos_error = np.sqrt((custom_orbit['x'] - poliastro_orbit['x'])**2 +
                            (custom_orbit['y'] - poliastro_orbit['y'])**2 +
                            (custom_orbit['z']- poliastro_orbit['z'])**2)

        # Calculate velocity error
        vel_error = np.sqrt((custom_orbit['vx'] - poliastro_orbit['vx'])**2 +
                            (custom_orbit['vy'] - poliastro_orbit['vy'])**2 +
                            (custom_orbit['vz'] - poliastro_orbit['vz'])**2)


        # Convert trajectories to orbital plane coordinates
        custom_orbit_plane = {'x': [], 'y': []}
        poliastro_orbit_plane = {'x': [], 'y': []}
        raan_rad = np.radians(initial_orbit['raan'])
        inc_rad = np.radians(initial_orbit['inclination'])
        argp_rad = np.radians(initial_orbit['arg_of_perigee'])

        for cx, cy, cz in zip(custom_trajectory['x'], custom_trajectory['y'], custom_trajectory['z']):
            x_plane, y_plane, _ = self.to_orbital_plane(cx, cy, cz, raan_rad, inc_rad, argp_rad)
            custom_orbit_plane['x'].append(x_plane)
            custom_orbit_plane['y'].append(y_plane)

        for px, py, pz in zip(poliastro_trajectory['x'], poliastro_trajectory['y'], poliastro_trajectory['z']):
            x_plane, y_plane, _ = self.to_orbital_plane(px, py, pz, raan_rad, inc_rad, argp_rad)
            poliastro_orbit_plane['x'].append(x_plane)
            poliastro_orbit_plane['y'].append(y_plane)

        # Plot comparison in 3D and 2D and save the figure
        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(custom_trajectory['x'], custom_trajectory['y'], custom_trajectory['z'], label='Custom Kleperian Orbit Propagator', color='b')
        ax1.plot(poliastro_trajectory['x'], poliastro_trajectory['y'], poliastro_trajectory['z'], label='Poliastro', color='r')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.legend()
        ax1.set_title(f'Position Error: {pos_error:.3f} m')

        # Add a sphere representing the Earth's surface
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = 6371000 * np.cos(u) * np.sin(v)
        y = 6371000 * np.sin(u) * np.sin(v)
        z = 6371000 * np.cos(v)
        ax1.plot_surface(x, y, z, color='b', alpha=0.3)

        ax2 = fig.add_subplot(222, projection='3d')
        ax2.plot([0, custom_orbit['vx']], [0, custom_orbit['vy']], [0, custom_orbit['vz']], label='Custom Kleperian Orbit Propagator', color='b')
        ax2.plot([0, poliastro_orbit['vx']], [0, poliastro_orbit['vy']], [0, poliastro_orbit['vz']], label='Poliastro Orbit Propagator', color='r')
        ax2.set_xlabel('X Velocity (m/s)')
        ax2.set_ylabel('Y Velocity (m/s)')
        ax2.set_zlabel('Z Velocity (m/s)')
        ax2.legend()
        ax2.set_title(f'Velocity Error: {vel_error:.3f} m/s')

        ax3 = fig.add_subplot(223)
        ax3.plot(custom_orbit_plane['x'], custom_orbit_plane['y'], 'b-', label='Custom Kleperian Orbit Propagator')
        ax3.plot(poliastro_orbit_plane['x'], poliastro_orbit_plane['y'], 'r-', label='Poliastro Orbit Propagator')
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.legend()
        ax3.set_title('Orbital Plane')

        plt.suptitle('Orbit Propagation Comparison')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('orbit_propagation_comparison.png')

        # Assertions
        self.assertLess(pos_error, 1000.0, f"Position error is too high: {pos_error} m")
        self.assertLess(vel_error, 100.0, f"Velocity error is too high: {vel_error} m/s")

        print(f"\nPosition error: {pos_error:.3f} m")
        print(f"Velocity error: {vel_error:.3f} m/s")



    def test_attitude_propagation(self):
        print("\n######### Attitude propagation test ##########################################\n")
        # Initial attitude parameters
        initial_attitude = self.observer1.attitude.copy()
        print("Initial attitude parameters:")
        print(f"Quaternion: {initial_attitude['quaternion']}, Angular velocity: {initial_attitude['angular_velocity']}")

        # Store trajectories
        custom_attitude_trajectory = []
        scipy_attitude_trajectory = []

        custom_attitude_trajectory.append(initial_attitude['quaternion'].tolist())

        # Propagate once with custom propagator
        self.observer1.propagate_attitude(self.time_step)
        custom_attitude = self.observer1.attitude
        custom_attitude_trajectory.append(custom_attitude['quaternion'].tolist())

        # Propagate with scipy's solve_ivp
        initial_quaternion = initial_attitude['quaternion']
        angular_velocity = initial_attitude['angular_velocity']
        t_span = [0, self.time_step]
        sol = solve_ivp(self.quaternion_kinematics, t_span, initial_quaternion, args=(angular_velocity,), method='RK45')
        scipy_quaternion = sol.y[:, -1]
        scipy_attitude_trajectory.append(scipy_quaternion.tolist())

        print("\nCustom attitude propagator results after propagation:")
        print(f"Quaternion: {custom_attitude['quaternion']}")

        print("\nSciPy solver results after propagation:")
        print(f"Quaternion: {scipy_quaternion}")

        # Calculate attitude error (using quaternion distance)
        attitude_error = np.linalg.norm(custom_attitude['quaternion'] - scipy_quaternion)

        # Assertions
        self.assertLess(np.max(attitude_error), 0.1, f"Quaternion error is too high: {np.max(attitude_error)}")

        print(f"\nMax Quaternion error: {np.max(attitude_error)}")

    

    def test_battery_propagation(self):
        print("\n######### Battery propagation test ##########################################\n")
        # Initial battery state
        initial_battery_state = self.observer1.epsys.copy()
        print("Initial battery state:")
        print(f"Energy Available: {initial_battery_state['EnergyAvailable'] / 3600} Wh, Solar Panel Size: {initial_battery_state['SolarPanelSize']} m^2, Efficiency: {initial_battery_state['Efficiency']}, Solar Constant: {initial_battery_state['SolarConstant']} W/m^2")

        # Store energy trajectory
        energy_trajectory = []

        # Propagate for a certain period
        total_time = 20000  # in seconds
        charging_constant = random.choice([1, 3])
        print(f"\nPower consumption constant (1 for normal, 3 for faster discharging): {charging_constant}")
        for t in range(0, total_time, self.time_step):
            self.observer1.propagate_orbit(self.time_step)
            self.observer1.propagate_attitude(self.time_step)
            sunlight_exposure = self.observer1.get_sunlight_exposure()
            self.observer1.charge_battery(sunlight_exposure, self.time_step)
            power_consumption = charging_constant*random.choice([self.observer1.power_consumption_rates["standby"],
                                                   self.observer1.power_consumption_rates["communication"],
                                                   self.observer1.power_consumption_rates["observation"]])
            self.observer1.epsys['EnergyAvailable'] -= power_consumption * self.time_step
            # From process_actions in simulator.py
            if self.observer1.epsys['EnergyAvailable'] < 0: # or observer.DataHand['StorageAvailable'] < 0:
                    print(f"Satellite energy depleted ({self.observer1.name}). Terminating simulation.")
                    self.observer1.epsys['EnergyAvailable'] = 0
                    energy_trajectory.append(self.observer1.epsys['EnergyAvailable'] / 3600)
                    total_time = t + 1
                    break
            energy_trajectory.append(self.observer1.epsys['EnergyAvailable'] / 3600)


        # Print final energy available
        print(f"\nFinal Energy Available: {self.observer1.epsys['EnergyAvailable'] / 3600} Wh")

        # Plot Energy over time
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, total_time, self.time_step), energy_trajectory, label='Energy Available (Wh)')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy Available (Wh)')
        plt.title('Battery Energy Available Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('battery_propagation.png')

        # Assertions
        self.assertGreaterEqual(self.observer1.epsys['EnergyAvailable'], 0, "Energy available is below 0")
        self.assertLessEqual(self.observer1.epsys['EnergyAvailable'], self.observer1.epsys['EnergyStorage'], "Energy available exceeds storage capacity")



    def test_distance_and_effective_data_rate(self):
        print("\n######### Distance and Effective Data Rate Test ##########################################\n")

        # Define test positions
        test_positions = [
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), (7142846.101516889 + 100000, -1129.3850175896163, 7380.564140244884)),   # 100 km apart
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), (7142846.101516889 + 500000, -11293.850175896163, 7380.564140244884)),
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), (7142846.101516889 + 2300000, -11293.850175896163, 7380.564140244884)),
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), (7142846.101516889 + 2325000, -11293.850175896163, 7380.564140244884)),
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), (7142846.101516889 + 2350000, -11293.850175896163, 7380.564140244884)),
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), (7142846.101516889 + 2400000, -11293.850175896163, 7380.564140244884)),
        ]

        for pos1, pos2 in test_positions:
            # Set positions
            self.observer2.orbit['x'], self.observer2.orbit['y'], self.observer2.orbit['z'] = pos1
            self.observer3.orbit['x'], self.observer3.orbit['y'], self.observer3.orbit['z'] = pos2

            # Calculate distance
            dist = self.observer2.distance_between(self.observer3, self.time_step)
            print(f"Distance between satellites: {dist:.2f} m")

            # Calculate effective data rate
            eff_data_rate = self.observer1.data_comms.calculateEffectiveDataRate(dist) # in meters
            print(f"Effective Data Rate: {eff_data_rate:.2f} bits/sec")

            agents = ["observer_" + str(r) for r in range(2)]
            self.observer2.is_processing = False
            self.observer3.is_processing = False
            self.observer2.has_new_data[1] = True
            data_to_transmit = 3000  # bits
            reward_step = {agent: 0 for agent in agents}
            max_steps = 0
            communication_done = False
            steps = 0
            data_transmitted = 0
            total_data_transmitted = 0
            can_commmunicate = self.observer2.can_communicate(1)
            can_communicate_with = False
            print(f"Available and new info to communicate: {can_commmunicate}")
            if can_commmunicate:
                can_communicate_with = self.observer2.can_communicate_with_everyone(self.observer3, self.time_step)
            print(f"Can communicate with other: {can_communicate_with}")
            while not communication_done and total_data_transmitted < data_to_transmit:
                reward_step[agents[0]], communication_done, steps, contacts_matrix, contacts_matrix_acc, adjacency_matrix, adjacency_matrix_acc, data_matrix, data_matrix_acc, global_observation_counts, max_pointing_accuracy_avg, data_transmitted = self.observer2.propagate_information(0, self.observer3, 1, self.time_step + steps * self.time_step, "everyone", reward_step[agents[0]], steps, communication_done, total_data_transmitted, data_to_transmit)
                total_data_transmitted += data_transmitted  # Accumulate data transmitted
                if communication_done or data_transmitted == 0:
                    print(f"Reward: {reward_step[agents[0]]}, Data transmitted: {total_data_transmitted}, Steps: {steps} \n")
                    break  # Exit if communication is done or no data was transmitted
            
            max_steps = max(steps, max_steps)

            # Assertions
            expected_distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
            self.assertAlmostEqual(dist, expected_distance, places=2, msg=f"Distance mismatch for positions {pos1} and {pos2}")
            self.assertGreaterEqual(eff_data_rate, 0, f"Effective data rate should be non-negative for distance {dist}")


    
    def test_observation_accuracy(self):
        print("\n######### Observation Accuracy Test ##########################################\n")
        # Define test positions around the limit distance of 263 km (263.47 km)
        # Field of view is 40º, so angles between pointing direction and target greater than 20º should be out of view
        # Define test positions and attitudes
        test_positions_and_attitudes = [
            # (observer position, target distance, attitude quaternion, should_observe)
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 100000, [1, 0, 0, 0], True),  # Perfect alignment
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 263000, [1, 0, 0, 0], True),  # On the edge of the max distance
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 264000, [1, 0, 0, 0], False),  # Just beyond the max distance
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 200000, [0.707, 0, 0.707, 0], False),  # 90-degree rotation
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 200000, [0.866, 0, -0.5, 0], False),  # negative 60-degree rotation
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 200000, [0.966, 0, 0.259, 0], False),  # 30-degree rotation
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 200000, [0.9848, 0, 0.1736, 0], True),  # 20-degree rotation
            ((7142846.101516889, -1129.3850175896163, 7380.564140244884), 200000, [0.9962, 0, -0.0872, 0], True),  # -10-degree rotation
        ]

        for pos1, distance, quaternion, should_observe in test_positions_and_attitudes:
            # Set positions and attitude
            self.observer1.orbit['x'], self.observer1.orbit['y'], self.observer1.orbit['z'] = pos1
            self.target.orbit['x'] = pos1[0] + distance
            self.target.orbit['y'], self.target.orbit['z'] = pos1[1], pos1[2]
            self.observer1.attitude['quaternion'] = quaternion

            agent = "observer_1"
            reward_step = {agent: 0}

            self.observer1.is_processing = False
            steps = 0
            self.observer1.observation_status_matrix[0] = 0
            self.observer1.cumulative_pointing_accuracy[0, 0] = 0
            self.observer1.observation_counts[0] = 0
            self.observer1.global_observation_counts[0, 0] = 0
            self.observer1.max_pointing_accuracy_avg_sat[0] = 0

            print(f"Testing positions:\nObserver 1: {pos1}\nTarget: ({self.target.orbit['x']}, {self.target.orbit['y']}, {self.target.orbit['z']})")

            reward_step[agent], steps, contacts_matrix, contacts_matrix_acc, adjacency_matrix, adjacency_matrix_acc, data_matrix, data_matrix_acc, global_observation_counts, max_pointing_accuracy_avg = self.observer1.observe_target(0, self.target, 0, self.time_step + steps*self.time_step, reward_step[agent], steps)
            
            pointing_accuracy = self.observer1.evaluate_pointing_accuracy(self.target, self.time_step)
            print(f"Distance: {distance} m")
            print(f"Reward: {reward_step[agent]}")
            print(f"Pointing Accuracy: {pointing_accuracy:.2f}")
            print(f"Observation Status Matrix: {self.observer1.observation_status_matrix}")
            print(f"Cumulative Pointing Accuracy: {self.observer1.cumulative_pointing_accuracy}")
            print(f"Observation Counts: {self.observer1.observation_counts[0]}")
            print(f"Global Observation Counts: {global_observation_counts}")
            print(f"Max Pointing Accuracy Avg: {max_pointing_accuracy_avg} \n")

            if should_observe:
                assert pointing_accuracy > 0, "Expected positive pointing accuracy but got zero or negative."
                # Check if the observation status matrix is correctly updated
                assert self.observer1.observation_status_matrix[0] == 2 or self.observer1.observation_status_matrix[0] == 3, "Observation status matrix not updated correctly."

                # Check if the pointing accuracy is positive and cumulative pointing accuracy is updated
                assert self.observer1.cumulative_pointing_accuracy[0, 0] > 0, "Cumulative pointing accuracy not updated correctly."

                # Check if the observation counts are incremented
                assert self.observer1.observation_counts[0] > 0, "Observation counts not incremented correctly."

                # Check if global observation counts are updated
                assert global_observation_counts[0, 0] > 0, "Global observation counts not updated correctly."

                # Check if the max pointing accuracy average is calculated correctly
                assert max_pointing_accuracy_avg[0] > 0, "Max pointing accuracy average not calculated correctly."
            else:
                assert pointing_accuracy == 0, "Expected zero pointing accuracy but got positive."
                # Check if the observation was not successful
                assert self.observer1.observation_status_matrix[0] != 2 and self.observer1.observation_status_matrix[0] != 3, "Observation status matrix incorrectly updated for failed observation."

                # Check if the pointing accuracy is not positive and cumulative pointing accuracy is not updated
                assert self.observer1.cumulative_pointing_accuracy[0, 0] == 0, "Cumulative pointing accuracy incorrectly updated for failed observation."

                # Check if the observation counts are not incremented
                assert self.observer1.observation_counts[0] == 0, "Observation counts incorrectly incremented for failed observation."

                # Check if global observation counts are not updated
                assert global_observation_counts[0, 0] == 0, "Global observation counts incorrectly updated for failed observation."

                # Check if the max pointing accuracy average is not calculated
                assert max_pointing_accuracy_avg[0] == 0, "Max pointing accuracy average incorrectly calculated for failed observation."

    
    def test_dummy_simulation(self):
        print("\n######## Starting Dummy Simulation ########\n")
        
        total_reward = 0
        observation, infos = self.env.reset()

        # Manually set positions and attitudes to control the simulation
        initial_positions = [
            (7142846.101516889, -1129.3850175896163, 7380.564140244884),  # Observer 1
            (7142846.101516889 + 10, -1129.3850175896163, 7380.564140244884)  # Observer 2
        ]
        target_position = (7142846.101516889 + 1500, -1129.3850175896163, 7380.564140244884)  # Target

        self.env.simulator.observer_satellites[0].orbit.update({'x': initial_positions[0][0], 'y': initial_positions[0][1], 'z': initial_positions[0][2]})
        self.env.simulator.observer_satellites[1].orbit.update({'x': initial_positions[1][0], 'y': initial_positions[1][1], 'z': initial_positions[1][2]})
        self.env.simulator.target_satellites[0].orbit.update({'x': target_position[0], 'y': target_position[1], 'z': target_position[2]})

        start_time = time.time()

        while self.env.agents:
            step_start_time = time.time()

            # Manually control the actions for the agents
            actions = {}
            for agent in self.env.agents:
                if agent == "observer_0":
                    actions[agent] = 2 # Observation
                elif agent == "observer_1":
                    actions[agent] = 0 # Standby
            
            if self.env.simulator.time_step_number > 1:
                actions["observer_0"] = 1 # Communication

            if self.env.simulator.time_step_number > 10:
                actions["observer_0"] = 0 # Standby
                actions["observer_1"] = 0 # Standby


            observation, rewards, terminated, truncated, infos = self.env.step(actions)
            total_reward += sum(rewards.values())
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

            if any(terminated.values()) or any(truncated.values()):
                print("Episode finished")
                break

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\nTotal steps: {self.env.simulator.time_step_number}")
        print(f"Total duration of episode: {total_duration:.3f} seconds")
        print(f"Total reward: {total_reward}")

        # Log the results
        results = {
            'total_steps': self.env.simulator.time_step_number,
            'total_duration': total_duration,
            'total_reward': total_reward,
            'adjacency_matrix_acc': self.env.simulator.adjacency_matrix_acc,
            'data_matrix_acc': self.env.simulator.data_matrix_acc,
            'contacts_matrix_acc': self.env.simulator.contacts_matrix_acc,
            'global_observation_counts': self.env.simulator.global_observation_counts,
            'max_pointing_accuracy_avg': self.env.simulator.max_pointing_accuracy_avg,
            'global_observation_status_matrix': self.env.simulator.global_observation_status_matrix,
            'batteries': self.env.simulator.batteries,
            'storage': self.env.simulator.storage
        }

        # Print some of the results for verification
        print("\nResults:")
        print(f"Adjacency Matrix:\n{results['adjacency_matrix_acc']}")
        print(f"Data Matrix:\n{results['data_matrix_acc']}")
        print(f"Contacts Matrix:\n{results['contacts_matrix_acc']}")
        print(f"Global Observation Counts:\n{results['global_observation_counts']}")
        print(f"Max Pointing Accuracy Avg:\n{results['max_pointing_accuracy_avg']}")
        print(f"Global Observation Status Matrix:\n{results['global_observation_status_matrix']}")
        print(f"Batteries:\n{results['batteries']}")
        print(f"Storage:\n{results['storage']}")

        # Assertions to check if the simulation ran correctly
        self.assertGreater(results['total_steps'], 0, "Simulation did not run any steps.")
        self.assertGreater(results['total_duration'], 0, "Simulation did not take any time.")
        # self.assertGreater(results['total_reward'], 0, "Total reward is not positive, indicating potential issues in observation or communication.")
        self.assertTrue((results['batteries'] >= 0).all(), "Battery levels went below zero.")
        self.assertTrue((results['storage'] >= 0).all(), "Storage levels went below zero.")
        # Create a mask that excludes the diagonal elements
        non_diagonal_mask = ~np.eye(results['adjacency_matrix_acc'].shape[0], dtype=bool)
        self.assertTrue((results['adjacency_matrix_acc'][non_diagonal_mask] > 0).any(), "No communications were made (non-diagonal elements).")
        self.assertTrue((results['contacts_matrix_acc'] > 0).any(), "No contacts were made.")
        self.assertTrue((results['global_observation_counts'] > 0).any(), "No observations were made.")

        

















    def propagate_battery(self, time_step):
        sunlight_exposure = self.get_sunlight_exposure()
        self.charge_battery(sunlight_exposure, time_step)

    def quaternion_kinematics(self, t, q, omega):
        """
        Quaternion kinematics differential equation.

        Args:
            t (float): Time (not used in this autonomous system).
            q (array): Quaternion [q0, q1, q2, q3].
            omega (array): Angular velocity [wx, wy, wz].

        Returns:
            array: Derivative of quaternion [q0_dot, q1_dot, q2_dot, q3_dot].
        """
        q0, q1, q2, q3 = q
        wx, wy, wz = omega
        q_dot = 0.5 * np.array([
            -q1 * wx - q2 * wy - q3 * wz,
            q0 * wx + q2 * wz - q3 * wy,
            q0 * wy - q1 * wz + q3 * wx,
            q0 * wz + q1 * wy - q2 * wx,
        ])
        return q_dot


    def to_orbital_plane(self, x, y, z, raan, inc, argp):
        # Rotation matrices for transforming from inertial to orbital plane coordinates
        R3_raan = np.array([
            [np.cos(raan), np.sin(raan), 0],
            [-np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        R1_inc = np.array([
            [1, 0, 0],
            [0, np.cos(inc), np.sin(inc)],
            [0, -np.sin(inc), np.cos(inc)]
        ])
        R3_argp = np.array([
            [np.cos(argp), np.sin(argp), 0],
            [-np.sin(argp), np.cos(argp), 0],
            [0, 0, 1]
        ])
        R = R3_argp @ R1_inc @ R3_raan
        
        coords = np.array([x, y, z])
        return R @ coords

    def generate_tle(self, orbit):
        """
        Generate a simplified TLE-like structure from given orbital parameters.

        Args:
            orbit (dict): A dictionary containing the orbital parameters:
                        'semimajoraxis', 'eccentricity', 'inclination', 'raan', 'arg_of_perigee', 'true_anomaly'

        Returns:
            tuple: A tuple containing two TLE lines as strings.
        """
        from datetime import datetime

        # Dummy satellite number and international designator
        satellite_number = "00000"
        classification = "U"
        launch_year = "00"
        launch_number = "000"
        piece_of_launch = "A"

        # Epoch time
        now = datetime.now()
        epoch_year = now.year % 100
        epoch_day = now.timetuple().tm_yday + now.hour / 24.0 + now.minute / 1440.0 + now.second / 86400.0

        # Mean motion calculation
        mu = 398600.4418  # Earth's gravitational parameter in km^3/s^2
        a = orbit['semimajoraxis']  # semi-major axis in km
        mean_motion = np.sqrt(mu / a**3) * 86400 / (2 * np.pi)  # revolutions per day
        # print(f"Mean motion: {mean_motion}")

        mean_motion_dot = 0.00001264
        mean_motion_ddot = 0.00000
        bstar = 0.29621e-4

        # Orbital elements
        inclination = orbit['inclination']
        raan = orbit['raan']
        eccentricity = int(orbit['eccentricity'] * 1e7)
        arg_of_perigee = orbit['arg_of_perigee']
        mean_anomaly = orbit['true_anomaly']

        # Construct TLE lines
        line1 = f"1 {satellite_number}{classification} {launch_year}{launch_number}{piece_of_launch}   {epoch_year:02}{epoch_day:012.8f}  .{mean_motion_dot:8.8f}  00000-0 {bstar:8.8f} 0  9991"
        line2 = f"2 {satellite_number} {inclination:8.4f} {raan:8.4f} {eccentricity:7d} {arg_of_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}00000"

        return line1, line2

if __name__ == "__main__":
    unittest.main()