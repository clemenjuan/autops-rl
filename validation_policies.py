import os
import numpy as np
import unittest
from collections import defaultdict
import matplotlib.pyplot as plt
from ray.rllib.policy.policy import Policy
from FSS_env import FSS_env  # Replace with actual import
from satellites import ObserverSatellite, TargetSatellite
from simulator import Simulator, CentralizedSimulator

def env_creator(env_config):
    return FSS_env(**env_config)

def generate_fixed_initial_conditions(num_observers, num_targets, energy_level, storage_level):
    initial_conditions = {
        'observer_orbits': [],
        'target_orbits': [],
        'energy_levels': [energy_level] * num_observers + [70 * 3600] * num_targets,  # Ensure sufficient energy levels for targets
        'storage_levels': [storage_level] * num_observers + [7.9 * 64e9] * num_targets,  # Ensure sufficient storage levels for targets
    }

    # Observer 1 and 2 within communication range (2325 km), isolated from observation test
    observer1_orbit = {
        'semimajoraxis': 7140000,  # m
        'eccentricity': 0.001,
        'inclination': 98.7,  # degrees
        'raan': 0,
        'arg_of_perigee': 0,
        'true_anomaly': 0,
        'x': 7142846.101516889,
        'y': -1129.3850175896163,
        'z': 7380.564140244884,
        'vx': 0,
        'vy': 0,
        'vz': 0,
    }
    observer2_orbit = observer1_orbit.copy()
    observer2_orbit['x'] += 200000  # 2000 km along x-axis

    # Observer 3 and Target 1 within observation range (263 km), isolated from communication test
    observer3_orbit = observer1_orbit.copy()
    observer3_orbit['true_anomaly'] = 180  # 180º offset

    target1_orbit = observer3_orbit.copy()
    target1_orbit['x'] += 200000  # 200 km along x-axis

    # Additional observer for standby test
    observer4_orbit = observer1_orbit.copy()
    observer4_orbit['true_anomaly'] = 90  # 90º offset to isolate

    initial_conditions['observer_orbits'].extend([observer1_orbit, observer2_orbit, observer3_orbit, observer4_orbit])

    # Add random orbits for the remaining observers and targets
    for _ in range(num_observers - len(initial_conditions['observer_orbits'])):
        random_observer_orbit = {
            'semimajoraxis': 10000000,  # 10k km, way above the rest of the orbits
            'eccentricity': np.random.uniform(0, 0.001),
            'inclination': 98.7,  # degrees
            'raan': np.random.uniform(0, 360),
            'arg_of_perigee': 0,
            'true_anomaly': np.random.uniform(0, 360),
        }
        initial_conditions['observer_orbits'].append(random_observer_orbit)

    for _ in range(num_targets - len(initial_conditions['target_orbits'])):
        random_target_orbit = {
            'semimajoraxis': 10000000,  # 10k km, way above the rest of the orbits
            'eccentricity': np.random.uniform(0, 0.001),
            'inclination': 98.7,  # degrees
            'raan': np.random.uniform(0, 360),
            'arg_of_perigee': 0,
            'true_anomaly': np.random.uniform(0, 360),
        }
        initial_conditions['target_orbits'].append(random_target_orbit)

    return initial_conditions

def initialize_specific_satellites(initial_conditions, num_observers, num_targets):
    observers = []
    targets = []
    initial_epsys_template = {
        'EnergyStorage': 84 * 3600,  # J
        'SolarPanelSize': 0.4 * 0.3,  # m^2
        'Efficiency': 0.3,
        'SolarConstant': 1370  # W/m^2
    }

    initial_datahand_template = {
        'DataStorage': 8 * 64e9,  # Maximum storage onboard. 8 * 64G bytes, from ISISpace bus
        'DataSize': 52,  # Data package size per satellite in bytes
    }

    for i in range(num_observers):
        initial_epsys = initial_epsys_template.copy()
        initial_epsys['EnergyAvailable'] = initial_conditions['energy_levels'][i]

        initial_datahand = initial_datahand_template.copy()
        initial_datahand['StorageAvailable'] = initial_conditions['storage_levels'][i]

        observer = ObserverSatellite(
            num_targets=num_targets,
            num_observers=num_observers,
            orbit=initial_conditions['observer_orbits'][i],
            epsys=initial_epsys,
            DataHand=initial_datahand
        )
        observers.append(observer)

    for i in range(num_targets):
        initial_epsys = initial_epsys_template.copy()
        initial_epsys['EnergyAvailable'] = initial_conditions['energy_levels'][num_observers + i]

        initial_datahand = initial_datahand_template.copy()
        initial_datahand['StorageAvailable'] = initial_conditions['storage_levels'][num_observers + i]

        target = TargetSatellite(
            orbit=initial_conditions['target_orbits'][i],
            epsys=initial_epsys,
            DataHand=initial_datahand
        )
        targets.append(target)

    return observers, targets

def test_policy_once(env, policy, total_reward, observation):
    actions = {agent: policy.compute_single_action(observation[agent])[0] for agent in env.agents}

    observation, rewards, terminated, truncated, infos = env.step(actions)
    total_reward += sum(rewards.values())

    return actions, total_reward, observation, rewards

class TestSatelliteSimulation(unittest.TestCase):
    
    def setUp(self):
        self.time_step = 1  # seconds
        self.duration = 5  # Run for a few steps
        self.num_targets = 20
        self.num_observers = 20  # Updated number of observers
        self.simulator_type = 'everyone'
        
        self.env_config = {
            'num_targets': self.num_targets,
            'num_observers': self.num_observers,
            'simulator_type': self.simulator_type,
            'time_step': self.time_step,
            'duration': self.duration
        }

    def run_tests_for_policy(self, policy_type, checkpoint_dir, num_repetitions):
        action_results = defaultdict(lambda: np.zeros(22)) 

        policy_checkpoint_path = os.path.join(checkpoint_dir, "policies", "default_policy")
        my_restored_policy = Policy.from_checkpoint(policy_checkpoint_path)

        # Define the energy and storage levels for each scenario
        resource_scenarios = {
            'low_low': (1 * 3600, 7.9 * 64e9),
            'low_high': (1 * 3600, 2 * 64e9),
            'high_low': (70 * 3600, 7.9 * 64e9),
            'high_high': (70 * 3600, 2 * 64e9)
        }

        for scenario, (energy_level, storage_level) in resource_scenarios.items():
            for _ in range(num_repetitions):
                initial_conditions = generate_fixed_initial_conditions(self.num_observers, self.num_targets, energy_level, storage_level)
                self.observers, self.targets = initialize_specific_satellites(initial_conditions, self.num_observers, self.num_targets)
                self.env = env_creator(self.env_config)

                total_reward = 0
                observation, _ = self.env.reset()

                actions, total_reward, observation, rewards = test_policy_once(self.env, my_restored_policy, total_reward, observation)

                for agent_id, scenario_key in [(0, f'{scenario}_comm'), (2, f'{scenario}_observe'), (3, f'{scenario}_standby')]:
                    if agent_id < len(self.env.agents):
                        action_idx = actions[self.env.agents[agent_id]]
                        # print(f"Action chosen by {self.env.agents[agent_id]}: {action_idx}")

                        action_results[scenario_key][action_idx] += 1

        return action_results

    def test_all_scenarios(self):
        checkpoint_dirs = {
            "DQN": "checkpoints_new/dqn",
            "SAC": "checkpoints_new/sac",
            "PPO": "checkpoints_new/ppo"
        }

        num_repetitions = 100
        all_results = {}

        for policy_type, checkpoint_dir in checkpoint_dirs.items():
            print(f"Testing {policy_type} policy")
            results = self.run_tests_for_policy(policy_type, checkpoint_dir, num_repetitions)
            all_results[policy_type] = results

        # Plot the results
        fig, axs = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Policy Actions Distribution', fontsize=16)
        scenarios = ['comm', 'observe', 'standby']
        resource_levels = ['low_low', 'low_high', 'high_low', 'high_high']
        policies = ['DQN', 'SAC', 'PPO']

        bar_width = 0.25  # Width of each bar
        x = np.arange(22)  # The action indices

        for i, scenario in enumerate(scenarios):
            for j, resource_level in enumerate(resource_levels):
                ax = axs[i, j]
                for k, policy in enumerate(policies):
                    scenario_key = f'{resource_level}_{scenario}'
                    data = all_results[policy][scenario_key]
                    ax.bar(x + k * bar_width, data, width=bar_width, label=f'{policy} Agent')
                
                ax.set_title(f'{resource_level.capitalize()} (battery, storage) {scenario.capitalize()} Scenario', fontsize=14)
                ax.set_xlabel('Actions', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_xticks(x + bar_width)  # Center the ticks between the grouped bars
                ax.set_xticklabels(range(22), rotation=90)
                ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('policy_actions_distribution.png')
        plt.show()
        print("Results saved to policy_actions_distribution.png")

if __name__=="__main__":
    unittest.main()