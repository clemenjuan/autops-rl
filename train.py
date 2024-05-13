import os
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.algorithms.ppo import (
    PPOConfig,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from FSS_env import FSS_env

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--framework", choices=["tf", "tf2", "torch"], default="torch", help="The DL framework specifier.")
# parser.add_argument("--as-test", action="store_true", help="Whether this script should be run as a test.")
parser.add_argument("--stop-iters", type=int, default=20, help="Number of iterations to train.")
# parser.add_argument("--stop-timesteps", type=int, default=100000, help="Number of timesteps to train.")
parser.add_argument("--stop-reward", type=float, default=500000.0, help="Reward at which we stop training.")
args = parser.parse_args()

def env_creator(env_config):
    """Function to create and return your custom environment"""
    env = FSS_env(**env_config)
    # env = FlattenObservationWrapper(env)
    return env

# Register your environment to use it with Ray RLlib
env_name = "FSS_env-v0"
env_config = {
        "num_targets": 10, 
        "num_observers": 10, 
        "simulator_type": 'everyone', 
        "time_step": 1, 
        "duration": 24*60*60
        }

# register_env(env_name, lambda config=None: ValidateSpacesWrapper(env_creator(env_config)))
register_env(env_name, lambda config=None: env_creator(env_config))


tmp_env = env_creator(env_config)
act_space = tmp_env.action_space
obs_space = tmp_env.observation_space
observations, infos = tmp_env.reset()

# Check observation spaces and initial observations
# print("Observation space:", obs_space)
# print("Initial observation:", observations)

def check_observation_space(obs_space, observations):
    """Check if all expected keys are present in the observations"""
    # Iterate over expected agents and verify all keys are present
    missing_keys = {}

    # Check each agent's observation against the expected keys in the observation space
    for agent_id, agent_obs in observations.items():
        agent_space = obs_space[agent_id]  # This accesses the observation space for the current agent
        # List missing keys by checking if each expected key in agent_space is in the agent's actual observations
        missing_keys[agent_id] = [key for key in agent_space.spaces.keys() if key not in agent_obs]

    # Check if there are any agents with missing keys and raise an error if there are
    if any(missing_keys.values()):  # This checks if there is any non-empty list in the dictionary
        raise KeyError(f"Missing keys in observations for agents: {missing_keys}")
    else:
        print("All agents' observations contain the required keys.")

    # Continue with validation of each observation according to its space
    for agent_id, agent_obs in observations.items():
        agent_space = obs_space[agent_id]
        for key, space in agent_space.spaces.items():
            assert space.contains(agent_obs[key]), f"Observation {key} for {agent_id} out of bounds"

    print("All observations are within bounds and properly validated!")


# check_observation_space(obs_space, observations)



def select_policy(algorithm, framework):
        if algorithm == "PPO":
            if framework == "torch":
                return PPOTorchPolicy
            elif framework == "tf":
                return PPOTF1Policy
            else:
                return PPOTF2Policy
        elif algorithm == "DQN":
            if framework == "torch":
                return DQNTorchPolicy
            else:
                return DQNTFPolicy
        else:
            raise ValueError("Unknown algorithm: ", algorithm)



# Configuration for PPO
ppo_config = PPOConfig()
ppo_config.environment(
     env=env_name,
     disable_env_checking=True,
    )
ppo_config.framework(args.framework)
ppo_config.rollouts(
    num_rollout_workers=4,
    num_envs_per_worker=2,
    rollout_fragment_length="auto",
    batch_mode="complete_episodes"
)
ppo_config.training(
        vf_loss_coeff=0.01,
        num_sgd_iter=6,
        train_batch_size=env_config["duration"],
        lr=0.0001, # tune.loguniform(1e-4, 1e-2),
        gamma=0.95, # tune.uniform(0.9, 0.99),
        use_gae=True,
        lambda_=0.95, # tune.uniform(0.9, 1.0),
        clip_param=0.2,
        entropy_coeff=0.01,
        sgd_minibatch_size=64,
    )
ppo_config.resources(num_gpus=0)

# Configuration for DQN
dqn_config = DQNConfig()
dqn_config.environment(
     env_name,
     disable_env_checking=True,
     )
dqn_config.framework(args.framework)
dqn_config.rollouts(
    num_rollout_workers=4,
    num_envs_per_worker=2,
    rollout_fragment_length="auto",
    batch_mode="complete_episodes"
)
dqn_config.training(
        n_step=3,
        lr=0.0001, # tune.loguniform(1e-4, 1e-2),
        gamma=0.95, # tune.uniform(0.9, 0.99)
    )
dqn_config.resources(num_gpus=0)



# Specify two policies, each with their own config created above
# You can also have multiple policies per algorithm, but here we just
# show one each for PPO and DQN.
policies = {
    "ppo_policy": (
        select_policy("PPO", args.framework),
        obs_space,
        act_space,
        ppo_config,
    ),
    "dqn_policy": (
        select_policy("DQN", args.framework),
        obs_space,
        act_space,
        dqn_config,
    ),
}

def policy_mapping_fn_ppo(agent_id, episode, worker, **kwargs):
        return "ppo_policy"
        
def policy_mapping_fn_dqn(agent_id, episode, worker, **kwargs):
        return "dqn_policy"
        
# Add multi-agent configuration options to both configs and build them.
ppo_config.multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapping_fn_ppo,
    policies_to_train=["ppo_policy"],
)

print("Building PPO")
ppo = ppo_config.build()
print("PPO built")

dqn_config.multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapping_fn_dqn,
    policies_to_train=["dqn_policy"],
)

'''
print("Building DQN")
dqn = dqn_config.build()
print("DQN built")
'''



# You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
for i in range(args.stop_iters):
    print("== Iteration", i, "==")

    '''
    # improve the DQN policy
    print("-- DQN --")
    result_dqn = dqn.train()
    print(pretty_print(result_dqn))
    '''

    # improve the PPO policy
    print("-- PPO --")
    result_ppo = ppo.train()
    print(pretty_print(result_ppo))

    # Test passed gracefully.
    if (
        args.as_test
        # and result_dqn["episode_reward_mean"] > args.stop_reward
        and result_ppo["episode_reward_mean"] > args.stop_reward
    ):
        print("test passed (agents above requested reward)")
        quit(0)

    # swap weights to synchronize
    # dqn.set_weights(ppo.get_weights(["ppo_policy"]))
    # ppo.set_weights(dqn.get_weights(["dqn_policy"]))

# Desired reward not reached.
# if args.as_test:
#    raise ValueError("Desired reward ({}) not reached!".format(args.stop_reward))
