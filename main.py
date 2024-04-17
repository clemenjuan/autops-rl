from ray.rllib.agents.ppo import PPOTrainer

# Load the best configuration and checkpoint path
best_config = analysis.get_best_config("episode_reward_mean", "max")
best_checkpoint = analysis.get_best_checkpoint(best_trial, "episode_reward_mean", "max")

# Initialize the trainer with the best configuration
trainer = PPOTrainer(config=best_config)

# Load the best model
trainer.restore(best_checkpoint)

for i in range(num_simulations):
    print(f"Starting simulation {i}...")
    env = SatelliteEnv(num_targets, num_observers, simulator_type, time_step, duration)
    observation, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Use the trained model to select actions
        action_dict = {}
        for agent_id in env.agents:
            action_dict[agent_id] = trainer.compute_action(observation[agent_id])

        # Step through the environment using the selected actions
        observation, reward, done, info = env.step(action_dict)
        total_reward += reward

    print(f"Episode finished")
    print(f"Total duration of episode: {env.simulator.time_step_number * time_step:.3f} seconds")
    print(f"Total reward: {total_reward}")
