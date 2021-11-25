import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from queue_env import QueueEnv


def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    step_rewards = []
    for i in range(num_steps):
 
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward)

    agg_reward = sum(step_rewards)

    print("Sum rewards:", agg_reward, "Num steps:", num_steps)

    return agg_reward


env = QueueEnv()
model = PPO(MlpPolicy, env, verbose=0)

model.learn(total_timesteps=2000000)

# Random Agent, before training
#reward_before_train = evaluate(model, num_steps=1000)


