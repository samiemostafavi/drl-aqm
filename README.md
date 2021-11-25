# Deep Reinforcement Learning-based Active Queue Management

In this repository, I invesitgate the implementation of a DRL-based agent for dropping the packets in a queuing model. I use an stable version of [OpenAI baselines](https://github.com/DLR-RM/stable-baselines3) for this purpose. The queuing simulation is implemented using MATLAB in [another repository](https://github.com/samiemostafavi/matlab-queuing-simulation).

## Setup

1. Install stable-baselines3


## Development Guide

I started by creating a custom environment that follows OpenAI gym interface based on [this](https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1) notebook and [this](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html) tutorial from [OpenAI baselines](https://github.com/DLR-RM/stable-baselines3). The result is a shared memory communication module `queue_env.py` that connects to MATLAB and exchanges the information.

In `run.py`, a [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) RL agent is instantiated using the custom environment and MLP policy.

