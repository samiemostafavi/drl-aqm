from stable_baselines3.common.env_checker import check_env
from queue_env import QueueEnv

# Stable Baselines provides a helper to check that your environment follows the Gym interface. It also optionally checks that the environment is compatible with Stable-Baselines (and emits warning if necessary).
env = QueueEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

# Testing the environment
env = QueueEnv()

obs = env.reset()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

PASS = 0
# Hardcoded agent: always pass!
n_steps = 100
for step in range(n_steps):
  print("Step {}".format(step + 1))
  obs, reward, done, info = env.step(PASS)
  print('obs=', obs, 'reward=', reward, 'done=', done)



