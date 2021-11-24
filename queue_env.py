import numpy as np
import gym
from gym import spaces
import mmap
import os
import struct
import time

def create_file(filename,size):
    with open(filename, 'wb') as f:
        f.seek(size-1)
        f.write(b'\x00')
        f.close()

def memory_map(filename, access=mmap.ACCESS_WRITE):
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)


class QueueEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where connects to MATLAB through a shared memory interface.
  It is an implementation of this paper "Deep Reinforcement Learning Based Active Queue
  Management for IoT Networks" by Kim et al.
  """
 
  # Define constants for clearer code
  PASS = 0
  DROP = 1

  def __init__(self):
    super(QueueEnv, self).__init__()

    # Define action and observation space
    # They must be gym.spaces objects
    # Using discrete actions, we have two: { drop , pass }
    n_actions = 2
    self.action_space = spaces.Discrete(n_actions)

    # The observation will be the { number of packets L, dequeue rate R_deq, and queuing delay d }
    # this can be described both by Discrete and Box space
    # number of packets L, dequeue rate R_deq, queuing delay d
    self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.float32, shape=(3,))    

    # Shared memory MATLAB communication
    self.mm_filename = 'sm_comm.dat'
    create_file(self.mm_filename,40) # 5 doubles are stored
    self.mmap_obj = memory_map(self.mm_filename)

  def reset(self):
    """
    Important: the observation must be a Tuple of numpy arrays 
    each box represents a numpy array
    :return: Tuple(np.array)
    """
    # Initialize the agent
    self.backlog = 0
    self.dequeue_rate = 0
    self.cur_delay = 0
    # here we convert each number to its corresponding type to make it more general
    # (in case we want to use continuous actions)
    return np.array([self.backlog,self.dequeue_rate,self.cur_delay]).astype(np.float32)

  def step(self, action):
    if (action != self.DROP) and (action != self.PASS) :
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    # write to the file
    action_float = float(action)
    self.mmap_obj.seek(0)
    self.mmap_obj.write(struct.pack('ddddd', 1.0, action_float, 0.0, 0.0, 0.0))

    # wait for MATLAB to write the response (first number becomes zero)
    while True:
        self.mmap_obj.seek(0)
        res = struct.unpack('ddddd', self.mmap_obj.read())
        if res[0] == 0:
            break
        else:
            time.sleep(0.0001)

    # prepare observations and reward
    self.backlog = res[1]
    self.dequeue_rate = res[2]
    self.queue_delay = res[3]
    reward = res[4]

    # Account for the boundaries of the grid
    self.backlog = np.clip(self.backlog, 0, np.inf)
    self.dequeue_rate = np.clip(self.dequeue_rate, 0, np.inf)
    self.cur_delay = np.clip(self.queue_delay, 0, np.inf)

    # The experiment never finishes
    done = False

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return np.array([self.backlog,self.dequeue_rate,self.queue_delay]).astype(np.float32), reward, done, info

  def close(self):
    pass
