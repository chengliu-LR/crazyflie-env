"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Optional

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CrazyflieEnv(gym.Env):

    def __init__(self):
        self.state = 0
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        self.state += action
        reward = self.state >= 5
        print("reward: {}".format(reward))
        return np.array(self.state, dtype=np.float32), reward, None, {}

    def reset(self):
        self.state = 0
        return np.array(self.state, dtype=np.float32)
    
    def render(self, mode="human"):
        print(self.state)