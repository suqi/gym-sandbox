from collections import defaultdict
import gym
import gym.spaces
import numpy as np
import random

from .police_kill_all import Balls1vnEnv
from gym_sandbox.envs.plot import balls_game_dashboard


class RandomBallsEnv(Balls1vnEnv):
    """
    Focus to add more randomness into env
    Feature:
    1. Thief are incremently added into map in each step
    2. Each add batch has random num of thief
    3. Thief walk in a random way
    """
    pass


