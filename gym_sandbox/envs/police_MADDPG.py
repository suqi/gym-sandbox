from collections import defaultdict
import gym
import gym.spaces
import numpy as np
import random

from .police_kill_one import PoliceKillOneEnv
from gym_sandbox.envs.plot import balls_game_dashboard


class MAPoliceKillOneEnv(PoliceKillOneEnv):
    """
    This is a reproduction env of MADDPG
    DDPG: require env accept a continous action
    MA: step() accept multi action, and return multi result
    Attention: when any thief is caught, game end!  So this env is suitable for state of fixed postion list
    """
    pass
