from collections import defaultdict
import gym
import gym.spaces
import numpy as np
import random

from .police_base import PoliceKillAllEnv
from gym_sandbox.envs.plot import balls_game_dashboard


class PoliceKillOneEnv(PoliceKillAllEnv):
    """A very simple balls game to demo MA algo
    Principle: don't introduce any complexity, focus on algo test!
    To make it simple, all state and game logic use int!
    Attention: when any thief is caught, game end!  So this env is suitable for state of fixed postion list
    """
    def _cal_done(self, state, kill_num):
        # police win when any thief is caught
        _pass_step_limit = self.elapsed_steps >= self.spec.max_episode_steps
        return (kill_num or _pass_step_limit)
