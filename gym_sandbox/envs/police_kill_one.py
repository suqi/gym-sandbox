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
        return True if kill_num else False

    def check_thief_caught(self):
        # don't change state here!
        thief_list = self.current_state['thief']
        police_list = self.current_state['police']

        for _thief in thief_list:
            for _police in police_list:
                if self.calc_dist(_thief, _police) <= self.MIN_CATCH_DIST:
                    return 1

        return 0

