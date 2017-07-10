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
        return bool(kill_num or _pass_step_limit)

    def check_thief_caught(self, cur_state):
        # don't change state here! keep killed thief in state so that state shape is fixed
        thief_list = cur_state['thief']
        police_list = cur_state['police']

        kill_num = 0
        for _thief in thief_list:
            for _police in police_list:
                if self.calc_dist(_thief, _police) <= self.min_catch_dist:
                    kill_num = 1
                    break

        return cur_state, kill_num
