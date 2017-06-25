from collections import defaultdict
import gym
import gym.spaces
import numpy as np
import random

from .balls_1vn import Balls1vnEnv
from gym_multiagent.envs.plot import balls_game_dashboard


class MABallsEnv(Balls1vnEnv):
    """A very simple balls game to demo MA algo
    Principle: don't introduce any complexity, focus on algo test!
    To make it simple, all state and game logic use int!
    Attention:
    1. in multi agent env, all function should return a list of actions/states/rewards
    2. the env has a fully observable internal state, while agent only has a partially observable state
    3. when any thief is caught, game end!  So this env is suitable for state of fixed postion list
    """
    def _cal_done(self, state, kill_num):
        is_exceed_max_round = self.round_count > self.MAX_ROUND_COUNT
        if is_exceed_max_round or kill_num:
            return True

        return False

    def check_thief_caught(self):
        # police win when any thief is caught
        thief_list = self.current_state['thief']
        police_list = self.current_state['police']

        for _thief in thief_list:
            for _police in police_list:
                if self.calc_dist(_thief, _police) <= self.MIN_CATCH_DIST:
                    return 1

        return 0

