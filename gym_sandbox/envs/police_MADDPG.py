from collections import defaultdict
import gym
import gym.spaces
import numpy as np
import random

from .police_kill_one import PoliceKillOneEnv
from gym_sandbox.envs.plot import balls_game_dashboard


class PoliceMADDPGEnv(PoliceKillOneEnv):
    """
    This is a reproduction env of MADDPG
    Main difference with Single Agent Env is that: action is a list, and state is a list
    Observation: a list of relative cord of each police
    Action: a continous angle(0~2pi)
    DDPG: require env accept a continous action
    MA: step() accept multi action, and return multi result
    KillOne: when any thief is caught, game end!  So this env is suitable for state of fixed postion list
    """
    def __init__(self, **kwargs):
        # assert kwargs["agent_num"] > 1  # come on, this is MA!
        assert kwargs["state_format"] == "cord_list_unfixed"  # lower complexity, only fixed cord
        super().__init__(**kwargs)

    def calc_dist(self, pos1, pos2):
        """as action is a continous angle with any direction, now dist should be radius based"""
        _coords1 = np.array(pos1)
        _coords2 = np.array(pos2)

        # calc Euclidean Distance
        # alternative way: np.linalg.norm(_coords1 - _coords2)
        eucl_dist = np.sqrt(np.sum((_coords1 - _coords2) ** 2))
        return eucl_dist

    def _trans_state(self, state):
        # now only support cord_list_unfixed, so must be KillOne mode!
        # Firstly absolute cord
        abs_ob = [_p for _p in state["police"]] + [_t for _t in state["thief"]]
        abs_ob = np.array(abs_ob)

        # now relative cord (make self position as (0,0))
        # abs_state = [abs_ob.copy() for _ in range(self.agent_num)]
        relative_state = [abs_ob.copy() - np.array(_p) for _p in state["police"]]
        relative_state = [_.ravel() / self.map_size for _ in relative_state]

        return np.array(relative_state)

    # here MADDPG defaultly require a list of reward,
    # so that it allows an different reward for different agent
    # but for simplicity here we return a single same reward for all agent
    def _cal_reward(self, kill_num, is_done):

        reward = super()._cal_reward(kill_num, is_done)
        rewards = np.array([[reward] for _ in range(self.agent_num)]).astype(np.float)

        # if not is_done:
        #
        #     # if not done, add up dist reward to help boost
        #     thief_list = self.current_state['thief']
        #     police_list = self.current_state['police']
        #
        #     max_dist = self.map_size * np.sqrt(2) * len(thief_list)  # for normalize
        #
        #     for _i in range(self.agent_num):
        #         _p = police_list[_i]
        #         _all_dist = sum([self.calc_dist(_p, _t) for _t in thief_list])
        #         _dist_reward = 0.9 - _all_dist/max_dist
        #
        #         rewards[_i] += _dist_reward

        return rewards

    def _get_avail_new_loc(self, my_pos, my_speed):
        """as it get easier for multiagent to catch, we need to make thief stronger"""
        # stop or eight direction, it's A33 of [0,1,-1]
        new_loc = np.array([
            (0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1,-1)
        ]) * my_speed  + np.array(my_pos)

        new_loc = [self.ensure_inside(_l) for _l in new_loc]
        return new_loc

    # make thief smarter, keep away only from the nearest one
    def get_position_rating(self, my_new_pos, adversary_list):
        all_dist = [self.calc_dist(my_new_pos, _ad) for _ad in adversary_list]
        return min(all_dist)