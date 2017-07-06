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

    def police_move(self, police_list, police_actions):
        # Accept continous move action, which is more suitable for MADDPG
        police_actions = np.clip(police_actions, 0, np.pi * 2)

        police_new_loc = police_list.copy()
        police_speed = self.teams['police']['speed']
        for _i, _a in enumerate(police_actions):
            action_dir = np.array([np.cos(_a), np.sin(_a)])
            police_dir = action_dir * police_speed
            _p = police_list[_i]
            _p = (_p[0] + police_dir[0], _p[1] + police_dir[1])
            _p = self.ensure_inside(_p)
            police_new_loc[_i] = _p

        return police_new_loc

    def _trans_state(self, state):
        # now only support cord_list_unfixed, so must be KillOne mode!
        # Firstly absolute cord
        abs_ob = [_p for _p in state["police"]] + [_t for _t in state["thief"]]
        abs_ob = np.array(abs_ob)

        # now relative cord (make self position as (0,0))
        relative_state = [abs_ob.copy() - np.array(_p) for _p in state["police"]]

        return np.array(relative_state) / self.map_size
