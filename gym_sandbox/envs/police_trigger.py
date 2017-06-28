from collections import defaultdict
import gym
import gym.spaces
import numpy as np
import random

from .police_base import PoliceKillAllEnv, MOVE_ACTIONS
from gym_sandbox.envs.plot import balls_game_dashboard


class PoliceTriggerEnv(PoliceKillAllEnv):
    """
    Focus on Multi-Task
    Police must get close to thief, AND PULL TRIGGER!
    So there're 2 types of action: move and trigger, and they need co-operation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = gym.spaces.Discrete(len(MOVE_ACTIONS) + 1)

    def _step(self, action):
        """action space is (0~4), move is the same, 4 is pull trigger
        firstly check police pull trigger, then move
        """
        new_state = self.current_state.copy()
        if action == 4:  # pull trigger
            new_state = self.check_thief_caught(new_state)

        new_state = self.everybody_move(new_state, action)

        kill_num = len(self.current_state["thief"]) - len(new_state["thief"])
        self.current_is_caught = kill_num > 0

        self.last_state = self.current_state
        self.current_state = new_state
        self.current_action = action
        self.elapsed_steps += 1

        ob = self._trans_state(self.current_state)
        self.current_done = self._cal_done(self.current_state, kill_num)
        reward = self._cal_reward(kill_num, self.current_done)

        info = self._get_step_info()

        return ob, reward, self.current_done, info

