from collections import defaultdict
import gym
import gym.spaces
import numpy as np
import random

from .police_base import PoliceKillAllEnv, MOVE_ACTIONS
from gym_sandbox.envs.plot.balls_bokeh_server import BallsBokehServer


class PoliceBokehServerEnv(PoliceKillAllEnv):
    """
    Introduce new render using bokeh server embeded!
    This env only demo this new render, no other change on env logic.
    """
    def init_params(self, show_dashboard=True):
        """to control something after env is made"""
        self.game_dashboard = BallsBokehServer(
            self.map_size, self.team_size) if show_dashboard else None




